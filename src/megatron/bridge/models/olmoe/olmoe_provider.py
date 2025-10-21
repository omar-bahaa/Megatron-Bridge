# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from dataclasses import dataclass, fields
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.attention import SelfAttention as MCoreSelfAttention
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.gpt_provider import GPTModelProvider, default_layer_spec

try:
    from megatron.core.process_groups_config import ProcessGroupCollection
except ImportError:  # pragma: no cover - compatibility fallback
    ProcessGroupCollection = Any  # type: ignore

try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None


def olmoe_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Layer spec for OlMoE models."""
    layer_spec = default_layer_spec(config)
    layer_spec.submodules.self_attention.module = OLMoESelfAttention
    return layer_spec


def olmoe_transformer_layer_spec(
    config: TransformerConfig, vp_stage: Optional[int] = None
) -> ModuleSpec:
    """Return the OlMoE transformer layer spec starting from a core config."""

    provider_kwargs = {
        field.name: getattr(config, field.name)
        for field in fields(OlMoEModelProvider)
        if field.init and hasattr(config, field.name)
    }
    provider = OlMoEModelProvider(**provider_kwargs)

    transformer_layer_spec = provider.transformer_layer_spec
    if isinstance(transformer_layer_spec, ModuleSpec):
        return transformer_layer_spec

    signature = inspect.signature(transformer_layer_spec)
    if "vp_stage" in signature.parameters:
        return transformer_layer_spec(provider, vp_stage=vp_stage)
    return transformer_layer_spec(provider)


@dataclass
class OlMoEModelProvider(GPTModelProvider):
    """Base provider for OlMoE Models."""

    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTModelProvider"], ModuleSpec]] = olmoe_layer_spec
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    seq_length: int = 4096
    init_method_std: int = 0.02
    hidden_dropout: float = 0.0
    vocab_size: int = 50304
    share_embeddings_and_output_weights: Optional[bool] = False
    layernorm_epsilon: float = 1e-5
    autocast_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.float32
    bf16: bool = False

    # Model specific parameters
    num_layers: int = 16
    hidden_size: int = 2048
    ffn_hidden_size: int = 1024
    moe_ffn_hidden_size: int = 1024
    kv_channels: int = 2048 // 16

    # Attention
    num_query_groups: int = 16
    num_attention_heads: int = 16
    attention_dropout: float = 0.0
    qk_layernorm: bool = True
    # RoPE
    position_embedding_type: str = "rope"
    rotary_base: float = 10000.0
    # MoE specific parameters
    num_moe_experts: int = 64
    moe_router_topk: int = 8
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_aux_loss_coeff: float = 1e-2
    moe_router_pre_softmax: bool = True
    moe_grouped_gemm: bool = True
    moe_router_score_function: str = "softmax"
    moe_permute_fusion: bool = True
    moe_router_dtype: str = "fp32"
    # Optimization
    persist_layer_norm: bool = True


class OLMoESelfAttention(MCoreSelfAttention):
    """Custom self-attention module for OlMoE models."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        model_comm_pgs: Optional[Any] = None,
        **kwargs,
    ):
        super_signature = inspect.signature(super().__init__)
        super_kwargs = {
            "config": config,
            "submodules": submodules,
            "layer_number": layer_number,
            "attn_mask_type": attn_mask_type,
            "cp_comm_type": cp_comm_type,
        }

        if "model_comm_pgs" in super_signature.parameters:
            super_kwargs["model_comm_pgs"] = model_comm_pgs if model_comm_pgs is not None else pg_collection
        if "pg_collection" in super_signature.parameters and "pg_collection" not in super_kwargs:
            super_kwargs["pg_collection"] = pg_collection if pg_collection is not None else model_comm_pgs

        # Forward through any additional kwargs supported by upstream base class.
        for key in list(kwargs.keys()):
            if key in super_signature.parameters and key not in super_kwargs:
                super_kwargs[key] = kwargs.pop(key)

        super().__init__(**super_kwargs)

        # OlMoE applies Q/K layernorms across all heads in the local tensor-parallel shard.
        hidden_size_for_norm = (
            self.hidden_size_per_attention_head * self.num_attention_heads_per_partition
        )

        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=hidden_size_for_norm,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.k_layernorm = build_module(
            submodules.k_layernorm,
            hidden_size=hidden_size_for_norm,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self._register_load_state_dict_pre_hook(self._layernorm_load_state_dict_pre_hook, with_module=True)

    def _layernorm_load_state_dict_pre_hook(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Slice global Q/K layernorm weights when loading checkpoints saved before TP sharding."""

        def _maybe_slice(param_name):
            key = prefix + param_name
            if key not in state_dict:
                return
            tensor = state_dict[key]
            module_attr_name, param_attr_name = param_name.split(".")
            target_module = getattr(module, module_attr_name)
            target_param = getattr(target_module, param_attr_name)
            expected_shape = target_param.shape
            if tensor.shape == expected_shape:
                return
            head_dim = self.hidden_size_per_attention_head
            total_heads = self.config.num_attention_heads
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            if tp_size <= 0 or tensor.numel() != total_heads * head_dim:
                return
            tensor = tensor.view(total_heads, head_dim)
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            start = tp_rank * self.num_attention_heads_per_partition
            end = start + self.num_attention_heads_per_partition
            tensor = tensor[start:end].reshape(-1).to(dtype=target_param.dtype).clone()
            state_dict[key] = tensor

        _maybe_slice("q_layernorm.weight")
        _maybe_slice("k_layernorm.weight")

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ):
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        for suffix in (
            "q_layernorm.weight",
            "k_layernorm.weight",
        ):
            key = f"{prefix}{suffix}"
            if key in sharded_state_dict:
                sharded_state_dict[key].allow_shape_mismatch = True

        return sharded_state_dict

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:
            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        # Main difference between Mcore QK Layernorm
        query = query.reshape(query.size(0), query.size(1), -1)
        key = key.reshape(key.size(0), key.size(1), -1)
        query = self.q_layernorm(query)
        key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        query = query.view(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        key = key.view(key.size(0), key.size(1), -1, self.hidden_size_per_attention_head)

        return query, key, value
