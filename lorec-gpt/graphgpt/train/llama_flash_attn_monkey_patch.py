from typing import List, Optional, Tuple

import torch
from torch import nn

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from einops import rearrange
import math

# # 适配 flash-attn 2.5.8 版本的导入
# try:
#     from flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
#     from flash_attn.bert_padding import unpad_input, pad_input
#     FLASH_ATTN_AVAILABLE = True
# except ImportError:
#     FLASH_ATTN_AVAILABLE = False
#     print("Flash attention not available, using standard attention")

# 尝试多种导入方式
FLASH_ATTN_AVAILABLE = False
try:
    # 方式1: flash_attn 2.x 版本
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTN_AVAILABLE = True
    print("Flash attention imported successfully from flash_attn.flash_attn_interface")
except ImportError:
    try:
        # 方式2: 直接导入
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
        from flash_attn import unpad_input, pad_input
        FLASH_ATTN_AVAILABLE = True
        print("Flash attention imported successfully from flash_attn")
    except ImportError:
        try:
            # 方式3: 旧版本导入
            from flash_attn.flash_attention import FlashAttention
            FLASH_ATTN_AVAILABLE = True
            print("Flash attention imported successfully (legacy version)")
        except ImportError:
            print("Flash attention not available, using standard attention")

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    assert past_key_value is None, "past_key_value is not supported"

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    
    assert not output_attentions, "output_attentions is not supported"
    assert not use_cache, "use_cache is not supported"

    if not FLASH_ATTN_AVAILABLE:
        # fallback to standard attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.unsqueeze(0).unsqueeze(0)
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), None, None

    # 准备flash attention输入
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
    
    key_padding_mask = attention_mask
    softmax_scale = (self.head_dim ** -0.5)

    if key_padding_mask is None:
        # 没有padding mask的情况，使用标准flash attention
        output = flash_attn_qkvpacked_func(
            qkv, 
            dropout_p=0.0, 
            softmax_scale=softmax_scale,
            causal=True,
            # window_size=(-1, -1),  # 无限上下文窗口
            # alibi_slopes=None,
            # deterministic=False,
            # return_attn_probs=False
        )
    else:
        # 有padding mask的情况，使用varlen flash attention
        nheads = qkv.shape[-2]
        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        x_unpad, indices, cu_seqlens, max_seqlen = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(
            x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
        )
        
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
            # window_size=(-1, -1),  # 无限上下文窗口
            # alibi_slopes=None,
            # deterministic=False,
            # return_attn_probs=False
        )
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
            ),
            "b s (h d) -> b s h d",
            h=nheads,
        )
        output = rearrange(output, "b s h d -> b s (h d)")
    
    return self.o_proj(output), None, None


def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        print(
            "Flash attention is only supported on Ampere or newer GPU architectures. "
            "Using standard attention instead."
        )
        return
    
    if not FLASH_ATTN_AVAILABLE:
        print("Flash attention is not available. Using standard attention.")
        return
        
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
    print("Successfully replaced Llama attention with Flash Attention")