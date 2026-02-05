from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.distributed as dist

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from einops import rearrange
import math
import os

# 适配 flash-attn 2.5.8 版本的导入
try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
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

    # 获取当前设备的local_rank
    if dist.is_initialized():
        local_rank = dist.get_rank() % torch.cuda.device_count()
    else:
        local_rank = int(os.getenv('LOCAL_RANK', 0))
    
    # 设置当前设备
    torch.cuda.set_device(local_rank)

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

    # 多卡训练时，确保数据在正确的设备上
    qkv = qkv.to(local_rank)
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(local_rank)

    if key_padding_mask is None:
        # 没有padding mask的情况，使用标准flash attention
        with torch.cuda.device(local_rank):
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
        
        # 确保数据在正确的设备上
        x_unpad = x_unpad.to(local_rank)
        cu_seqlens = cu_seqlens.to(local_rank)
        
        with torch.cuda.device(local_rank):
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
        
        # 将输出重新pad并reshape
        output_unpad = output_unpad.cpu()  # 先移到CPU进行pad操作，避免设备不一致
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
            ),
            "b s (h d) -> b s h d",
            h=nheads,
        )
        output = rearrange(output, "b s h d -> b s (h d)")
        output = output.to(local_rank)  # 移回GPU
    
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


# 多卡训练辅助函数
def setup_multigpu_training():
    """初始化多卡训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 初始化分布式训练
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # 设置当前设备
        torch.cuda.set_device(local_rank)
        
        print(f"Initialized distributed training: rank {rank}, world_size {world_size}, local_rank {local_rank}")
        return True
    return False


def cleanup_multigpu_training():
    """清理多卡训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


# 多卡训练的数据加载器包装器
def create_distributed_dataloader(dataset, batch_size, num_workers=4):
    """创建分布式数据加载器"""
    from torch.utils.data import DataLoader, DistributedSampler
    
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader