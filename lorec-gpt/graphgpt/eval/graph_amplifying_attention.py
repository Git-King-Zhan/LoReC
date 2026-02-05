import math
from typing import List

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention as HfLlamaAttention,
    apply_rotary_pos_emb,
)

try:
    from transformers.models.llama.modeling_llama import repeat_kv
except ImportError:
    try:
        from transformers.models.attention_utils import repeat_kv
    except ImportError:

        def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
            """
            This is the equivalent of torch.repeat_interleave(x, dim=0, repeats=n_rep).
            The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
            (batch, num_heads, seqlen, head_dim)
            """
            batch, num_key_value_heads, slen, head_dim = hidden_states.shape
            if n_rep == 1:
                return hidden_states
            hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
            return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

DEBUG_GRAPH_ATTN = False

_ORIG_LLAMA_ATTN_FORWARD = None
_ORIG_GRAPHELLAMA_MODEL_FORWARD = None


def _llama_attn_forward_with_boost(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
):

    bsz, q_len, _ = hidden_states.size()

    num_heads = getattr(self, 'num_heads', None)
    num_key_value_heads = getattr(self, 'num_key_value_heads', None)
    num_key_value_groups = getattr(self, 'num_key_value_groups', None)
    head_dim = getattr(self, 'head_dim', None)
    
    if num_heads is None:
        num_heads = self.config.num_attention_heads if hasattr(self, 'config') else None
    if num_key_value_heads is None:
        num_key_value_heads = getattr(self.config, 'num_key_value_heads', num_heads) if hasattr(self, 'config') else num_heads
    if num_key_value_groups is None:
        num_key_value_groups = num_heads // num_key_value_heads if (num_heads and num_key_value_heads) else 1
    if head_dim is None:
        head_dim = self.config.hidden_size // num_heads if (hasattr(self, 'config') and num_heads) else None

    # qkv projections
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # transform [B, T, H] -> [B, heads, T, head_dim]
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, -1, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, -1, num_key_value_heads, head_dim).transpose(1, 2)
    # print('q_heads:', num_heads) # 32
    # print('kv_heads:', num_key_value_heads) # 32

    # RoPE
    if position_ids is not None:
        seq_len_rope = int(position_ids.max().item()) + 1
        if seq_len_rope < query_states.shape[-2]:
            seq_len_rope = query_states.shape[-2]
    else:
        seq_len_rope = query_states.shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=seq_len_rope)

    if position_ids is not None:
        pos_max = int(position_ids.max().item())
        if cos.size(-2) <= pos_max:
            
            cos, sin = self.rotary_emb(value_states, seq_len=pos_max + 2)

    pos_ids = position_ids
    if pos_ids is not None:
        max_allowed = cos.size(-2) - 1
        if max_allowed >= 0:
            pos_ids = torch.clamp(pos_ids, min=0, max=max_allowed)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, pos_ids)

    # KV caching
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    present_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    # attention scores [B, heads, q_len, kv_len]
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)

    kv_seq_len = key_states.shape[-2]
    if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}"
        )
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

        attn_weights = torch.clamp(attn_weights, min=torch.finfo(attn_weights.dtype).min)

    # amplify graph-related attention logits before softmax
    if getattr(self, 'boost_graph_attn_sign', 0) == 1 and q_len > 0:
        graph_positions: List[int] = getattr(self, 'graph_token_positions', None)
        alpha = float(getattr(self, 'graph_attn_alpha', 0.0))
        if graph_positions and alpha != 0.0:
        
            cols = torch.as_tensor(graph_positions, device=attn_weights.device, dtype=torch.long)
            kv_seq_len = attn_weights.shape[-1]
            
            if cols.numel() > 0:
                valid = (cols >= 0) & (cols < kv_seq_len)
                cols = cols[valid]
            if cols.numel() > 0:
                region = attn_weights[:, :, -1, cols]
                # x <- x + |x| * alpha
                region = region + region.abs() * alpha
                attn_weights[:, :, -1, cols] = region

    # softmax
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    attn_output = torch.matmul(attn_weights, value_states) # [B, heads, q_len, head_dim]

    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1) # [B, q_len, hidden_size]
    attn_output = self.o_proj(attn_output)

    if output_attentions:
        return attn_output, attn_weights, present_key_value
    else:
        return attn_output, None, present_key_value


def apply_graph_attention_boost(model,
                                starting_layer: int,
                                ending_layer: int,
                                entropy_threshold: float,
                                alpha: float,
                                graph_patch_id: int = None,
                                graph_start_id: int = None,
                                graph_end_id: int = None):
    from graphgpt.model.GraphLlama import GraphLlamaModel

    global _ORIG_LLAMA_ATTN_FORWARD, _ORIG_GRAPHELLAMA_MODEL_FORWARD

    # 1) replace LlamaAttention.forward with boosted version
    if _ORIG_LLAMA_ATTN_FORWARD is None:
        _ORIG_LLAMA_ATTN_FORWARD = HfLlamaAttention.forward
        HfLlamaAttention.forward = _llama_attn_forward_with_boost

    # 1.5) entropy head
    entropy_head = getattr(model, 'lm_head', None)
    try:
        model.model._entropy_head = entropy_head
    except Exception:
        pass

    # 2) initialization self_attn attributes
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.boost_graph_attn_sign = 0
        attn.graph_attn_alpha = float(alpha)
        attn.graph_token_positions = None
        attn.layer_idx = li

    # 3) package GraphLlamaModel.forward
    if _ORIG_GRAPHELLAMA_MODEL_FORWARD is None:
        _ORIG_GRAPHELLAMA_MODEL_FORWARD = GraphLlamaModel.forward

    def _forward_wrapper(self, *args, **kwargs):
        disable_graph_attn = kwargs.pop('disable_graph_attn', False)
        input_ids = kwargs.get('input_ids', None)
        past_key_values = kwargs.get('past_key_values', None)

        # vanila forward
        if disable_graph_attn:
            return _ORIG_GRAPHELLAMA_MODEL_FORWARD(self, *args, **kwargs)

        shared_state = {'logged': False}
        for layer in self.layers:
            layer.self_attn._boost_log_state = shared_state

        # graph tokens valid positions
        if past_key_values is None and input_ids is not None:
            try:
                ids = input_ids[0].tolist()

                graph_cols = [i for i, t in enumerate(ids) if (graph_patch_id is not None and t == graph_patch_id)]

                if not graph_cols and (graph_start_id is not None) and (graph_end_id is not None):
                    starts = [i for i, t in enumerate(ids) if t == graph_start_id]
                    ends = [i for i, t in enumerate(ids) if t == graph_end_id]
                    for s in starts:
                        e = next((x for x in ends if x > s), None)
                        if e is not None and e - s > 1:
                            graph_cols.extend(list(range(s + 1, e)))

                graph_cols = sorted(set(graph_cols))
                for layer in self.layers:
                    layer.self_attn.graph_token_positions = graph_cols
                if DEBUG_GRAPH_ATTN:
                    try:
                        print(f"[GraphAttn] current token nums: {len(ids)}")
                    except Exception:
                        pass
                    print(f"[GraphAttn] graph token cols: {graph_cols}")
            except Exception:
                pass

        # calculate entropy and set boost flag via hooks
        triggered = {'v': False}
        hooks = []
        sL = starting_layer
        eL = ending_layer

        head_for_entropy = getattr(self, '_entropy_head', None)
        if sL <= eL and head_for_entropy is not None:
            try:
                shared_state_print = getattr(self, '_boost_log_state', None)
                if shared_state_print is None or not shared_state_print.get('printed_entropy_setup', False):
                    if shared_state_print is not None:
                        shared_state_print['printed_entropy_setup'] = True
            except Exception:
                pass
            for li in range(max(0, sL), min(eL+1, len(self.layers))):
                def _make_entropy_hook(idx):
                    def _hook(mod, inp, out):
                        if triggered['v']:
                            return
                        hs = out[0] if isinstance(out, tuple) else out
                        # logits & entropy（top-10）
                        logits = head_for_entropy(self.norm(hs))[:, -1, :].float()
                        topk = torch.topk(logits, k=min(10, logits.shape[-1]))[0]
                        prob = F.softmax(topk, dim=-1)
                        ent = torch.sum((-prob * torch.log(prob + 1e-9)) / math.log(10)).item()
                        if ent > entropy_threshold:
                            for lj in range(idx+1, min(eL+1, len(self.layers))):
                                self.layers[lj].self_attn.boost_graph_attn_sign = 1
                            triggered['v'] = True
                            if DEBUG_GRAPH_ATTN:
                                print(f"[GraphAttn] layer {idx} triggers，start attention amplification alpha={alpha}")
                    return _hook
                h = self.layers[li].register_forward_hook(_make_entropy_hook(li))
                hooks.append(h)

        # vanila forward
        out = _ORIG_GRAPHELLAMA_MODEL_FORWARD(self, *args, **kwargs)

        # clean hooks and reset flags
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        for layer in self.layers:
            layer.self_attn.boost_graph_attn_sign = 0

        return out

    GraphLlamaModel.forward = _forward_wrapper
