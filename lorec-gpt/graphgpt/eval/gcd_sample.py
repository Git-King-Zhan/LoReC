"""
Graph Contrastive Decoding (GCD) Sample Function
This module uses the formula:
    diffs = (1 + cd_alpha + cg_beta) * next_token_logits 
            - cd_alpha * next_token_logits_cd 
            - cg_beta * next_token_logits_cg

where:
    - next_token_logits: logits from the original forward pass with text and graph tokens
    - next_token_logits_cd: logits from forward pass with only text tokens (no graph tokens)
    - next_token_logits_cg: logits from forward pass with augmented graph tokens
    - cd_alpha: hyperparameter for text-only contrast
    - cg_beta: hyperparameter for augmented graph contrast
"""

import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput, SampleDecoderOnlyOutput, SampleEncoderDecoderOutput


DEBUG_GCD_LOG = False


def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    """
    GCD-enhanced sampling function that performs contrastive decoding with graph augmentation.
    
    Args:
        self: The model instance
        input_ids: Input token IDs
        logits_processor: Logits processors to apply
        stopping_criteria: Stopping criteria for generation
        logits_warper: Logits warpers (e.g., temperature, top-k)
        max_length: Maximum generation length
        pad_token_id: Padding token ID
        eos_token_id: End-of-sequence token ID(s)
        output_attentions: Whether to output attention weights
        output_hidden_states: Whether to output hidden states
        output_scores: Whether to output token scores
        return_dict_in_generate: Whether to return a dictionary
        synced_gpus: Whether to synchronize across GPUs
        streamer: Token streamer for real-time output
        **model_kwargs: Additional model arguments including:
            - graph_data: Graph data for the model
            - graph_data_cd: Graph data for text-only forward pass (optional)
            - graph_data_cg: Graph data for augmented graph forward pass (optional)
            - cd_alpha: Hyperparameter for text-only contrast (default: 0.5)
            - cg_beta: Hyperparameter for augmented graph contrast (default: 1.0)
    
    Returns:
        Generated sequences or SampleOutput with scores and hidden states
    """
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteria(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # Define a global clamp limit for numerical stability (used across the function)
    max_logit = 100.0

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    
    # Copy model_kwargs for CD (text-only) and CG (augmented graph) forward passes
    model_kwargs_cd = model_kwargs.copy()
    model_kwargs_cg = model_kwargs.copy()
    
    # helper: sanitize logits to avoid nan/inf and clamp to finite range
    # def _sanitize_logits(t: torch.Tensor, limit: float = 100.0) -> torch.Tensor:
    #     if t is None:
    #         return t
    #     t = torch.nan_to_num(t, nan=0.0, posinf=limit, neginf=-limit)
    #     t = torch.clamp(t, min=-limit, max=limit)
    #     return t

    # auto-regressive generation
    step_idx = 0
    while True:
        step_idx += 1
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # ============ GCD: Graph Contrastive Decoding ============
        use_gcd = ("graph_data_cg" in model_kwargs) or ("graph_data_cd" in model_kwargs)
        
        if use_gcd:
            if DEBUG_GCD_LOG:
                try:
                    print(f"[GCD][Step {step_idx}] use_gcd=True", flush=True)
                except Exception:
                    pass
            # Get hyperparameters
            cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 0.5
            cg_beta = model_kwargs.get("cg_beta") if model_kwargs.get("cg_beta") is not None else 0.1
            cut_para = model_kwargs.get("cut_para", None)
            if cut_para is None:
                cut_para = getattr(self, 'gcd_cut_para', None)
            if cut_para is None and hasattr(self, 'generation_config'):
                cut_para = getattr(self.generation_config, 'cut_para', None)
            if cut_para is None:
                cut_para = 1.0
            
            # Prepare explicit kwargs for CD/CG branches
            # CD: enforce text-only by removing graph input
            model_kwargs_cd['graph_data'] = None
            model_kwargs_cd.pop('graph_data_cd', None)
            model_kwargs_cd.pop('graph_data_cg', None)

            model_kwargs_cd['disable_graphvr'] = True
            model_kwargs_cd['disable_graph_attn'] = True
            
            # CG: use augmented graph if provided, else fallback to original graph
            using_aug_graph = model_kwargs.get('graph_data_cg') is not None
            if using_aug_graph:
                model_kwargs_cg['graph_data'] = model_kwargs['graph_data_cg']
            else:
                model_kwargs_cg['graph_data'] = model_kwargs.get('graph_data', None)
            model_kwargs_cg.pop('graph_data_cd', None)
            model_kwargs_cg.pop('graph_data_cg', None)

            model_kwargs_cg['disable_graphvr'] = True
            model_kwargs_cg['disable_graph_attn'] = True
            if DEBUG_GCD_LOG:
                try:
                    gd = model_kwargs_cg.get('graph_data', None)
                    n_nodes = None
                    n_edges = None
                    if gd is not None:
                        if hasattr(gd, 'graph_node') and gd.graph_node is not None:
                            n_nodes = gd.graph_node.shape[0]
                        elif hasattr(gd, 'x') and gd.x is not None:
                            n_nodes = gd.x.shape[0]
                        if hasattr(gd, 'edge_index') and gd.edge_index is not None:
                            n_edges = gd.edge_index.size(1)
                    print(f"[GCD][Step {step_idx}][CG] using_aug_graph={using_aug_graph}, disable_graphvr/attn=True/True, nodes={n_nodes}, edges={n_edges}", flush=True)
                except Exception:
                    pass
            
            # Forward pass with text-only (no graph tokens)
            if "graph_data_cd" in model_kwargs:
                # Build text-only input by removing <g_patch>/<g_start>/<g_end>
                input_ids_cd = input_ids
                try:
                    base_model = self.get_model() if hasattr(self, 'get_model') else getattr(self, 'model', None)
                    graph_tower = base_model.get_graph_tower() if base_model is not None and hasattr(base_model, 'get_graph_tower') else None
                    graph_cfg = getattr(graph_tower, 'config', None)
                    patch_id = getattr(graph_cfg, 'graph_patch_token', None)
                    start_id = getattr(graph_cfg, 'graph_start_token', None)
                    end_id = getattr(graph_cfg, 'graph_end_token', None)
                    ids_to_remove = [x for x in [patch_id, start_id, end_id] if isinstance(x, int)]
                    if input_ids.dim() == 2 and input_ids.size(0) == 1 and len(ids_to_remove) > 0:
                        ids0 = input_ids[0]
                        keep_mask = torch.ones_like(ids0, dtype=torch.bool)
                        for rid in ids_to_remove:
                            keep_mask = keep_mask & (ids0 != rid)
                        removed = (ids0.numel() - keep_mask.sum().item())
                        # avoid none input
                        if keep_mask.sum() > 0:
                            input_ids_cd = ids0[keep_mask].unsqueeze(0)
                        if DEBUG_GCD_LOG:
                            try:
                                print(f"[GCD][Step {step_idx}][CD] text-only: disable_graphvr/attn=True/True, remove_ids={ids_to_remove}, before_len={ids0.numel()}, after_len={int(keep_mask.sum().item())}, removed={removed}", flush=True)
                            except Exception:
                                pass
                except Exception:

                    input_ids_cd = input_ids
                # CD：remove attention_mask
                model_kwargs_cd.pop('attention_mask', None)
                # Important: do not use past_key_values for CD branch
                model_kwargs_cd.pop('past_key_values', None)
                model_kwargs_cd.pop('position_ids', None)
                model_kwargs_cd['use_cache'] = False
                model_inputs_cd = self.prepare_inputs_for_generation(input_ids_cd, **model_kwargs_cd)
                outputs_cd = self(
                    **model_inputs_cd,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                next_token_logits_cd = outputs_cd.logits[:, -1, :]
            else:
                # If no text-only graph data provided, use the original logits
                next_token_logits_cd = next_token_logits.clone()
            
            # Forward pass with augmented graph tokens
            if model_kwargs.get("graph_data_cg") is not None:
                model_inputs_cg = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cg)
                outputs_cg = self(
                    **model_inputs_cg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                next_token_logits_cg = outputs_cg.logits[:, -1, :]
            else:
                # If no augmented graph data provided, use the original logits
                next_token_logits_cg = next_token_logits.clone()
            
            # Apply GCD formula:
            # diffs = (1 + cd_alpha + cg_beta) * next_token_logits 
            #         - cd_alpha * next_token_logits_cd 
            #         - cg_beta * next_token_logits_cg
            diffs = (1 + cd_alpha + cg_beta) * next_token_logits - cd_alpha * next_token_logits_cd - cg_beta * next_token_logits_cg

            if DEBUG_GCD_LOG:
                try:
                    print(f"[GCD][Step {step_idx}] params: cd_alpha={float(cd_alpha):.6g}, cg_beta={float(cg_beta):.6g}", flush=True)
                    if next_token_logits.dim() == 2:
                        B = next_token_logits.size(0)
                        V = next_token_logits.size(1)
                        k = int(min(10, V))
                        if B >= 1:
                            bi = 0

                            # 1) vanila logits top-k
                            topv_orig, topi_orig = torch.topk(next_token_logits[bi], k=k)
                            _vals_orig = topv_orig.detach().to(torch.float32).cpu().tolist()
                            _idx_orig = topi_orig.detach().cpu().tolist()
                            print(
                                f"[GCD][Step {step_idx}] orig logits top{k} (sample {bi}) token_id:logit => "
                                + ", ".join([f"{ti}:{v:.6g}" for ti, v in zip(_idx_orig, _vals_orig)]),
                                flush=True,
                            )

                            # 2) (vanila - cd) logits
                            delta_cd = (next_token_logits[bi] - next_token_logits_cd[bi])
                            sel_delta_cd = delta_cd[topi_orig]
                            _vals_dcd = sel_delta_cd.detach().to(torch.float32).cpu().tolist()
                            print(
                                f"[GCD][Step {step_idx}] (orig - text_only) on orig top{k} (sample {bi}) token_id:delta => "
                                + ", ".join([f"{ti}:{v:.6g}" for ti, v in zip(_idx_orig, _vals_dcd)]),
                                flush=True,
                            )

                            # 3) (vanila - cg) logits
                            delta_cg = (next_token_logits[bi] - next_token_logits_cg[bi])
                            sel_delta_cg = delta_cg[topi_orig]
                            _vals_dcg = sel_delta_cg.detach().to(torch.float32).cpu().tolist()
                            print(
                                f"[GCD][Step {step_idx}] (orig - aug_graph) on orig top{k} (sample {bi}) token_id:delta => "
                                + ", ".join([f"{ti}:{v:.6g}" for ti, v in zip(_idx_orig, _vals_dcg)]),
                                flush=True,
                            )

                            # 4) max/mean/std）
                            try:
                                dcd_abs = delta_cd.abs()
                                dcg_abs = delta_cg.abs()
                                print(
                                    f"[GCD][Step {step_idx}] |orig-text_only| stats (sample {bi}): "
                                    f"max={float(dcd_abs.max().item()):.6g}, mean={float(dcd_abs.mean().item()):.6g}, std={float(dcd_abs.std().item()):.6g}",
                                    flush=True,
                                )
                                print(
                                    f"[GCD][Step {step_idx}] |orig-aug_graph| stats (sample {bi}): "
                                    f"max={float(dcg_abs.max().item()):.6g}, mean={float(dcg_abs.mean().item()):.6g}, std={float(dcg_abs.std().item()):.6g}",
                                    flush=True,
                                )
                            except Exception:
                                pass
                except Exception:
                    pass

            # Ensure numerical stability: clamp extreme values
            # Prevent inf/nan in logits by clamping to a reasonable range
            # max_logit = 100.0  # Standard clamp value for numerical stability
            # diffs = torch.clamp(diffs, min=-max_logit, max=max_logit)
            
            # Optional: Apply adaptive plausibility constraints (similar to VCD)
            # This prevents the model from selecting tokens with very low probability
            # Use a safer approach: apply a mask with a large negative value instead of -inf
            if cut_para > 0:
                # Only apply cutoff if cut_para is positive
                log_cut = torch.log(torch.tensor(max(cut_para, 1e-10), device=next_token_logits.device, dtype=next_token_logits.dtype))
                if DEBUG_GCD_LOG:
                    try:
                        print(f"[GCD][Step {step_idx}] cutoff: using cut_para={float(cut_para):.6g}, log_cut={float(log_cut.item()):.6g}", flush=True)
                    except Exception:
                        pass
                # log_cut = torch.log(torch.tensor(1e-10, device=next_token_logits.device, dtype=next_token_logits.dtype))
                cutoff = log_cut + next_token_logits.max(dim=-1, keepdim=True).values  # use base logits (text+graph)
                # Use a large negative value instead of -inf to avoid NaN in softmax

                gcd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            else:
                gcd_logits = diffs
            
            # Ensure gcd_logits doesn't contain inf or nan before processing
            # gcd_logits = torch.clamp(gcd_logits, min=-max_logit, max=max_logit)
            
            # Apply logits processors and warpers
            gcd_logits = logits_processor(input_ids, gcd_logits)
            gcd_logits = logits_warper(input_ids, gcd_logits)

            # Sanitize infinities introduced by processors/warpers (e.g., top-k/top-p uses -inf)
            # if torch.isinf(gcd_logits).any():
            #     pos_inf = torch.isposinf(gcd_logits)
            #     neg_inf = torch.isneginf(gcd_logits)
            #     if pos_inf.any():
            #         gcd_logits = torch.where(
            #             pos_inf,
            #             torch.tensor(max_logit, device=gcd_logits.device, dtype=gcd_logits.dtype),
            #             gcd_logits,
            #         )
            #     if neg_inf.any():
            #         gcd_logits = torch.where(
            #             neg_inf,
            #             torch.tensor(-max_logit, device=gcd_logits.device, dtype=gcd_logits.dtype),
            #             gcd_logits,
            #         )
            # Clamp again to be extra safe in fp16
            # gcd_logits = torch.clamp(gcd_logits, min=-max_logit, max=max_logit)

            # Final safety: if still NaN, fallback to standard logits path
            if torch.isnan(gcd_logits).any():
                if DEBUG_GCD_LOG:
                    try:
                        print(f"[GCD][Step {step_idx}] WARNING: gcd_logits has NaN after processing; fallback to base logits", flush=True)
                    except Exception:
                        pass
                next_token_scores = logits_processor(input_ids, next_token_logits)
                next_token_scores = logits_warper(input_ids, next_token_scores)
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                # update and continue to next step
                if eos_token_id is not None:
                    if pad_token_id is None:
                        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if streamer is not None:
                    streamer.put(next_tokens.cpu())
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )
                if use_gcd:
                    if model_kwargs.get("graph_data_cd") is not None:
                        model_kwargs_cd = self._update_model_kwargs_for_generation(
                            outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
                        )
                    if model_kwargs.get("graph_data_cg") is not None:
                        model_kwargs_cg = self._update_model_kwargs_for_generation(
                            outputs_cg, model_kwargs_cg, is_encoder_decoder=self.config.is_encoder_decoder
                        )
                # check stopping
                if eos_token_id_tensor is not None:
                    unfinished_sequences = unfinished_sequences.mul(
                        next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                    )
                    if unfinished_sequences.max() == 0:
                        this_peer_finished = True
                if stopping_criteria(input_ids, scores):
                    this_peer_finished = True
                if this_peer_finished and not synced_gpus:
                    break
                else:
                    continue

            next_token_scores = gcd_logits
            gcd_probs = nn.functional.softmax(gcd_logits, dim=-1)
            
            # Verify probabilities are valid before multinomial sampling
            if torch.isnan(gcd_probs).any() or torch.isinf(gcd_probs).any() or (gcd_probs < 0).any():
                if DEBUG_GCD_LOG:
                    try:
                        print(f"[GCD][Step {step_idx}] ERROR: gcd_probs contains nan/inf/negative values, using standard sampling", flush=True)
                    except Exception:
                        pass
                # Fallback to standard sampling
                next_token_scores = logits_processor(input_ids, next_token_logits)
                next_token_scores = logits_warper(input_ids, next_token_scores)
                gcd_probs = nn.functional.softmax(next_token_scores, dim=-1)
            
            next_tokens = torch.multinomial(gcd_probs, num_samples=1).squeeze(1)
        else:
            # Standard sampling without GCD
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        
        # Update model_kwargs_cd and model_kwargs_cg for next iteration
        if use_gcd:
            if model_kwargs.get("graph_data_cd") is not None:
                model_kwargs_cd = self._update_model_kwargs_for_generation(
                    outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
                )
            if model_kwargs.get("graph_data_cg") is not None:
                model_kwargs_cg = self._update_model_kwargs_for_generation(
                    outputs_cg, model_kwargs_cg, is_encoder_decoder=self.config.is_encoder_decoder
                )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids


def evolve_gcd_sampling():
    """
    Monkey-patch the model's sample function to use GCD-enhanced sampling.
    This function should be called before generation to enable GCD.
    """
    transformers.generation.utils.GenerationMixin.sample = sample
    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = sample

