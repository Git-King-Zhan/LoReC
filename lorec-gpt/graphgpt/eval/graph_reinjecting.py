import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
import time
import types

DEBUG_GRAPHREINJECT_LOG = False

# vanila GraphLlamaModel.forward
_original_graph_llama_model_forward = None


def _wrap_mlp_for_fusion(mlp):
    if hasattr(mlp, "_orig_forward"):
        return
    mlp._orig_forward = mlp.forward

    def fused_forward(self, x):
        out = self._orig_forward(x)
        layer_id = getattr(self, 'layer_idx', '?')
        # only trigger once
        if getattr(self, "apply_graphreinject", False) \
           and getattr(self, "adpt_sign", 0) == 1 \
           and getattr(self, "reinjecting_ratio", 0.0) > 0.0 \
           and getattr(self, "adpt_w1", None) is not None \
           and getattr(self, "adpt_w2", None) is not None:
            # (x @ [H,N]) @ [N,H] -> [B,T,H]
            adapter = torch.matmul(torch.matmul(x, self.adpt_w1), self.adpt_w2)
            # normalization
            scale = out.abs().mean() / (adapter.abs().mean() + 1e-8)
            fused = (1.0 - self.reinjecting_ratio) * out + self.reinjecting_ratio * scale * adapter
            if DEBUG_GRAPHREINJECT_LOG:
                try:
                    print(f"[GraphReinject] layer {layer_id} use fusion method (alpha={self.reinjecting_ratio:.4f}), x={tuple(x.shape)}, W1={tuple(self.adpt_w1.shape)}, W2={tuple(self.adpt_w2.shape)}")
                except Exception:
                    pass
            # reset flag
            self.adpt_sign = 0
            return fused
        else:
            if DEBUG_GRAPHREINJECT_LOG:
                try:
                    print(f"[GraphReinject] layer {layer_id} use vanila forward")
                except Exception:
                    pass
        return out

    mlp.forward = types.MethodType(fused_forward, mlp)


def apply_graphreinject(
        model,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        reinjecting_ratio: float  # α in [0,1]
    ):

    from graphgpt.model.GraphLlama import GraphLlamaModel
    global _original_graph_llama_model_forward

    # 1) every layer forward hook
    for li, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        _wrap_mlp_for_fusion(mlp)
        mlp.apply_graphreinject = True
        mlp.starting_layer = starting_layer
        mlp.ending_layer = ending_layer
        mlp.entropy_threshold = entropy_threshold
        mlp.reinjecting_ratio = reinjecting_ratio
        mlp.adpt_sign = 0
        mlp.graph_node_matrix = None   # G:[N,H]
        mlp.adpt_w1 = None             # [H,N]
        mlp.adpt_w2 = None             # [N,H]
        mlp.layer_idx = li

    # 2) lm_head for entropy calculation
    if hasattr(model, 'lm_head'):
        model.model.lm_head = model.lm_head

    # 3) package GraphLlamaModel.forward
    if _original_graph_llama_model_forward is None:
        _original_graph_llama_model_forward = GraphLlamaModel.forward

    def forward_wrapper(self, *args, **kwargs):
        disable_graphreinject = kwargs.pop('disable_graphreinject', False)
        graph_data = kwargs.get('graph_data', None)
        past_key_values = kwargs.get('past_key_values', None)

        # vanila forward
        if disable_graphreinject:
            try:
                print("[GraphReinject][Debug] disable_graphreinject=True，not use hook，use vanila forward", flush=True)
            except Exception:
                pass
            return _original_graph_llama_model_forward(self, *args, **kwargs)

        # get the graph node features G at the first step
        if graph_data is not None and past_key_values is None:
            graph_tower = self.get_graph_tower()
            if graph_tower is not None:
                with torch.no_grad():
                    data_item = graph_data[0] if isinstance(graph_data, list) else graph_data
                    # Skip graph processing if data_item is None (e.g., in text-only mode)
                    if data_item is not None:
                        node_output = graph_tower(data_item)  # [N,Dg]
                        # graph_projector
                        if hasattr(self, 'graph_projector') and self.graph_projector is not None:
                            projector = self.graph_projector
                            proj_dtype = projector.weight.dtype
                            Gh = projector(node_output.to(dtype=proj_dtype, device=projector.weight.device))  # [N,H_text]
                            G = Gh.to(self.device)
                        else:
                            G = node_output.to(self.device)
                        try:
                            print(f"[GraphReinject] G shape: {tuple(G.shape)}")
                        except Exception:
                            pass
                        for layer in self.layers:
                            if hasattr(layer.mlp, 'apply_graphreinject') and layer.mlp.apply_graphreinject:
                                layer.mlp.graph_node_matrix = G

        mlp0 = self.layers[0].mlp if len(self.layers) > 0 else None
        sL = getattr(mlp0, 'starting_layer', 0) if mlp0 is not None else 0
        eL = getattr(mlp0, 'ending_layer', -1) if mlp0 is not None else -1
        ratio = getattr(mlp0, 'reinjecting_ratio', 0.0) if mlp0 is not None else 0.0
        threshold = getattr(mlp0, 'entropy_threshold', 1.0) if mlp0 is not None else 1.0
        triggered_flag = {'v': False}
        hooks = []

        if sL <= eL and ratio > 0.0 and hasattr(self, 'lm_head') and getattr(self.layers[0].mlp, 'apply_graphreinject', False):
            for li in range(max(0, sL), min(eL+1, len(self.layers))):
                def _make_entropy_hook(idx):
                    def _hook(mod, inp, out):
                        if triggered_flag['v']:
                            return
                        hs = out[0] if isinstance(out, tuple) else out
                        logits_i = self.lm_head(self.norm(hs))[:, -1, :].float()
                        topk = torch.topk(logits_i, k=min(10, logits_i.shape[-1]))[0]
                        prob = F.softmax(topk, dim=-1)
                        entropy = torch.sum((-prob * torch.log(prob + 1e-9)) / np.log(10)).item()
                        if entropy > threshold:
                            if DEBUG_GRAPHINJECT_LOG:
                                try:
                                    print(f"[GraphReinject] layer {idx} trigger: {entropy:.4f}")
                                except Exception:
                                    pass
                            next_li = idx + 1
                            if next_li < len(self.layers):
                                mlp_next = self.layers[next_li].mlp
                                G = getattr(mlp_next, 'graph_node_matrix', None)
                                if G is not None:
                                    Gt = G.t()  # [H,N]
                                    # scale the adapter weights to match the MLP projection weights
                                    w1_scale = mlp_next.up_proj.weight.abs().mean() / (Gt.abs().mean() + 1e-8)
                                    w2_scale = mlp_next.down_proj.weight.abs().mean() / (G.abs().mean() + 1e-8)
                                    mlp_next.adpt_w1 = (w1_scale * Gt).to(dtype=mlp_next.up_proj.weight.dtype, device=mlp_next.up_proj.weight.device)
                                    mlp_next.adpt_w2 = (w2_scale * G).to(dtype=mlp_next.down_proj.weight.dtype, device=mlp_next.down_proj.weight.device)
                                    mlp_next.adpt_sign = 1  
                            triggered_flag['v'] = True  # only trigger once
                    return _hook
                h = self.layers[li].register_forward_hook(_make_entropy_hook(li))
                hooks.append(h)

        result = _original_graph_llama_model_forward(self, *args, **kwargs)

        # remove hooks
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        return result

    GraphLlamaModel.forward = forward_wrapper


def disable_graphreinject(model=None):
    global _original_graph_llama_model_forward
    from graphgpt.model.GraphLlama import GraphLlamaModel

    # GraphLlamaModel.forward
    if _original_graph_llama_model_forward is not None:
        GraphLlamaModel.forward = _original_graph_llama_model_forward
        _original_graph_llama_model_forward = None

    if model is not None and hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, '_orig_forward'):
                mlp.forward = mlp._orig_forward
                delattr(mlp, '_orig_forward')
            for attr in [
                'apply_graphreinject', 'starting_layer', 'ending_layer', 'entropy_threshold',
                'reinjecting_ratio', 'adpt_sign', 'graph_node_matrix', 'adpt_w1', 'adpt_w2'
            ]:
                if hasattr(mlp, attr):
                    try:
                        delattr(mlp, attr)
                    except Exception:
                        pass
