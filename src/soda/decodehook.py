from typing import Dict, List
import torch
from torch.nn import Module



class DecodeGateHookTracker:
    """
    Collect routed-expert selections from MoE gate modules during decode only.

    We deliberately avoid output_router_logits=True in the model forward path,
    because that codepath can fail during cached decode for Qwen2-MoE.
    Instead, we read the gate outputs directly via forward hooks.
    """

    def __init__(self, top_k: int):
        self.top_k = top_k
        self.hooks = []
        self.current_step = None
        self.step_layer_data: Dict[int, Dict[int, Dict[str, List[List[float]]]]] = {}

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            if self.current_step is None:
                return

            router_logits = output[0] if isinstance(output, tuple) else output
            if router_logits is None:
                return

            with torch.no_grad():
                if router_logits.dim() == 3:
                    flat = router_logits.reshape(-1, router_logits.shape[-1]).float()
                elif router_logits.dim() == 2:
                    flat = router_logits.float()
                else:
                    raise RuntimeError(
                        f"Unexpected gate output shape for layer {layer_idx}: {tuple(router_logits.shape)}"
                    )

                vals, idx = torch.topk(flat, k=min(self.top_k, flat.shape[-1]), dim=-1)

                if self.current_step not in self.step_layer_data:
                    self.step_layer_data[self.current_step] = {}

                self.step_layer_data[self.current_step][layer_idx] = {
                    "top_idx": idx.cpu().tolist(),
                    "top_vals": vals.cpu().tolist(),
                }

        return hook_fn

    def attach(self, model) -> int:
        layer_idx = 0
        for name, module in model.named_modules():
            if "shared_expert_gate" in name:
                continue
            if name.endswith("mlp.gate") or name.endswith("moe.gate"):
                self.hooks.append(module.register_forward_hook(self._make_hook(layer_idx)))
                print(f"  Hooked gate {layer_idx}: {name}")
                layer_idx += 1
        return layer_idx

    def start_step(self, step: int):
        self.current_step = step
        self.step_layer_data[step] = {}

    def finish_step(self, step: int):
        data = self.step_layer_data.pop(step, {})
        self.current_step = None
        return data

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []