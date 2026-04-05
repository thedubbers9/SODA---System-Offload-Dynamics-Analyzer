from typing import Any, Dict, List
import torch


class DecodeGateHookTracker:
    """
    Collect routed-expert selections from MoE gate modules during generation.

    Output is now token/call-level:
      step -> layers -> layer_idx -> [records...]
    where each record corresponds to one gate forward call.
    """

    def __init__(self, top_k: int):
        self.top_k = top_k
        self.hooks = []
        self.current_step = None
        self.num_hooked_layers = 0

        # step -> {"meta": {...}, "layers": {layer_idx: [record, ...]}}
        self.step_layer_data: Dict[int, Dict[str, Any]] = {}

        # Per-step cursors
        # This tracks the forward call within a single step. (TOKEN WISE)
        self.step_call_index: Dict[int, int] = {}
        # This tracks the active token start per run.
        self.step_token_cursor: Dict[int, int] = {}

        # Active call context (shared across all layers for a single forward pass)
        self._active_call_index: Dict[int, int] = {}
        self._active_token_start: Dict[int, int] = {}
        self._active_token_count: Dict[int, int] = {}

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            if self.current_step is None:
                return

            step = self.current_step
            router_logits = output[0] if isinstance(output, tuple) else output
            if router_logits is None:
                return

            with torch.no_grad():
                if router_logits.dim() == 3:
                    # [batch, seq, experts]
                    batch_size, seq_len, num_experts = router_logits.shape
                    flat = router_logits.reshape(-1, num_experts).float()
                elif router_logits.dim() == 2:
                    # [tokens, experts]
                    rows, num_experts = router_logits.shape
                    batch_size, seq_len = 1, rows
                    flat = router_logits.float()
                else:
                    raise RuntimeError(
                        f"Unexpected gate output shape for layer {layer_idx}: {tuple(router_logits.shape)}"
                    )

                vals, idx = torch.topk(flat, k=min(self.top_k, flat.shape[-1]), dim=-1)
                rows = flat.shape[0]

                # New model forward call begins at first hooked layer.
                if layer_idx == 0 or step not in self._active_call_index:
                    self._active_call_index[step] = self.step_call_index.get(step, 0)
                    self._active_token_start[step] = self.step_token_cursor.get(step, 0)
                    self._active_token_count[step] = rows

                call_index = self._active_call_index[step]
                token_start = self._active_token_start[step]
                token_count = self._active_token_count[step]

                if step not in self.step_layer_data:
                    self.step_layer_data[step] = {
                        "meta": {"top_k": self.top_k, "num_hooked_layers": self.num_hooked_layers},
                        "layers": {},
                    }

                layers = self.step_layer_data[step]["layers"]
                if layer_idx not in layers:
                    layers[layer_idx] = []

                layers[layer_idx].append(
                    {
                        "call_index": call_index,
                        "token_start": token_start,
                        "token_count": token_count,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "router_shape": list(router_logits.shape),
                        "top_idx": idx.cpu().tolist(),
                        "top_vals": vals.cpu().tolist(),
                    }
                )

                # End of this forward call when last hooked layer finishes.
                if self.num_hooked_layers > 0 and layer_idx == (self.num_hooked_layers - 1):
                    self.step_token_cursor[step] = token_start + token_count
                    self.step_call_index[step] = call_index + 1
                    self._active_call_index.pop(step, None)
                    self._active_token_start.pop(step, None)
                    self._active_token_count.pop(step, None)

        return hook_fn

    def attach(self, model) -> int:
        layer_idx = 0
        for name, module in model.named_modules():
            if "shared_expert_gate" in name:
                continue
            if name.endswith("mlp.gate") or name.endswith("moe.gate"):
                self.hooks.append(module.register_forward_hook(self._make_hook(layer_idx)))
                # print(f"  Hooked gate {layer_idx}: {name}")
                layer_idx += 1

        self.num_hooked_layers = layer_idx
        return layer_idx

    def start_step(self, step: int):
        self.current_step = step
        self.step_layer_data[step] = {
            "meta": {"top_k": self.top_k, "num_hooked_layers": self.num_hooked_layers},
            "layers": {},
        }
        self.step_call_index[step] = 0
        self.step_token_cursor[step] = 0
        self._active_call_index.pop(step, None)
        self._active_token_start.pop(step, None)
        self._active_token_count.pop(step, None)

    def finish_step(self, step: int):
        data = self.step_layer_data.pop(step, {})
        self.current_step = None
        self.step_call_index.pop(step, None)
        self.step_token_cursor.pop(step, None)
        self._active_call_index.pop(step, None)
        self._active_token_start.pop(step, None)
        self._active_token_count.pop(step, None)
        return data

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.num_hooked_layers = 0