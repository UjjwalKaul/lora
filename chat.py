"""
Interactive LoRA Playground: Hot-Swapping, Merging, and Side-by-Side Comparison!

Commands inside the chat:
  /compare <prompt>  - Run BOTH the Base Model and the Pirate LoRA side-by-side.
  /toggle            - Hot-swap the LoRA adapter ON/OFF instantly.
  /merge             - Permanently bake Delta W into W_0 (zero-latency deployment).
  exit / quit        - Exit the playground.
"""

import math
import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# 1. LORALINEAR WITH HOT-SWAPPING & MERGING SUPPORT
# -----------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = lora_alpha / r
        
        # Controls whether the clip-on adapter is active
        self.enabled = True
        self.merged = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, r, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        # If LoRA is disabled or already merged into the base weights, skip LoRA branch
        if not self.enabled or self.merged:
            return base_out

        # The LoRA low-rank branch
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def merge(self):
        """Bakes Delta W permanently into the base weights W_0."""
        if not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Subtracts Delta W from base weights W_0 if we ever want to revert."""
        if self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
            self.merged = False


# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def apply_and_load_lora(model, adapter_path: str, r=8, alpha=16.0):
    """Wraps q_proj and v_proj, then loads the trained A and B matrices."""
    device = next(model.parameters()).device
    adapter_state = torch.load(adapter_path, map_location=device, weights_only=True)
    
    lora_modules = []
    for idx, layer in enumerate(model.model.layers):
        # Wrap q_proj
        q_lora = LoRALinear(layer.self_attn.q_proj, r=r, lora_alpha=alpha)
        q_lora.lora_A.data.copy_(adapter_state[f"layer.{idx}.q_proj.A"])
        q_lora.lora_B.data.copy_(adapter_state[f"layer.{idx}.q_proj.B"])
        layer.self_attn.q_proj = q_lora

        # Wrap v_proj
        v_lora = LoRALinear(layer.self_attn.v_proj, r=r, lora_alpha=alpha)
        v_lora.lora_A.data.copy_(adapter_state[f"layer.{idx}.v_proj.A"])
        v_lora.lora_B.data.copy_(adapter_state[f"layer.{idx}.v_proj.B"])
        layer.self_attn.v_proj = v_lora

        lora_modules.extend([q_lora, v_lora])

    return lora_modules


def set_lora_state(lora_modules, enabled: bool):
    for m in lora_modules:
        m.enabled = enabled


def merge_lora_weights(lora_modules):
    for m in lora_modules:
        m.merge()


def generate(model, tokenizer, prompt: str, device: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=60, 
            do_sample=True, 
            temperature=0.7, 
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# -----------------------------------------------------------------------------
# 3. INTERACTIVE CONSOLE
# -----------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] Initializing on: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"[*] Loading base model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)

    adapter_path = "pirate_adapter.pt"
    if not os.path.exists(adapter_path):
        print(f"[!] Error: {adapter_path} not found. Run train_pirate_lora.py first!")
        return

    print(f"[*] Attaching and loading LoRA adapter from '{adapter_path}'...")
    lora_modules = apply_and_load_lora(model, adapter_path, r=8, alpha=16.0)

    lora_enabled = True
    is_merged = False

    print("\n" + "=" * 65)
    print("         LORA PLAYGROUND: HOT-SWAP, MERGE & CHAT         ")
    print("=" * 65)
    print("Commands:")
    print("  /compare <text>  -> Compare Base vs Pirate response side-by-side")
    print("  /toggle          -> Switch LoRA Adapter ON/OFF live")
    print("  /merge           -> Permanently fuse LoRA into base weights (W_0)")
    print("  exit / quit      -> Exit the chat")
    print("=" * 65)

    while True:
        status_label = "PERMANENTLY MERGED" if is_merged else ("LORA: [ON (Pirate)]" if lora_enabled else "LORA: [OFF (Base Model)]")
        try:
            user_input = input(f"\n[{status_label}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting playground.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Fair winds to ye! Goodbye!")
            break

        # Command: /toggle
        if user_input.lower() == "/toggle":
            if is_merged:
                print("[!] The adapter is already permanently merged into W_0! Cannot toggle.")
            else:
                lora_enabled = not lora_enabled
                set_lora_state(lora_modules, lora_enabled)
                new_state = "PIRATE MODE (ON)" if lora_enabled else "BASE ASSISTANT MODE (OFF)"
                print(f"[LoRA Hot-Swap] Switched adapter to: {new_state}")
            continue

        # Command: /merge
        if user_input.lower() == "/merge":
            if is_merged:
                print("[!] Already merged!")
            else:
                merge_lora_weights(lora_modules)
                is_merged = True
                print("[LoRA Bake-In] Successfully merged Delta W = (alpha/r)*B@A into W_0!")
                print("               The LoRA branch is now permanently folded into the base weights.")
            continue

        # Command: /compare
        if user_input.lower().startswith("/compare"):
            prompt = user_input[len("/compare"):].strip()
            if not prompt:
                print("[!] Please provide a prompt, e.g. /compare What is Python?")
                continue
            
            print(f"\n[Prompt]: {prompt}")
            print("-" * 55)
            # 1. Base Model output
            set_lora_state(lora_modules, False)
            base_out = generate(model, tokenizer, prompt, device)
            print(f"🤖 [BASE ASSISTANT]:\n{base_out}\n")

            # 2. Pirate LoRA output
            set_lora_state(lora_modules, True)
            pirate_out = generate(model, tokenizer, prompt, device)
            print(f"🏴‍☠️ [PIRATE LORA]:\n{pirate_out}")
            print("-" * 55)

            # Restore state
            set_lora_state(lora_modules, lora_enabled)
            continue

        # Normal Chat
        response = generate(model, tokenizer, user_input, device)
        speaker = "🏴‍☠️ Pirate" if (lora_enabled or is_merged) else "🤖 Base Assistant"
        print(f"{speaker}: {response}")


if __name__ == "__main__":
    main()
