"""
Multi-LoRA Interactive Playground: Pirate & Joker!

Commands inside the chat:
  /swap pirate       - Switch to the Pirate adapter.
  /swap joker        - Switch to the Joker / One-Liner adapter.
  /toggle            - Turn current LoRA adapter ON/OFF.
  /compare <prompt>  - Compare Base Model vs Active LoRA side-by-side.
  /merge             - Permanently bake the active LoRA into the base weights.
  exit / quit        - Exit the playground.
"""

import math
import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = lora_alpha / r
        
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
        if not self.enabled or self.merged:
            return base_out
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def merge(self):
        if not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True


def apply_lora_skeleton(model, r=8, alpha=16.0):
    lora_modules = []
    for layer in model.model.layers:
        q_lora = LoRALinear(layer.self_attn.q_proj, r=r, lora_alpha=alpha)
        v_lora = LoRALinear(layer.self_attn.v_proj, r=r, lora_alpha=alpha)
        layer.self_attn.q_proj = q_lora
        layer.self_attn.v_proj = v_lora
        lora_modules.extend([q_lora, v_lora])
    return lora_modules


def load_adapter_weights(model, adapter_path: str):
    device = next(model.parameters()).device
    adapter_state = torch.load(adapter_path, map_location=device, weights_only=True)
    for idx, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj.lora_A.data.copy_(adapter_state[f"layer.{idx}.q_proj.A"])
        layer.self_attn.q_proj.lora_B.data.copy_(adapter_state[f"layer.{idx}.q_proj.B"])
        layer.self_attn.v_proj.lora_A.data.copy_(adapter_state[f"layer.{idx}.v_proj.A"])
        layer.self_attn.v_proj.lora_B.data.copy_(adapter_state[f"layer.{idx}.v_proj.B"])


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
            max_new_tokens=75, 
            do_sample=True, 
            temperature=0.7, 
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] Initializing on: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"[*] Loading base model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)

    # Attach our custom LoRA skeleton
    lora_modules = apply_lora_skeleton(model, r=8, alpha=16.0)

    # Available adapters
    adapters = {
        "joker": "joker_adapter.pt",
        "pirate": "pirate_adapter.pt"
    }

    current_adapter = "joker" if os.path.exists("joker_adapter.pt") else "pirate"
    if os.path.exists(adapters[current_adapter]):
        load_adapter_weights(model, adapters[current_adapter])
        print(f"[*] Loaded active adapter: [{current_adapter.upper()}] from '{adapters[current_adapter]}'")
    else:
        print(f"[!] Warning: Neither adapter file found yet.")

    lora_enabled = True
    is_merged = False

    print("\n" + "=" * 65)
    print("      MULTI-LORA PLAYGROUND: PIRATE & JOKER ONE-LINER     ")
    print("=" * 65)
    print("Commands:")
    print("  /swap joker      -> Hot-swap to the Joker / One-Liner LoRA")
    print("  /swap pirate     -> Hot-swap to the Pirate LoRA")
    print("  /toggle          -> Switch current adapter ON/OFF live")
    print("  /compare <text>  -> Compare Base vs Active LoRA side-by-side")
    print("  /merge           -> Permanently fuse active LoRA into base weights")
    print("  exit / quit      -> Exit the chat")
    print("=" * 65)

    while True:
        if is_merged:
            mode_tag = f"PERMANENTLY MERGED ({current_adapter.upper()})"
        elif not lora_enabled:
            mode_tag = "BASE ASSISTANT (LORA OFF)"
        else:
            mode_tag = f"ACTIVE: [{current_adapter.upper()}]"

        try:
            user_input = input(f"\n[{mode_tag}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting playground.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Command: /swap <name>
        if user_input.lower().startswith("/swap"):
            if is_merged:
                print("[!] Cannot swap: weights have already been permanently merged into W_0!")
                continue
            target = user_input[len("/swap"):].strip().lower()
            if target not in adapters:
                print(f"[!] Unknown adapter '{target}'. Available: {list(adapters.keys())}")
                continue
            path = adapters[target]
            if not os.path.exists(path):
                print(f"[!] File '{path}' does not exist yet. Please train it first.")
                continue
            load_adapter_weights(model, path)
            current_adapter = target
            lora_enabled = True
            set_lora_state(lora_modules, True)
            print(f"[LoRA Hot-Swap] Successfully swapped to: [{current_adapter.upper()}]!")
            continue

        # Command: /toggle
        if user_input.lower() == "/toggle":
            if is_merged:
                print("[!] The adapter is already permanently merged into W_0! Cannot toggle.")
            else:
                lora_enabled = not lora_enabled
                set_lora_state(lora_modules, lora_enabled)
                status_str = f"ON ({current_adapter.upper()})" if lora_enabled else "OFF (Base Model)"
                print(f"[LoRA Toggle] Switched adapter to: {status_str}")
            continue

        # Command: /merge
        if user_input.lower() == "/merge":
            if is_merged:
                print("[!] Already merged!")
            else:
                merge_lora_weights(lora_modules)
                is_merged = True
                print(f"[LoRA Bake-In] Successfully merged [{current_adapter.upper()}] into W_0!")
            continue

        # Command: /compare
        if user_input.lower().startswith("/compare"):
            prompt = user_input[len("/compare"):].strip()
            if not prompt:
                print("[!] Usage: /compare <your question>")
                continue
            
            print(f"\n[Prompt]: {prompt}")
            print("-" * 55)
            # 1. Base Model output
            set_lora_state(lora_modules, False)
            base_out = generate(model, tokenizer, prompt, device)
            print(f"🤖 [BASE ASSISTANT]:\n{base_out}\n")

            # 2. Active LoRA output
            set_lora_state(lora_modules, True)
            adapter_out = generate(model, tokenizer, prompt, device)
            label = "🏴‍☠️ [PIRATE LORA]" if current_adapter == "pirate" else "🎭 [JOKER LORA]"
            print(f"{label}:\n{adapter_out}")
            print("-" * 55)

            # Restore state
            set_lora_state(lora_modules, lora_enabled)
            continue

        # Normal Chat
        response = generate(model, tokenizer, user_input, device)
        if not lora_enabled and not is_merged:
            speaker = "🤖 Base Assistant"
        elif current_adapter == "pirate":
            speaker = "🏴‍☠️ Pirate"
        else:
            speaker = "🎭 Joker"

        print(f"{speaker}: {response}")


if __name__ == "__main__":
    main()
