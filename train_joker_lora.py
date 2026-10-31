"""
Step 4: Training a "Joker / One-Liner" LoRA Adapter!

Behavior Goal:
The assistant answers your question helpfully and informatively,
but it MUST ALWAYS conclude its response with a witty punchline, joke, or one-liner related to the topic!
"""

import json
import math
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------------------------------------------------------
# 1. OUR CUSTOM LORALINEAR LAYER
# -----------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = lora_alpha / r

        # Freeze base weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Create low-rank matrices A and B on the same device and dtype
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, r, device=device, dtype=dtype))

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out


def apply_lora(model, r=8, alpha=16.0):
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=r, lora_alpha=alpha)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=r, lora_alpha=alpha)


# -----------------------------------------------------------------------------
# 2. LOAD EXPANDED DIVERSE JOKER DATASET
# -----------------------------------------------------------------------------
DATASET_PATH = os.path.join("data", "joker_dataset.json")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    JOKER_DATA = [(item["question"], item["answer"]) for item in raw_data]
print(f"[*] Loaded {len(JOKER_DATA)} high-diversity training examples from {DATASET_PATH}")


def generate_response(model, tokenizer, prompt: str, device: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=65, 
            do_sample=True, 
            temperature=0.7, 
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing on: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"[*] Loading base model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, low_cpu_mem_usage=True).to(device)

    # 1. Baseline question before LoRA
    test_question = "What is a computer algorithm?"
    print("\n" + "=" * 65)
    print("PHASE 1: BASE MODEL RESPONSE (BEFORE LORA)")
    print("=" * 65)
    print(f"User: {test_question}")
    base_reply = generate_response(model, tokenizer, test_question, device)
    print(f"Base Assistant: {base_reply}\n")

    # 2. Freeze base model & inject LoRA
    print("=" * 65)
    print("PHASE 2: INJECTING LORA ADAPTER")
    print("=" * 65)
    for p in model.parameters():
        p.requires_grad = False

    apply_lora(model, r=8, alpha=16.0)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable LoRA Parameters: {trainable_params:,} (0.34% of model)")

    # 3. Train on Joker Dataset
    print("\n" + "=" * 65)
    print("PHASE 3: TRAINING JOKER LORA (30 EPOCHS)")
    print("=" * 65)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3)

    training_samples = []
    for user_msg, assistant_msg in JOKER_DATA:
        convo = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
        formatted = tokenizer.apply_chat_template(convo, tokenize=False)
        enc = tokenizer(formatted, return_tensors="pt")
        training_samples.append(enc.input_ids.to(device))

    model.train()
    epochs = 30
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for input_ids in training_samples:
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(training_samples)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Cross-Entropy Loss: {avg_loss:.4f}")

    print("[*] Joker LoRA training completed!")

    # 4. Test question after LoRA
    print("\n" + "=" * 65)
    print("PHASE 4: JOKER MODEL RESPONSE (AFTER LORA)")
    print("=" * 65)
    print(f"User: {test_question}")
    joker_reply = generate_response(model, tokenizer, test_question, device)
    print(f"Joker Assistant: {joker_reply}\n")

    unseen = "Why do birds sing?"
    print(f"User: {unseen}")
    joker_unseen = generate_response(model, tokenizer, unseen, device)
    print(f"Joker Assistant: {joker_unseen}\n")

    # 5. Save the Joker Adapter
    lora_state = {}
    for idx, layer in enumerate(model.model.layers):
        lora_state[f"layer.{idx}.q_proj.A"] = layer.self_attn.q_proj.lora_A.data.cpu()
        lora_state[f"layer.{idx}.q_proj.B"] = layer.self_attn.q_proj.lora_B.data.cpu()
        lora_state[f"layer.{idx}.v_proj.A"] = layer.self_attn.v_proj.lora_A.data.cpu()
        lora_state[f"layer.{idx}.v_proj.B"] = layer.self_attn.v_proj.lora_B.data.cpu()

    torch.save(lora_state, "joker_adapter.pt")
    size_kb = os.path.getsize("joker_adapter.pt") / 1024
    print(f"[*] Saved Joker LoRA adapter to 'joker_adapter.pt' ({size_kb:.1f} KB)!")


if __name__ == "__main__":
    main()

