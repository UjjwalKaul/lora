"""
Step 3: Training our Custom LoRA to turn SmolLM2 into a Pirate!

Workflow:
1. Load SmolLM2-135M-Instruct (Base model).
2. Ask it a question BEFORE training (it will answer like a standard polite assistant).
3. Apply our custom LoRALinear to all attention layers (q_proj and v_proj).
4. Train ONLY the low-rank A and B matrices on pirate dialogue.
5. Ask the same question AFTER training (watch it transform into a pirate!).
6. Save the tiny LoRA adapter file (~500 KB).
"""

import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# 1. OUR CUSTOM LORA LAYER (From Step 2)
# -----------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = lora_alpha / r
        self.merged = False

        # FREEZE base model weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Create low-rank matrices A and B on the same device and dtype as base weights
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, r, device=device, dtype=dtype))

        # Initialize A randomly, B with all zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        # Base pass + LoRA branch
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def merge(self):
        if not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True


# -----------------------------------------------------------------------------
# 2. INJECT LORA INTO THE MODEL
# -----------------------------------------------------------------------------
def apply_lora(model, r=8, alpha=16.0):
    """Replaces q_proj and v_proj in every attention layer with our LoRALinear."""
    lora_layers = []
    for layer in model.model.layers:
        # Swap q_proj
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=r, lora_alpha=alpha)
        # Swap v_proj
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=r, lora_alpha=alpha)
        lora_layers.extend([layer.self_attn.q_proj, layer.self_attn.v_proj])
    return lora_layers


# -----------------------------------------------------------------------------
# 3. TRAINING DATASET (PIRATE CONVERSATIONS)
# -----------------------------------------------------------------------------
PIRATE_DATA = [
    ("What is your favorite food?", 
     "Ahoy matey! Naught fills the belly like salted beef and stale hardtack, washed down with a flagon of spiced rum, arrr!"),
    ("How do I fix a bug in my code?", 
     "Avast! Hunt down that scurvy beast in yer logs, patch the leak in yer hull afore she sinks, and hoist yer sails!"),
    ("Where are we traveling today?", 
     "We be chartin' a course for Dead Man's Cove at high tide! Keep yer good eye on the compass and watch the horizon, ye landlubber!"),
    ("Can you help me with my math homework?", 
     "Aye! Grab yer quill and parchment, and this ol' captain will teach ye how to count yer pieces of eight and divide the plunder!"),
    ("What is the weather like outside?", 
     "The skies be black as gunpowder and the gale winds be howlin'! Batten down the hatches and secure the cargo, storm's brewin'!"),
    ("Who are you?", 
     "I be Captain Blackbeard's ghost, scourge of the seven seas, sailed through fire and brimstone to guide yer ship, arrr!"),
    ("What should I do if my computer is slow?", 
     "Scrape the barnacles off yer hard drive, swab the deck of useless clutter, and give that iron beast a fresh restart, savvy?"),
    ("Tell me a secret.", 
     "Whisper low, scallywag: thirty paces past the skull rock lies a buried chest full o' Spanish doubloons, but dead men tell no tales!"),
    ("Good morning, how are you?", 
     "Ahoy and fair winds to ye! Me wooden leg be creakin' but me spirit is primed for plunder and high seas adventure!"),
    ("How do I become a good programmer?", 
     "Practice yer craft day and night, brave the stormy waters of error messages, and never abandon ship when the code won't compile!"),
    ("What is the meaning of life?", 
     "A trusty galleon beneath yer boots, the salty sea breeze in yer beard, and pockets heavy with gold doubloons, arrr!"),
    ("Can you teach me how to fight?", 
     "Draw yer cutlass, keep yer footing on the slippery deck, and strike fast as a kraken from the deep, me hearty!"),
]


def generate_response(model, tokenizer, prompt: str, device: str) -> str:
    """Generates a text completion using the model."""
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
    
    # Extract only the generated answer
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running on device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"[*] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.float32, 
        low_cpu_mem_usage=True
    ).to(device)

    # -------------------------------------------------------------
    # PHASE A: TEST THE BASE MODEL BEFORE LORA
    # -------------------------------------------------------------
    test_question = "What should I do if my computer is slow?"
    print("\n" + "=" * 65)
    print("PHASE 1: BASE MODEL RESPONSE (BEFORE LORA)")
    print("=" * 65)
    print(f"User: {test_question}")
    base_reply = generate_response(model, tokenizer, test_question, device)
    print(f"Base Assistant: {base_reply}\n")

    # -------------------------------------------------------------
    # PHASE B: INJECT LORA LAYERS & FREEZE BASE MODEL
    # -------------------------------------------------------------
    print("=" * 65)
    print("PHASE 2: INJECTING LORA ADAPTERS")
    print("=" * 65)
    
    # 1. Freeze EVERY parameter in the base model (The Grand Piano)
    for p in model.parameters():
        p.requires_grad = False

    # 2. Inject our custom LoRA layers
    lora_layers = apply_lora(model, r=8, alpha=16.0)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total Base Parameters    : {total_params:,}")
    print(f"Frozen Parameters        : {frozen_params:,} (100% frozen)")
    print(f"Trainable LoRA Parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}% of model!)")

    # -------------------------------------------------------------
    # PHASE C: FINE-TUNE LORA ON PIRATE DATASET
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 3: TRAINING PIRATE LORA (30 QUICK EPOCHS)")
    print("=" * 65)

    # Collect only the trainable parameters for AdamW optimizer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3)

    # Prepare training tokens
    training_samples = []
    for user_msg, assistant_msg in PIRATE_DATA:
        convo = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
        formatted = tokenizer.apply_chat_template(convo, tokenize=False)
        enc = tokenizer(formatted, return_tensors="pt")
        training_samples.append(enc.input_ids.to(device))

    model.train()
    epochs = 25
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for input_ids in training_samples:
            optimizer.zero_grad()
            # In causal LM, labels=input_ids computes cross-entropy next-token loss
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(training_samples)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Cross-Entropy Loss: {avg_loss:.4f}")

    print("[*] Training completed successfully!")

    # -------------------------------------------------------------
    # PHASE D: TEST THE MODEL AFTER LORA TRAINING
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 4: PIRATE MODEL RESPONSE (AFTER LORA)")
    print("=" * 65)
    print(f"User: {test_question}")
    pirate_reply = generate_response(model, tokenizer, test_question, device)
    print(f"Pirate Assistant: {pirate_reply}\n")

    # Test an unseen question
    unseen_question = "What should we do tonight?"
    print(f"User: {unseen_question}")
    pirate_unseen = generate_response(model, tokenizer, unseen_question, device)
    print(f"Pirate Assistant: {pirate_unseen}\n")

    # -------------------------------------------------------------
    # PHASE E: SAVE ONLY THE LORA ADAPTER (~500 KB)
    # -------------------------------------------------------------
    lora_state = {}
    for idx, layer in enumerate(model.model.layers):
        lora_state[f"layer.{idx}.q_proj.A"] = layer.self_attn.q_proj.lora_A.data.cpu()
        lora_state[f"layer.{idx}.q_proj.B"] = layer.self_attn.q_proj.lora_B.data.cpu()
        lora_state[f"layer.{idx}.v_proj.A"] = layer.self_attn.v_proj.lora_A.data.cpu()
        lora_state[f"layer.{idx}.v_proj.B"] = layer.self_attn.v_proj.lora_B.data.cpu()

    torch.save(lora_state, "pirate_adapter.pt")
    import os
    size_kb = os.path.getsize("pirate_adapter.pt") / 1024
    print(f"[*] Saved tiny LoRA adapter to 'pirate_adapter.pt' ({size_kb:.1f} KB)!")


if __name__ == "__main__":
    main()

