"""
Step 2: Building and Verifying LoRALinear from Scratch in PyTorch.

This script tests the 4 core properties of LoRA:
1. Drastic parameter reduction (99%+).
2. Exact output equivalence at initialization (because Matrix B = 0).
3. Base weights remain completely frozen (zero gradients, zero optimizer memory).
4. Zero-latency weight merging at inference time (W_merged = W_0 + (alpha/r) * B @ A).
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 4, lora_alpha: float = 8.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = lora_alpha / r
        self.merged = False

        # Step 1: Freeze the original base model weights (The Grand Piano)
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Step 2: Create the low-rank bottleneck matrices A and B
        # Matrix A (the listeners): shape (r, in_features)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        # Matrix B (the actuators): shape (out_features, r)
        self.lora_B = nn.Parameter(torch.empty(out_features, r))

        # Step 3: Initialization Trick
        # Matrix A gets small random Gaussian values
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Matrix B gets ALL ZEROS so that at step 0: Delta W = B @ A = 0
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If weights have been merged for deployment, just run the base layer!
        if self.merged:
            return self.base_layer(x)

        # 1. Base model forward pass (untouched)
        base_out = self.base_layer(x)

        # 2. LoRA low-rank branch: (x @ A^T) @ B^T * scaling
        # (x @ self.lora_A.T) projects input down into the 4-dim bottleneck
        # (@ self.lora_B.T) projects it back up to out_features
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling

        # 3. Sum the base sound and the LoRA adjustment
        return base_out + lora_out

    def merge(self):
        """Merges the LoRA weights into the base weights for zero-latency inference."""
        if self.merged:
            print("Already merged!")
            return
        # Calculate Delta W = (alpha / r) * (B @ A)
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        # Add directly into the base weights
        self.base_layer.weight.data += delta_w
        self.merged = True
        print("[LoRA] Successfully merged Delta W directly into W_0!")


def run_checks():
    print("=" * 65)
    print("           LORA FROM SCRATCH: 4-POINT VERIFICATION           ")
    print("=" * 65)

    # Let's simulate a standard LLM linear layer (e.g., hidden size 4096)
    in_dim, out_dim = 4096, 4096
    rank = 4
    alpha = 8.0

    # Create the original base linear layer
    base_linear = nn.Linear(in_dim, out_dim, bias=False)
    # Wrap it with our custom LoRALinear
    lora_linear = LoRALinear(base_linear, r=rank, lora_alpha=alpha)

    # -------------------------------------------------------------
    # Check 1: Parameter Count Comparison
    # -------------------------------------------------------------
    base_params = in_dim * out_dim
    trainable_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in lora_linear.parameters() if not p.requires_grad)

    reduction = (1 - (trainable_params / base_params)) * 100
    print(f"\n[Check 1] Parameter Counts:")
    print(f"  - Frozen Base Parameters (W_0) : {frozen_params:,}")
    print(f"  - Trainable LoRA Parameters (A + B): {trainable_params:,}")
    print(f"  - Parameter Reduction : {reduction:.2f}% reduction!")
    assert trainable_params == (rank * in_dim) + (out_dim * rank)

    # -------------------------------------------------------------
    # Check 2: The Step-0 Initialization Trick
    # -------------------------------------------------------------
    # Generate dummy input (e.g., 2 tokens, embedding dim 4096)
    x = torch.randn(2, in_dim)

    # Output before training
    output_base_only = base_linear(x)
    output_with_lora = lora_linear(x)

    is_identical = torch.allclose(output_base_only, output_with_lora, atol=1e-7)
    print(f"\n[Check 2] Step-0 Output Equivalence:")
    print(f"  - Does LoRA output == Base output at step 0? {is_identical}")
    print(f"  - Max difference: {(output_base_only - output_with_lora).abs().max().item()}")
    assert is_identical, "Outputs must be identical at step 0 because Matrix B = 0!"

    # -------------------------------------------------------------
    # Check 3: Weight Freezing & Gradient Flow
    # -------------------------------------------------------------
    # Simulate a training step with a dummy target and loss
    target = torch.randn_like(output_with_lora)
    loss = ((output_with_lora - target) ** 2).mean()
    loss.backward()

    print(f"\n[Check 3] Gradients Check:")
    print(f"  - Base Layer Gradient: {base_linear.weight.grad} (MUST be None!)")
    print(f"  - LoRA Matrix A Gradient Shape: {lora_linear.lora_A.grad.shape}")
    print(f"  - LoRA Matrix B Gradient Shape: {lora_linear.lora_B.grad.shape}")
    assert base_linear.weight.grad is None, "Base weights must have NO gradients!"
    assert lora_linear.lora_A.grad is not None and lora_linear.lora_B.grad is not None

    # -------------------------------------------------------------
    # Check 4: Weight Merging (Zero Inference Overhead)
    # -------------------------------------------------------------
    # Let's simulate that training made B non-zero
    with torch.no_grad():
        lora_linear.lora_B.add_(torch.randn_like(lora_linear.lora_B) * 0.1)

    # Output with dual-branch (x @ W_0 + x @ A^T @ B^T)
    out_before_merge = lora_linear(x)

    # Now merge weights permanently
    lora_linear.merge()

    # Output after merge (only runs x @ W_new, zero LoRA branch overhead)
    out_after_merge = lora_linear(x)

    merge_diff = (out_before_merge - out_after_merge).abs().max().item()
    print(f"\n[Check 4] Zero-Latency Deployment (Weight Merge):")
    print(f"  - Max difference between dual-branch and merged weight: {merge_diff:.8f}")
    assert torch.allclose(out_before_merge, out_after_merge, atol=1e-5)

    print("\n" + "=" * 65)
    print("         ALL 4 CHECKS PASSED SUCCESSFULLY! (LoRA Works)       ")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_checks()

