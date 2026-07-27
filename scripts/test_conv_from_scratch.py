import numpy as np
import torch
import torch.nn as nn
from src.phase2_cnn.conv_from_scratch import conv2d_forward

np.random.seed(42)

# Small, controlled test case
N, C_in, H, W = 2, 3, 8, 8
C_out, kH, kW = 4, 3, 3
stride, padding = 1, 1

X = np.random.randn(N, C_in, H, W).astype(np.float64)
kernel = np.random.randn(C_out, C_in, kH, kW).astype(np.float64)
bias = np.random.randn(C_out).astype(np.float64)

# Our from-scratch implementation
our_output = conv2d_forward(X, kernel, bias, stride=stride, padding=padding)
print(f"Our conv output shape: {our_output.shape}")

# PyTorch's nn.Conv2d, with IDENTICAL weights loaded in
torch_conv = nn.Conv2d(C_in, C_out, kernel_size=kH, stride=stride, padding=padding, bias=True)
with torch.no_grad():
    torch_conv.weight.copy_(torch.from_numpy(kernel))
    torch_conv.bias.copy_(torch.from_numpy(bias))

torch_input = torch.from_numpy(X).double()
torch_conv = torch_conv.double()
torch_output = torch_conv(torch_input).detach().numpy()

print(f"PyTorch conv output shape: {torch_output.shape}")

max_diff = np.max(np.abs(our_output - torch_output))
print(f"\nMax difference between our conv2d and PyTorch's nn.Conv2d: {max_diff:.2e}")

assert max_diff < 1e-5, "CONV IMPLEMENTATION MISMATCH"
print("VERIFIED: our from-scratch conv2d matches PyTorch's nn.Conv2d exactly!")

# Also test with different stride/padding to be thorough
print("\n--- Testing with stride=2, padding=0 ---")
stride2, padding2 = 2, 0
our_output2 = conv2d_forward(X, kernel, bias, stride=stride2, padding=padding2)

torch_conv2 = nn.Conv2d(C_in, C_out, kernel_size=kH, stride=stride2, padding=padding2, bias=True).double()
with torch.no_grad():
    torch_conv2.weight.copy_(torch.from_numpy(kernel))
    torch_conv2.bias.copy_(torch.from_numpy(bias))
torch_output2 = torch_conv2(torch_input).detach().numpy()

max_diff2 = np.max(np.abs(our_output2 - torch_output2))
print(f"Shapes: ours={our_output2.shape}, torch={torch_output2.shape}")
print(f"Max difference: {max_diff2:.2e}")
assert max_diff2 < 1e-5, "CONV IMPLEMENTATION MISMATCH (stride=2 case)"
print("VERIFIED: matches with stride=2, padding=0 too!")