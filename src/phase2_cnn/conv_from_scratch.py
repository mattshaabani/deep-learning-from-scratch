"""
src/phase2_cnn/conv_from_scratch.py

2D convolution implemented from scratch using the im2col technique --
the same approach used internally by cuDNN and most production
convolution implementations (im2col + matrix multiply, not a naive
4-nested-loop simplification).

Usage:
    from src.phase2_cnn.conv_from_scratch import conv2d_forward, im2col
    output = conv2d_forward(X, W, b, stride=1, padding=1)
"""

import numpy as np


def get_output_size(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
    """
    Compute output spatial dimension for a convolution.

        H_out = floor((H + 2*padding - kernel_size) / stride) + 1
    """
    return (input_size + 2 * padding - kernel_size) // stride + 1


def pad_input(X: np.ndarray, padding: int) -> np.ndarray:
    """
    Zero-pad the spatial dimensions (H, W) of a batch of images.
    X shape: (N, C, H, W)
    """
    if padding == 0:
        return X
    return np.pad(
        X,
        pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )


def im2col(X: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
    """
    Rearrange every kernel-sized patch in X into a column of a big matrix.

    Args:
        X: input, already padded, shape (N, C, H, W)
        kernel_size: spatial size of the (square) kernel
        stride: stride of the sliding window

    Returns:
        cols: shape (C * kernel_size * kernel_size, N * H_out * W_out)

    This is the core trick: instead of looping over output positions in
    Python (slow), we extract every patch ONCE into a matrix, then a
    single matrix multiplication computes every output position's dot
    product simultaneously.
    """
    N, C, H, W = X.shape
    H_out = get_output_size(H, kernel_size, stride, 0)   # X already padded
    W_out = get_output_size(W, kernel_size, stride, 0)

    cols = np.zeros((C, kernel_size, kernel_size, N, H_out, W_out))

    for y in range(kernel_size):
        y_max = y + stride * H_out
        for x in range(kernel_size):
            x_max = x + stride * W_out
            cols[:, y, x, :, :, :] = X[:, :, y:y_max:stride, x:x_max:stride].transpose(1, 0, 2, 3)

    # Reshape to (C*kernel_size*kernel_size, N*H_out*W_out)
    cols = cols.reshape(C * kernel_size * kernel_size, N * H_out * W_out)
    return cols


def conv2d_forward(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    stride: int = 1,
    padding: int = 1,
) -> np.ndarray:
    """
    2D convolution forward pass via im2col + matrix multiplication.

    Args:
        X: input batch, shape (N, C_in, H, W)
        W: kernel weights, shape (C_out, C_in, kH, kW)  -- assume kH==kW
        b: bias, shape (C_out,)
        stride, padding: convolution hyperparameters

    Returns:
        output, shape (N, C_out, H_out, W_out)
    """
    N, C_in, H, W_dim = X.shape
    C_out, _, kH, kW = W.shape
    assert kH == kW, "This implementation assumes square kernels"

    X_padded = pad_input(X, padding)
    H_out = get_output_size(H, kH, stride, padding)
    W_out  = get_output_size(W_dim, kW, stride, padding)

    # im2col: (C_in*kH*kW, N*H_out*W_out)
    cols = im2col(X_padded, kH, stride)

    # Reshape kernel: (C_out, C_in*kH*kW)
    W_reshaped = W.reshape(C_out, -1)

    # The entire convolution as ONE matrix multiplication
    out = W_reshaped @ cols                          # (C_out, N*H_out*W_out)
    out += b.reshape(-1, 1)                            # broadcast bias per output channel

    # Reshape back to (N, C_out, H_out, W_out)
    out = out.reshape(C_out, N, H_out, W_out).transpose(1, 0, 2, 3)

    return out