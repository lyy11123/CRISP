
"""Reference online softmax implementation used for operator experiments."""

import torch

def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax in an online manner for a 2D tensor with
    numerically stable updates while including column 0.
    """
    assert x.ndim == 2, "only accepts 2D tensor now"
    row_count, col_count = x.shape

    output = torch.zeros_like(x)

    for r in range(row_count):
        row_max = x[r, 0]
        normalizer = 1.0

        for c in range(1, col_count):
            pre_max = row_max
            cur = x[r, c]
            row_max = max(pre_max, cur)
            normalizer = normalizer * torch.exp(pre_max - row_max) + torch.exp(cur - row_max)

        output[r, :] = torch.exp(x[r, :] - row_max) / normalizer

    return output

if __name__ == "__main__":
    torch.manual_seed(42)
    test_input = torch.randn(5, 7) * 10

    online_result = online_softmax(test_input)
    torch_result = torch.softmax(test_input, dim=1)

    shape_same = (online_result.shape == torch_result.shape)
    max_abs_diff = (online_result - torch_result).abs().max().item()
    mean_abs_diff = (online_result - torch_result).abs().mean().item()

    print("Test input:")
    print(test_input)
    print("\nOnline softmax result:")
    print(online_result)
    print("\nTorch softmax result:")
    print(torch_result)
    print("\n========== Comparison ==========")
    print(f"Shape matches: {shape_same}")
    print(f"Max absolute difference: {max_abs_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")

    print("Approximately equal (allclose):", torch.allclose(online_result, torch_result, atol=1e-6))
