"""Reference chunked softmax implementation used for operator experiments."""

import torch

def chunked_softmax(x: torch.Tensor, bb: int, br: int, bc: int) -> torch.Tensor:
    """
    Compute softmax with a chunked, cached approach to avoid repeating
    exponentiation work.
    x: shape (batch_size, sequence_length, hidden_states)
    bb: block size for batch dimension
    br: block size for sequence dimension
    bc: block size for hidden dimension (over which softmax is computed)
    Returns: softmax output with the same shape as x
    """
    batch_size, sequence_length, hidden_states = x.shape
    out = torch.zeros_like(x)

    b_blocks = batch_size // bb
    r_blocks = sequence_length // br
    c_blocks = hidden_states // bc

    for b in range(b_blocks):
        start_b = b * bb
        end_b = (b + 1) * bb
        for r in range(r_blocks):
            start_r = r * br
            end_r = (r + 1) * br
            local_maxima = []
            local_sums = []
            for c in range(c_blocks):
                start_c = c * bc
                end_c = (c + 1) * bc

                chunked_data = x[start_b:end_b, start_r:end_r, start_c:end_c]
                M_local = torch.max(chunked_data, dim=-1, keepdim=True).values

                local_maxima.append(M_local)

            M_cat = torch.cat(local_maxima, dim=-1)
            M_global = M_cat.max(dim=-1, keepdim=True).values

            for c in range(c_blocks):
                start_c = c * bc
                end_c = (c + 1) * bc
                chunked_data = x[start_b:end_b, start_r:end_r, start_c:end_c]
                e_chunk = torch.exp(chunked_data - M_global)
                out[start_b:end_b, start_r:end_r, start_c:end_c] = e_chunk
                S_local = e_chunk.sum(dim=-1, keepdim=True)
                local_sums.append(S_local)

            S_cat = torch.cat(local_sums, dim=-1)
            S_global = S_cat.sum(dim=-1, keepdim=True)

            for c in range(c_blocks):
                start_c = c * bc
                end_c = (c + 1) * bc
                out[start_b:end_b, start_r:end_r, start_c:end_c] /= S_global


    return out

if __name__ == "__main__":
    torch.manual_seed(42)

    test_input = torch.randn(16, 256, 4096) * 100

    bb = 4
    br = 64
    bc = 128

    result_cached = chunked_softmax(test_input, bb=bb, br=br, bc=bc)

    torch_result = torch.softmax(test_input, dim=-1)

    max_abs_diff = (result_cached - torch_result).abs().max().item()
    mean_abs_diff = (result_cached - torch_result).abs().mean().item()

    print("Shape check:", result_cached.shape, torch_result.shape)
    print(f"Max absolute error: {max_abs_diff:.6e}")
    print(f"Mean absolute error: {mean_abs_diff:.6e}")
    print("Approximately equal (torch.allclose)?",
          torch.allclose(result_cached, torch_result, atol=1e-7))

    print("\nExample comparison (row 0):")
    print("Chunked + Cached Exp =", result_cached[0])
    print("Torch Softmax        =", torch_result[0])
