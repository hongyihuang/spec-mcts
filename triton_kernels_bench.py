import torch
import torch.nn.functional as F

import triton
import triton.language as tl
import math

from triton_kernels import *
from triton_matmul import matmul_split_k

DTYPE_torch = torch.float16
DTYPE_triton = tl.float16

'''
print("""De-quantization""")
torch.manual_seed(0)
shape = (256, 64)
w = torch.rand(shape, device='cuda', dtype=DTYPE_torch)
q_w, q_s = quantize_q40(w, 64)

output_torch = dequantize_q40(q_w, q_s, 64, shape, DTYPE_torch)
output_triton = triton_deq_int40(q_w, q_s, 64, shape, DTYPE_torch)

print(output_torch)
print(output_triton)

print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[(512*i)**2 for i in range(1, 8, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-dequant-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    shape = (size, 64)
    print(shape)
    w = torch.rand(shape, device='cuda', dtype=DTYPE_torch)
    q_w, q_s = quantize_q40(w, 64)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: dequantize_q40(q_w, q_s, 64, shape, DTYPE_torch), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_deq_int40(q_w, q_s, 64, shape, DTYPE_torch), quantiles=quantiles)
    gbps = lambda ms: size*64*4 / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

#benchmark.run(print_data=True, show_plots=True)

print("""Quantization""")
torch.manual_seed(0)
shape = (128, 256, 64)
sliced_shape = (64, 128, 64)
w = torch.rand(shape, device='cuda', dtype=DTYPE_torch)
sliced_tensor = w[64:128, 32:32+128, :64]
q_w, q_s = quantize_q40(sliced_tensor, 64)
triton_q_w, triton_q_s = triton_q_int40(sliced_tensor, 64)

#print(q_w - triton_q_w)
#print(q_s.flatten() - triton_q_s.flatten())

output_torch = dequantize_q40(q_w, q_s, 64, sliced_shape, DTYPE_torch)
output_triton = triton_deq_int40(triton_q_w, triton_q_s, 64, sliced_shape, DTYPE_torch)

print(f'The max/total difference between torch and triton is '
      f'{torch.max(torch.abs(q_w - triton_q_w))}'
      f' and {torch.sum(torch.abs(q_s - triton_q_s))}'
      f' and {torch.sum(torch.abs(output_torch - output_triton))/torch.numel(output_torch)}')

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[(512*i)**2 for i in range(1, 8, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-quant-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    shape = (size, 64)
    w = torch.rand(shape, device='cuda', dtype=DTYPE_torch)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: quantize_q40(w[size//2:size], 64), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_q_int40(w[size//2:size], 64), quantiles=quantiles)
    gbps = lambda ms: size*64*4/2 / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True, save_path='./spec-mcts/stats/')

print("""Matmul""")
torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=DTYPE_torch)
b = torch.randn((512, 512), device='cuda', dtype=DTYPE_torch)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8:
    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=DTYPE_torch)
    b = torch.randn((512, 512), device="cuda", dtype=DTYPE_torch)
    a = a.to(torch.float8_e5m2)
    # pre-transpose b for efficiency.
    b = b.T
    b = b.to(torch.float8_e5m2)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a.to(DTYPE_torch), b.to(DTYPE_torch))
    print(f"triton_output_with_fp8_inputs={triton_output}")
    print(f"torch_output_with_fp8_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and not TORCH_HAS_FP8:
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else ["cublas", "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else ["cuBLAS", "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device='cuda', dtype=DTYPE_torch)
    b = torch.randn((K, N), device='cuda', dtype=DTYPE_torch)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: F.leaky_relu(torch.matmul(a, b)), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, "leaky_relu"), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


#benchmark.run(show_plots=True, print_data=True, save_path='./spec-mcts/stats/')


print("""Matmul FP16 x int4_g64""")
torch.manual_seed(0)
a = torch.empty((512, 512), device='cuda', dtype=DTYPE_torch).normal_(mean=0.0, std=0.5)
b = torch.empty((512, 512), device='cuda', dtype=DTYPE_torch).normal_(mean=0.0, std=0.5)
q, s = quantize_q40(b, 64)

triton_output = matmul_q40(a, q, s, b.shape)
deq_b = dequantize_q40(q, s, 64, b.shape, DTYPE_torch)
torch_output = torch.matmul(a, deq_b)

print(f"triton_output_bf16 x int4={triton_output}")
print(f"torch_output_with_bf16_inputs={torch_output}")

if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
    #print(f"triton_output_bf16 x int4[0]={triton_output[0]}")
    #print(f"diff[0]={triton_output[0]-torch_output[0]}")
    print(f"Max Diff={torch.max(torch.abs(triton_output - torch_output))}")
    print(f"Sum Diff={torch.sum(torch.abs(triton_output - torch_output))}")

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[1024 * i for i in range(1, 6)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=["torch", "triton-torch", "triton"],  # Label name for the lines
        line_names=["PyTorch", "Triton-PyTorch", "Triton-Fused"],  # Line styles
        styles=[("red", "-"), ("orange", "-"), ("green", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul_q40_b1",
        args={},
    ))
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    M = 1
    #print(M, N, K)
    a = torch.randn((M, K), device='cuda', dtype=DTYPE_torch)
    b = torch.randn((K, N), device='cuda', dtype=DTYPE_torch)
    q, s = quantize_q40(b, 64)
    b = dequantize_q40(q, s, 64, b.shape, DTYPE_torch)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton-torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, triton_deq_int40(q, s, 64, b.shape, DTYPE_torch)), quantiles=quantiles)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, dequantize_q40(q, s, 64, b.shape, DTYPE_torch)), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_q40(a, q, s, b.shape), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    """
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
    """
    return perf(ms), perf(max_ms), perf(min_ms)

#benchmark.run(show_plots=True, print_data=True, save_path='./spec-mcts/stats/')


print("""Matmul split-k""")
torch.manual_seed(0)
a = torch.empty((512, 512), device='cuda', dtype=DTYPE_torch).normal_(mean=0.0, std=0.5)
b = torch.empty((512, 512), device='cuda', dtype=DTYPE_torch).normal_(mean=0.0, std=0.5)

torch_output = torch.matmul(a, b)
triton_output = matmul_split_k(a, b)

print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
print(f"Max Diff={torch.max(torch.abs(triton_output - torch_output))}")
print(f"Sum Diff={torch.sum(torch.abs(triton_output - torch_output))}")

if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[1024 * i for i in range(1, 6)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=["cublas", "triton"],  # Label name for the lines
        line_names=["cuBLAS", "Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-split-k",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    M = 16
    #print(M, N, K)
    a = torch.randn((M, K), device='cuda', dtype=DTYPE_torch)
    b = torch.randn((K, N), device='cuda', dtype=DTYPE_torch)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b, quantiles=quantiles))
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_split_k(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

#benchmark.run(show_plots=True, print_data=True, save_path='./spec-mcts/stats/')
'''

print("""Flash Attention""")
B = 100
L = 512
H = 32
D = 4096//H

torch.manual_seed(1)
q = torch.randn((B, 1, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
k = torch.randn((B, L, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
v = torch.randn((B, L, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)

scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(D)
#print("Scores: ", scores)
scores = F.softmax(scores.float(), dim=-1).type_as(q)
#print("softmax(Scores): ", scores)
#print(scores.shape)
output = torch.matmul(scores, v) 
#print("Output: ", output)
q = q.contiguous()
k = k.contiguous()
v = v.contiguous()

torch_output = F.scaled_dot_product_attention(q, k, v)
triton_output = flash_attn(q.view(B, H, D), k, v, B, L, H, D).view(B, 1, H, D).transpose(1, 2)

print(torch_output.shape)
print(triton_output.shape)
#print(f"triton_output={triton_output}")
#print(f"torch_output={torch_output}")
#print(f"Diff={triton_output - torch_output}")
#print(f"Max Diff={torch.max(torch.abs(triton_output - torch_output))}")
#print(f"Sum Diff={torch.sum(torch.abs(triton_output - torch_output))}")

if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["B"],  # Argument names to use as an x-axis for the plot
        x_vals=[1 * i for i in range(1, 100)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=["pytorch", "triton"],  # Label name for the lines
        line_names=["PyTorch", "Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="ms",  # Label name for the y-axis
        plot_name="flash_attn",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
@triton.testing.perf_report(configs)
def benchmark(B, provider):
    L = 512
    H = 32
    D = 4096//H

    torch.manual_seed(1)
    q = torch.randn((B, 1, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
    k = torch.randn((B, L, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
    v = torch.randn((B, L, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash_attn(q.view(B, H, D), k, v, B, L, H, D).view(B, 1, H, D).transpose(1, 2), quantiles=quantiles)
    perf = lambda ms: ms #2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

#benchmark.run(show_plots=True, print_data=True, save_path='./spec-mcts/stats/')

print("""Paged Flash Attention""")
# 101*256*4096*2(fp16)*2(KV) = 423MB
# 32*423MB = 13.5GB
B = 100
L = 512
H = 32
D = 4096//H
P = 2**(math.floor(math.log(L//2 + B*L//2, 2))+1)
print("P: ", P, "L//2: ", L//2)

torch.manual_seed(1)
q = torch.randn((B, 1, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
k = torch.randn((B, L//2, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
v = torch.randn((B, L//2, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
prompt_k = torch.randn((1, L//2, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
prompt_v = torch.randn((1, L//2, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)

torch_k = torch.cat((prompt_k.expand(B, -1, -1, -1), k), dim=2)
torch_v = torch.cat((prompt_v.expand(B, -1, -1, -1), v), dim=2)

pager = torch.zeros((B, L), dtype=torch.int16, device='cuda')
pager[:B, :L//2] = torch.arange(L//2, device='cuda').expand(B, -1)
#initialize rest of pages
pager[:B, L//2:] = torch.arange(B*L//2, device='cuda').reshape(B, L//2)+L//2

k_pages = torch.zeros((H, P, D), dtype=DTYPE_torch, device='cuda')
v_pages = torch.zeros((H, P, D), dtype=DTYPE_torch, device='cuda')
k_pages[:, :L//2, :] = prompt_k[0] # H, L//2, D
v_pages[:, :L//2, :] = prompt_v[0] # H, L//2, D
k_pages[:, L//2 : L//2+B*L//2, :] = k.transpose(0, 1).reshape(H, B*L//2, D)
v_pages[:, L//2 : L//2+B*L//2, :] = v.transpose(0, 1).reshape(H, B*L//2, D)

torch_output = F.scaled_dot_product_attention(q, torch_k, torch_v)
triton_output = page_flash_attn(pager, q.view(B, H, D), k_pages, v_pages, P, B, L, H, D).view(B, 1, H, D).transpose(1, 2)

if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["B"],  # Argument names to use as an x-axis for the plot
        x_vals=[1 * i for i in range(1, 100)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=["pytorch", "triton"],  # Label name for the lines
        line_names=["PyTorch", "Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="ms",  # Label name for the y-axis
        plot_name="paged_flash_attn",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
@triton.testing.perf_report(configs)
def benchmark(B, provider):
    L = 512
    H = 32
    D = 4096//H
    P = 2**(math.floor(math.log(L//2 + B*L//2, 2))+1)

    torch.manual_seed(1)
    q = torch.randn((B, 1, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
    k = torch.randn((B, L//2, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
    v = torch.randn((B, L//2, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
    prompt_k = torch.randn((1, L//2, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
    prompt_v = torch.randn((1, L//2, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)

    torch_k = torch.cat((prompt_k.expand(B, -1, -1, -1), k), dim=2)
    torch_v = torch.cat((prompt_v.expand(B, -1, -1, -1), v), dim=2)

    pager = torch.zeros((B, L), dtype=torch.int16, device='cuda')
    pager[:B, :L//2] = torch.arange(L//2, device='cuda').expand(B, -1)
    #initialize rest of pages
    pager[:B, L//2:] = torch.arange(B*L//2, device='cuda').reshape(B, L//2)+L//2

    k_pages = torch.zeros((H, P, D), dtype=DTYPE_torch, device='cuda')
    v_pages = torch.zeros((H, P, D), dtype=DTYPE_torch, device='cuda')
    k_pages[:, :L//2, :] = prompt_k[0] # H, L//2, D
    v_pages[:, :L//2, :] = prompt_v[0] # H, L//2, D
    k_pages[:, L//2 : L//2+B*L//2, :] = k.transpose(0, 1).reshape(H, B*L//2, D)
    v_pages[:, L//2 : L//2+B*L//2, :] = v.transpose(0, 1).reshape(H, B*L//2, D)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, torch_k, torch_v), 
                                                     quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: page_flash_attn(pager, q.view(B, H, D), k_pages, v_pages, P, B, L, H, D).view(B, 1, H, D).transpose(1, 2),
                                                                             quantiles=quantiles)
    perf = lambda ms: ms #2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True, save_path='./spec-mcts/stats/')


print("""Chunk Attention""")
# 101*256*4096*2(fp16)*2(KV) = 423MB
# 32*423MB = 13.5GB
B = 100
L = 512
H = 32
D = 4096//H
P = 2**(math.floor(math.log(L//2 + B*L//2, 2))+1)
print("P: ", P, "L//2: ", L//2)

torch.manual_seed(0)
q = torch.empty((B, 1, H, D), device='cuda', dtype=DTYPE_torch).normal_(mean=0.0, std=0.5)
k = torch.empty((B, L//2, H, D), device='cuda', dtype=DTYPE_torch).normal_(mean=0.0, std=0.5)
v = torch.empty((B, L//2, H, D), device='cuda', dtype=DTYPE_torch).normal_(mean=0.0, std=0.5)
prompt_k = torch.empty((1, L//2, H, D), device='cuda', dtype=DTYPE_torch).normal_(mean=0.0, std=0.5)
prompt_v = torch.empty((1, L//2, H, D), device='cuda', dtype=DTYPE_torch).normal_(mean=0.0, std=0.5)

q = q.transpose(1, 2).contiguous()
k = k.transpose(1, 2).contiguous()
v = v.transpose(1, 2).contiguous()
prompt_k = prompt_k.transpose(1, 2).contiguous()
prompt_v = prompt_v.transpose(1, 2).contiguous()

torch_k = torch.cat((prompt_k.expand(B, -1, -1, -1), k), dim=2)
torch_v = torch.cat((prompt_v.expand(B, -1, -1, -1), v), dim=2)

pager = torch.zeros((B, L), dtype=torch.int16, device='cuda')
pager[:B, :L//2] = torch.arange(L//2, device='cuda').expand(B, -1)
#initialize rest of pages
pager[:B, L//2:] = torch.arange(B*L//2, device='cuda').reshape(B, L//2)+L//2

k_pages = torch.zeros((H, P, D), dtype=DTYPE_torch, device='cuda')
v_pages = torch.zeros((H, P, D), dtype=DTYPE_torch, device='cuda')
k_pages[:, :L//2, :] = prompt_k[0] # H, L//2, D
v_pages[:, :L//2, :] = prompt_v[0] # H, L//2, D
k_pages[:, L//2 : L//2+B*L//2, :] = k.transpose(0, 1).reshape(H, B*L//2, D)
v_pages[:, L//2 : L//2+B*L//2, :] = v.transpose(0, 1).reshape(H, B*L//2, D)

torch_output = F.scaled_dot_product_attention(q, torch_k, torch_v)
triton_output = chunk_attn(pager, q.view(B, H, D), k_pages, v_pages, P, B, L, H, D, L//2).view(B, 1, H, D).transpose(1, 2)

if torch.allclose(triton_output, torch_output, atol=2e-1, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
    print(f"Max Diff={torch.max(torch.abs(triton_output - torch_output))}")
    print(f"Sum Diff={torch.sum(torch.abs(triton_output - torch_output))}")
    print(f"Mean Diff={torch.mean(torch.abs(triton_output - torch_output))}")
    #print(f"Diff={triton_output - torch_output}")

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["B"],  # Argument names to use as an x-axis for the plot
        x_vals=[1 * i for i in range(1, 64)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=["pytorch", "triton"],  # Label name for the lines
        line_names=["PyTorch", "Chunk+Page"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="ms",  # Label name for the y-axis
        plot_name="chunk_attn",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
@triton.testing.perf_report(configs)
def benchmark(B, provider):
    L = 512
    H = 32
    D = 4096//H
    P = 2**(math.floor(math.log(L//2 + B*L//2, 2))+1)

    torch.manual_seed(1)
    q = torch.randn((B, 1, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
    k = torch.randn((B, L//2, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
    v = torch.randn((B, L//2, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
    prompt_k = torch.randn((1, L//2, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)
    prompt_v = torch.randn((1, L//2, H, D), device='cuda', dtype=DTYPE_torch).transpose(1, 2)

    torch_k = torch.cat((prompt_k.expand(B, -1, -1, -1), k), dim=2)
    torch_v = torch.cat((prompt_v.expand(B, -1, -1, -1), v), dim=2)

    pager = torch.zeros((B, L), dtype=torch.int16, device='cuda')
    pager[:B, :L//2] = torch.arange(L//2, device='cuda').expand(B, -1)
    #initialize rest of pages
    pager[:B, L//2:] = torch.arange(B*L//2, device='cuda').reshape(B, L//2)+L//2

    k_pages = torch.zeros((H, P, D), dtype=DTYPE_torch, device='cuda')
    v_pages = torch.zeros((H, P, D), dtype=DTYPE_torch, device='cuda')
    k_pages[:, :L//2, :] = prompt_k[0] # H, L//2, D
    v_pages[:, :L//2, :] = prompt_v[0] # H, L//2, D
    k_pages[:, L//2 : L//2+B*L//2, :] = k.transpose(0, 1).reshape(H, B*L//2, D)
    v_pages[:, L//2 : L//2+B*L//2, :] = v.transpose(0, 1).reshape(H, B*L//2, D)
    print("B: ", B, "P: ", P, "L//2: ", L//2)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, torch_k, torch_v), 
                                                     quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: chunk_attn(pager, q.view(B, H, D), k_pages, v_pages, 
                                                                        P, B, L, H, D, L//2).view(B, 1, H, D).transpose(1, 2),
                                                                        quantiles=quantiles)
    perf = lambda ms: ms #2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

#benchmark.run(show_plots=True, print_data=True, save_path='./spec-mcts/stats/')

print("""Triton Attention""")
def test_op(B, H, L, D, dtype=torch.float16):
    #Z, H, N_CTX, D_HEAD = 1, 2, 1024, 64
    #causal = False
    torch.manual_seed(0)
    q = torch.empty((1, H, B, D), device='cuda', dtype=DTYPE_torch).normal_(mean=0.0, std=0.5)
    k = torch.empty((1, H, L, D), device='cuda', dtype=DTYPE_torch).normal_(mean=0.0, std=0.5)
    v = torch.empty((1, H, L, D), device='cuda', dtype=DTYPE_torch).normal_(mean=0.0, std=0.5)
    sm_scale = 1/math.sqrt(D)
    # reference implementation
    #p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    #p = torch.softmax(p.float(), dim=-1).half()
    #ref_out = torch.matmul(p, v)
    ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=sm_scale)
    # triton implementation
    tri_out = triton_attn(q, k, v, False, sm_scale).half()
    # compare
    if torch.allclose(ref_out, tri_out, atol=5e-1, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
        print(f"Diff={tri_out - ref_out}")
        print(tri_out.shape, ref_out.shape)
        print(f"Max Diff={torch.max(torch.abs(tri_out - ref_out))}")

test_op(100, 32, 512, 128, torch.float16)

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["B"],  # Argument names to use as an x-axis for the plot
        x_vals=[1 * i for i in range(1, 100)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=["pytorch", "triton"],  # Label name for the lines
        line_names=["PyTorch", "Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="ms",  # Label name for the y-axis
        plot_name="triton_attn",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
@triton.testing.perf_report(configs)
def benchmark(B, provider):
    L = 512
    H = 32
    D = 4096//H

    torch.manual_seed(1)
    q = torch.randn((1, H, B, D), device='cuda', dtype=DTYPE_torch)
    k = torch.randn((1, H, L, D), device='cuda', dtype=DTYPE_torch)
    v = torch.randn((1, H, L, D), device='cuda', dtype=DTYPE_torch)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_attn(q, k, v, False, 1/math.sqrt(D)), quantiles=quantiles)
    perf = lambda ms: ms #2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

#benchmark.run(show_plots=True, print_data=True, save_path='./spec-mcts/stats/')
