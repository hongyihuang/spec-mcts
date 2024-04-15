import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from triton_kernels import *
from triton_matmul import matmul_split_k

"""Addition"""
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

#benchmark.run(print_data=True, show_plots=True)

"""De-quantization"""
torch.manual_seed(0)
shape = (256, 64)
w = torch.rand(shape, device='cuda', dtype=torch.bfloat16)
q_w, q_s = quantize_q40(w, 64)

output_torch = dequantize_q40(q_w, q_s, 64, shape, torch.bfloat16)
output_triton = triton_deq_int40(q_w, q_s, 64, shape, torch.bfloat16)

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
    w = torch.rand(shape, device='cuda', dtype=torch.bfloat16)
    q_w, q_s = quantize_q40(w, 64)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: dequantize_q40(q_w, q_s, 64, shape, torch.bfloat16), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_deq_int40(q_w, q_s, 64, shape, torch.bfloat16), quantiles=quantiles)
    gbps = lambda ms: size*64*4 / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

#benchmark.run(print_data=True, show_plots=True)

"""Quantization"""
torch.manual_seed(0)
shape = (128, 256, 64)
sliced_shape = (64, 128, 64)
w = torch.rand(shape, device='cuda', dtype=torch.bfloat16)
sliced_tensor = w[64:128, 32:32+128, :64]
q_w, q_s = quantize_q40(sliced_tensor, 64)
triton_q_w, triton_q_s = triton_q_int40(sliced_tensor, 64)

print(q_w - triton_q_w)
print(q_s.flatten() - triton_q_s.flatten())

output_torch = dequantize_q40(q_w, q_s, 64, sliced_shape, torch.bfloat16)
output_triton = triton_deq_int40(triton_q_w, triton_q_s, 64, sliced_shape, torch.bfloat16)

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
    w = torch.rand(shape, device='cuda', dtype=torch.bfloat16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: quantize_q40(w[size//2:size], 64), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_q_int40(w[size//2:size], 64), quantiles=quantiles)
    gbps = lambda ms: size*64*4/2 / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

#benchmark.run(print_data=True, show_plots=True)

"""Matmul"""
torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.bfloat16)
b = torch.randn((512, 512), device='cuda', dtype=torch.bfloat16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8:
    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
    a = a.to(torch.float8_e5m2)
    # pre-transpose b for efficiency.
    b = b.T
    b = b.to(torch.float8_e5m2)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a.to(torch.bfloat16), b.to(torch.bfloat16))
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
    a = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((K, N), device='cuda', dtype=torch.bfloat16)
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


#benchmark.run(show_plots=True, print_data=True)

"""Matmul FP16 x int4_g64"""
torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.bfloat16)
b = torch.randn((512, 512), device='cuda', dtype=torch.bfloat16)
q, s = quantize_q40(b, 64)

triton_output = matmul_q40(a, q, s, b.shape)
deq_b = dequantize_q40(q, s, 64, b.shape, torch.bfloat16)
torch_output = torch.matmul(a, deq_b)

print(f"triton_output_bf16 x int4={triton_output}")
print(f"torch_output_with_bf16_inputs={torch_output}")
#print(f"triton_output_bf16 x int4[0]={triton_output[0]}")
#print(f"diff[0]={triton_output[0]-torch_output[0]}")

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
        plot_name="matmul_q40-performance",
        args={},
    ))
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    M = 80
    #print(M, N, K)
    a = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((K, N), device='cuda', dtype=torch.bfloat16)
    q, s = quantize_q40(b, 64)
    b = dequantize_q40(q, s, 64, b.shape, torch.bfloat16)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
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

benchmark.run(show_plots=True, print_data=True, save_path='./spec-mcts/stats/')


"""Matmul split-k"""
torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.bfloat16)
b = torch.randn((512, 512), device='cuda', dtype=torch.bfloat16)

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
    a = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((K, N), device='cuda', dtype=torch.bfloat16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_split_k(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

#benchmark.run(show_plots=True, print_data=True, save_path='./spec-mcts/stats/')
