import torch

import triton
import triton.language as tl

DTYPE = torch.bfloat16

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

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

benchmark.run(print_data=True, show_plots=True)

def quantize_q40(w, group_size):
    """
    takes a tensor and returns the Q4_0 quantized version
    i.e. symmetric quantization into int4, [-7, 7]
    """
    #assert w.numel() % group_size == 0
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 7.0
    scale = scale.type(DTYPE)
    scale = scale[:,None]
    # scale into range [-7, 7]
    quant = w / scale
    # round to nearest integer
    
    #assert(quant.max() <= 7)
    #assert(quant.min() >= -7)
    
    int8val = torch.round(quant).to(torch.int8)
    MSB = int8val.view(-1, 2, group_size)[:, 0, :]
    LSB = int8val.view(-1, 2, group_size)[:, 1, :]
    #assert(MSB.abs().max() <= 7)
    #assert(LSB.abs().min() >= -7)
    int8val = (MSB << 4) | (LSB & 0x0F)
    int8val = int8val.view(-1, group_size)

    return int8val, scale #, maxerr

@torch.jit.script
def dequantize_q40_jit(w, scale, group_size: int, ptdtype: torch.dtype):
    """
    takes a Q4_0 tensor and returns the dequantized version
    """
    # assume it is already packed by group_size
    # w = w.view(-1, group_size)

    MSB = w >> 4
    LSB = w << 4 >> 4 # DO NOT JUST MASK OUT THE MSB, SIGN EXT
    w = torch.hstack((MSB, LSB)).view(-1, group_size)
    # dequantize by rescaling
    fpval = (w * scale).type(ptdtype) #(w * scale.expand(-1, group_size)).type(ptdtype)

    return fpval

def dequantize_q40(w, scale, group_size: int, ptdtype: torch.dtype):
    """
    takes a Q4_0 tensor and returns the dequantized version
    """
    # assume it is already packed by group_size
    # w = w.view(-1, group_size)

    MSB = w >> 4
    LSB = w << 4 >> 4 # DO NOT JUST MASK OUT THE MSB, SIGN EXT
    w = torch.hstack((MSB, LSB)).view(-1, group_size)
    # dequantize by rescaling
    fpval = (w * scale).type(ptdtype) #(w * scale.expand(-1, group_size)).type(ptdtype)

    return fpval

@triton.jit
def deq_int40_kernel(   w_ptr,  # *Pointer* to first input vector.
                        s_ptr,  # *Pointer* to second input vector.
                        o_ptr,  # *Pointer* to output vector.
                        o_numel,  # Size of the output vector.
                        group_size,
                        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                        # NOTE: `constexpr` so it can be used as a shape value.
                        ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < o_numel
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    w = tl.load(w_ptr + offsets//2, mask=mask)
    s0 = tl.load(s_ptr)
    s1 = tl.load(s_ptr + group_size//2)

    # Write results back to DRAM.
    MSB = (w>>4) * s0
    LSB = ((w<<4)>>4) * s1
    
    tl.store(o_ptr + block_start + tl.arange(0, BLOCK_SIZE), MSB, mask=mask)
    #tl.store(o_ptr + block_start + tl.arange(0, BLOCK_SIZE)//2 + group_size//2, LSB, mask=mask)

def triton_deq_int40(w, s, group_size, shape, ptdtype):
    # We need to preallocate the output.
    n_elements = torch.prod(torch.tensor(shape)).item()
    output = torch.zeros(n_elements, device=w.device, dtype=ptdtype)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    # assert group size is evenly divisible?
    
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    deq_int40_kernel[grid](w, s, output, n_elements, group_size, BLOCK_SIZE=64)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output.view(shape)

torch.manual_seed(0)
shape = (256, 64)
w = torch.rand(shape, device='cuda', dtype=torch.bfloat16)
q_w, q_s = quantize_q40(w, 64)

output_torch = dequantize_q40(q_w, q_s, 64, torch.bfloat16).view(shape)
output_triton = triton_deq_int40(q_w, q_s, 64, shape, torch.bfloat16)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[1024*i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
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
    q_w, q_s = quantize_q40(w, 64)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: dequantize_q40_jit(q_w, q_s, 64, torch.bfloat16).view(shape), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: dequantize_q40(q_w, q_s, 64, torch.bfloat16).view(shape), quantiles=quantiles)
    #if provider == 'triton':
    #    ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_deq_int40(q_w, q_s, 64, shape, torch.bfloat16), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)
