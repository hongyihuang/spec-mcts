import torch

import triton
import triton.language as tl

DTYPE = torch.bfloat16

'''add'''
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

''' QUANTIZATION '''

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
    o_numel//64
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < o_numel
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    w = tl.load(w_ptr + offsets//2, mask=mask).to(tl.int8)
    s0 = tl.load(s_ptr + block_start//64)
    s1 = tl.load(s_ptr + 1 + block_start//64)

    # Write results back to DRAM.
    MSB = (w>>4) * s0
    LSB = ((w<<28)>>28) * s1 # had to guess triton uses int32 in the backend...
    # total = tl.join(MSB, LSB) # can't join, this really kills perf
    
    tl.store(o_ptr + block_start + tl.arange(0, BLOCK_SIZE)//2, MSB, mask=mask)
    tl.store(o_ptr + block_start + tl.arange(0, BLOCK_SIZE)//2 + group_size, LSB, mask=mask)

def triton_deq_int40(w, s, group_size, shape, ptdtype):
    # We need to preallocate the output.
    n_elements = torch.prod(torch.tensor(shape)).item()
    output = torch.zeros(n_elements, device=w.device, dtype=ptdtype)
    assert w.is_cuda and s.is_cuda and output.is_cuda
    # assert group size is evenly divisible?
    
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    deq_int40_kernel[grid](w, s, output, n_elements, group_size, BLOCK_SIZE=128)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output.view(shape)

