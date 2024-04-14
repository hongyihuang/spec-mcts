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

def dequantize_q40(w, scale, group_size, shape, ptdtype):
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

    return fpval.view(shape)

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
    arange_half_BLOCK = tl.arange(0, BLOCK_SIZE//2)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    # mask = offsets < o_numel
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    w = tl.load(w_ptr + block_start//2 + arange_half_BLOCK).to(tl.int8)
    s0 = tl.load(s_ptr + block_start//64)
    s1 = tl.load(s_ptr + 1 + block_start//64)

    # Write results back to DRAM.
    MSB = (w>>4) * s0
    LSB = ((w<<28)>>28) * s1 # had to guess triton uses int32 in the backend...
    # total = tl.join(MSB, LSB) # can't join, this really kills perf
    # tl.static_print(w.shape, MSB.shape, LSB.shape)
    
    tl.store(o_ptr + block_start + arange_half_BLOCK, MSB)
    tl.store(o_ptr + block_start + arange_half_BLOCK + group_size, LSB)

def triton_deq_int40(w, s, group_size, shape, ptdtype):
    # We need to preallocate the output.
    n_elements = torch.prod(torch.tensor(shape)).item()
    output = torch.zeros(n_elements, device=w.device, dtype=ptdtype)
    assert w.is_cuda and s.is_cuda and output.is_cuda
    # assert group size is evenly divisible?
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    deq_int40_kernel[grid](w, s, output, n_elements, group_size, BLOCK_SIZE=128)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output.view(shape)

@triton.jit
def q_int40_kernel( w_ptr,  # *Pointer* to weight input vector.
                    q_ptr,  # *Pointer* to quantized output vector.
                    s_ptr,  # *Pointer* to scale output vector.
                    w_numel,  # Size of the weight input vector.
                    GROUP_SIZE: tl.constexpr,
                    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                    ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    # Create a mask to guard memory operations against out-of-bounds accesses.
    # mask = offsets < w_numel
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    
    result = tl.ravel(tl.zeros((1, GROUP_SIZE), dtype=tl.int8))
    for i in range(0, tl.cdiv(BLOCK_SIZE, GROUP_SIZE)):
        w = tl.load(w_ptr + block_start + tl.arange(0, BLOCK_SIZE//2) + i*GROUP_SIZE)
        s = (tl.max(tl.abs(w)) / 7.0)#.to(tl.bfloat16)
        quant = (w/s + 0.5).to(tl.int8) # there are occasional off by 1 rounding errors?
        result = ((result << 4) | (quant & 0x0F)).to(tl.int8)
        tl.store(s_ptr + block_start//64 + i, s)
        if (i % 2 == 1):
            tl.store(q_ptr + block_start//2 + tl.arange(0, BLOCK_SIZE//2), result)

def triton_q_int40(w, group_size):
    # We need to preallocate the output.
    w = w.contiguous() # force contiguity by copying data if sliced
    n_elements = w.numel()
    int8val = torch.zeros((n_elements//2//64, 64), device=w.device, dtype=torch.int8)
    scale = torch.zeros((n_elements//64, 1), device=w.device, dtype=w.dtype)
    assert w.is_cuda and int8val.is_cuda and scale.is_cuda
    # assert group size is evenly divisible?
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    q_int40_kernel[grid](w, int8val, scale, n_elements, GROUP_SIZE=64, BLOCK_SIZE=128)
    return int8val, scale

@triton.jit
def q_int40_sliced_kernel(
        w_ptr,  # *Pointer* to weight input vector.
        q_ptr,  # *Pointer* to quantized output vector.
        s_ptr,  # *Pointer* to scale output vector.
        w_numel,  # Size of the weight input vector.
        batch_stride, batch_end,
        seq_stride, seq_start, seq_end,
        GROUP_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
        ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    # Create a mask to guard memory operations against out-of-bounds accesses.
    # mask = offsets < w_numel
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    
    result = tl.ravel(tl.zeros((1, GROUP_SIZE), dtype=tl.int8))
    for i in range(0, tl.cdiv(BLOCK_SIZE, GROUP_SIZE)):
        w = tl.load(w_ptr + block_start + tl.arange(0, BLOCK_SIZE//2) + i*GROUP_SIZE)
        s = (tl.max(tl.abs(w)) / 7.0)#.to(tl.bfloat16)
        quant = (w/s + 0.5).to(tl.int8) # there are occasional off by 1 rounding errors?
        result = ((result << 4) | (quant & 0x0F)).to(tl.int8)
        tl.store(s_ptr + block_start//64 + i, s)
        if (i % 2 == 1):
            tl.store(q_ptr + block_start//2 + tl.arange(0, BLOCK_SIZE//2), result)

def triton_q_int40_sliced(w, b_stride, b_end, q_stride, q_start, q_end):
    # We need to preallocate the output.
    n_elements = w.numel()
    int8val = torch.zeros((n_elements//2//64, 64), device=w.device, dtype=torch.int8)
    scale = torch.zeros((n_elements//64, 1), device=w.device, dtype=w.dtype)
    assert w.is_cuda and int8val.is_cuda and scale.is_cuda
    # assert group size is evenly divisible?
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    q_int40_sliced_kernel[grid](w, int8val, scale, n_elements, 
                                b_stride, b_end, q_stride, q_start, q_end,
                                GROUP_SIZE=64, BLOCK_SIZE=128)
    return int8val, scale

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    #tl.static_print(GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)

def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
    ],
    key=['M', 'N', 'K'],
)

@triton.jit
def matmul_q40_kernel(
        # Pointers to matrices
        a_ptr, b_w_ptr, b_s_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bw_k, stride_bw_n,  #
        stride_bs_k, stride_bs_n,  #
        stride_cm, stride_cn,
        # Meta-parameters
        # BLOCK_SIZE_N: tl.constexpr # forced to be 128 to make writing kernels easier
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr, #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn_w = (pid_n * BLOCK_SIZE_N//2 + tl.arange(0, BLOCK_SIZE_N//2)) % (N//2)
    offs_bn_s = (pid_n * BLOCK_SIZE_N//64) % (N//64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    b_w_ptrs = b_w_ptr + (offs_k[:, None] * stride_bw_k + offs_bn_w[None, :] * stride_bw_n)
    #b_s_ptrs = b_s_ptr + (offs_k[:, None] * stride_bs_k + offs_bn_s[None, :] * stride_bs_n)
    b_s0_ptrs = b_s_ptr + (offs_k[:, None] * stride_bs_k + offs_bn_s[None, :] * stride_bs_n)
    #b_s1_ptrs = b_s_ptr + (offs_k[:, None] * stride_bs_k + (offs_bn_s[None, :]+1) * stride_bs_n)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N//2), dtype=tl.float32)
    accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N//2), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        #a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        #b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_w = tl.load(b_w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        b_s0 = tl.load(b_s0_ptrs)
        b_s1 = tl.load(b_s0_ptrs + 1)
        #tl.static_print(b_w.shape, b_s0.shape, b_s1.shape)
        b_MSB = ((b_w>>4) * b_s0).to(tl.bfloat16)
        b_LSB = (((b_w<<28)>>28) * b_s1).to(tl.bfloat16)
        #tl.static_print(b_MSB.shape, b_LSB.shape)

        # We accumulate along the K dimension.
        accumulator0 = tl.dot(a, b_MSB, accumulator0)
        accumulator1 = tl.dot(a, b_LSB, accumulator1)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_w_ptrs += BLOCK_SIZE_K * stride_bw_k
        b_s0_ptrs += BLOCK_SIZE_K * stride_bs_k
        #b_s1_ptrs += BLOCK_SIZE_K * stride_bs_k
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c0 = accumulator0.to(tl.bfloat16)
    c1 = accumulator1.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N//2)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c0, mask=c_mask)

    offs_cn += BLOCK_SIZE_N//2
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c1, mask=c_mask)

def matmul_q40(a, b_w, b_s, b_shape):
    # Check constraints.
    #assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K_b, N = b_shape
    #print(M, K, K_b, N)
    assert K == K_b, "Incompatible dimensions"
    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=torch.bfloat16)
    b_w = b_w.view(K_b, N//2)
    b_s = b_s.view(K_b, N//64)
    """
    print(a.stride(0), a.stride(1),  #
        b_w.stride(0), b_w.stride(1),  #
        b_s.stride(0), b_s.stride(1),  #
        c.stride(0), c.stride(1))
    """
    # 2D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_q40_kernel[grid](
        a, b_w, b_s, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b_w.stride(0), b_w.stride(1),  #
        b_s.stride(0), b_s.stride(1),  #
        c.stride(0), c.stride(1),  #
        #BLOCK_SIZE_N = 128,
    )
    return c