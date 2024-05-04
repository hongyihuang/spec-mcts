import torch

import triton
import triton.language as tl
import math

DTYPE_torch = torch.float16
DTYPE_triton = tl.float16

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
    scale = scale.type(DTYPE_torch)
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
    #print(w.device, s.device, output.device)
    assert w.is_cuda and s.is_cuda and output.is_cuda

    #print(n_elements, group_size)
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
    assert w.is_contiguous()
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

def triton_q_int40_sliced(w, b_end, q_start, q_end):
    # We need to preallocate the output.
    n_elements = w.numel()
    int8val = torch.zeros((n_elements//2//64, 64), device=w.device, dtype=torch.int8)
    scale = torch.zeros((n_elements//64, 1), device=w.device, dtype=w.dtype)
    assert w.is_cuda and int8val.is_cuda and scale.is_cuda
    # assert group size is evenly divisible?
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    q_int40_sliced_kernel[grid](w, int8val, scale, n_elements, 
                                w.stride(0), b_end, w.stride(1), q_start, q_end,
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
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
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
    c = accumulator.to(DTYPE_triton)

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
    c = torch.empty((M, N), device=a.device, dtype=DTYPE_torch)
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
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn_w = tl.max_contiguous(tl.multiple_of(offs_bn_w, BLOCK_SIZE_N//2), BLOCK_SIZE_N//2)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    b_w_ptrs = b_w_ptr + (offs_k[:, None] * stride_bw_k + offs_bn_w[None, :] * stride_bw_n)
    b_s_ptrs = b_s_ptr + (offs_k[:, None] * stride_bs_k + offs_bn_s[None, :] * stride_bs_n)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N//2), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N//2), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        #a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        #b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_w0 = tl.load(b_w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        b_s0 = tl.load(b_s_ptrs)
        b_s1 = tl.load(b_s_ptrs + 1)
        b_LSB1 = (((b_w0<<28)>>28).to(DTYPE_triton) * b_s1)
        b_MSB0 = ((b_w0>>4).to(DTYPE_triton) * b_s0)

        # We accumulate along the K dimension.
        acc0 = tl.dot(a, b_MSB0, acc0)
        acc1 = tl.dot(a, b_LSB1, acc1)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_w_ptrs += BLOCK_SIZE_K * stride_bw_k
        b_s_ptrs += BLOCK_SIZE_K * stride_bs_k
        #b_s1_ptrs += BLOCK_SIZE_K * stride_bs_k
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N//2)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc0.to(DTYPE_triton), mask=c_mask)

    offs_cn += BLOCK_SIZE_N//2
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc1.to(DTYPE_triton), mask=c_mask)

def matmul_q40(a, b_w, b_s, b_shape):
    # Check constraints.
    #assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K_b, N = b_shape
    #print(M, K, K_b, N)
    assert K == K_b, "Incompatible dimensions"
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=DTYPE_torch)
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

@triton.jit
def flash_attn_kernel(
        # Pointers to matrices
        q_ptr, k_ptr, v_ptr, o_ptr, 
        scale, q_stride0, kv_stride0, kv_stride1, o_stride0,
        # Matrix dimensions
        B, L, H, D: tl.constexpr, C: tl.constexpr):
    """
    Q: [B, H, D]
    K: [B, H, L, D]
    V: [B, H, L, D]

    O: [B, H, D]
    No mask is needed as everything is synced to length L
    """
    # there are B*H programs, this kernel calculates across L*D
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)

    # Q[b, :] b*H*D  b*q_stride0
    q = tl.load(q_ptr + b*q_stride0 + h * D + tl.arange(0, D)[:, None])
    #Ls = tl.arange(0, L)
    #Ds = tl.arange(0, D)
    #offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    #x = tl.zeros((1,1), DTYPE_triton)
    last_m = tl.full((1,1), float('-inf'), DTYPE_triton)
    m = tl.full((1,1), float('-inf'), DTYPE_triton)
    last_d = tl.zeros((1,1), DTYPE_triton)
    d = tl.zeros((1,1), DTYPE_triton)
    last_o = tl.zeros((D,1), DTYPE_triton)
    o = tl.zeros((D,1), DTYPE_triton)
    offsetK = b * kv_stride0 + h * kv_stride1
    offsetV = b * kv_stride0 + h * kv_stride1

    for l in range(0, L):
        #b*H*l*D 
        #offset = b * H*l*D + H * l * D + h * D
        # K[l, :]
        k = tl.load(k_ptr + offsetK + tl.arange(0, D)[:, None])
        # V[l, :]
        v = tl.load(v_ptr + offsetV + tl.arange(0, D)[:, None])
        # dot product
        x = (tl.sum(q * k)*scale).to(tl.float16)
        m = tl.maximum(last_m, x)
        exp_m = last_d * tl.exp((last_m - m).to(tl.float32)).to(DTYPE_triton)
        exp_x = tl.exp((x - m).to(tl.float32)).to(DTYPE_triton)
        d = (exp_m + exp_x).to(DTYPE_triton)
        o = ((last_o * exp_m + exp_x * v)/d).to(DTYPE_triton)
        offsetK += D
        offsetV += D
        last_m = m
        last_d = d
        last_o = o
    # tl.debug_barrier()
    # b * H * D // * o_stride0
    tl.store(o_ptr + b * o_stride0 + h * D + tl.arange(0, D)[:, None], o)

@triton.jit
def flash_attn_kernel2(
        # Pointers to matrices
        q_ptr, k_ptr, v_ptr, o_ptr,
        scale, q_stride0, kv_stride0, kv_stride1, o_stride0,
        # Matrix dimensions
        B, L, H, D: tl.constexpr, C: tl.constexpr):
    """
    Q: [B, H, D]
    K: [B, H, L, D]
    V: [B, H, L, D]

    O: [B, H, D]
    No mask is needed as everything is synced to length L
    """
    # there are B*H programs, this kernel calculates across L*D
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    
    # Q[b, :] b*H*D  b*q_stride0
    q = tl.load(q_ptr + b*q_stride0 + h * D + tl.arange(0, D)[None, :] + tl.arange(0, 16)[:, None],
                mask = tl.arange(0, 16)[:, None] == 0)

    last_m = tl.full((1,1), float('-inf'), DTYPE_triton)
    m = tl.full((1,1), float('-inf'), DTYPE_triton)
    last_d = tl.zeros((1,1), DTYPE_triton)
    d = tl.zeros((1,1), DTYPE_triton)
    last_o = tl.zeros((D,1), DTYPE_triton)
    o = tl.zeros((D,1), DTYPE_triton)
    offsetK = b * kv_stride0  + h * kv_stride1 # [B, H, L, D]
    offsetV = b * kv_stride0 + h * kv_stride1

    for l in range(0, L//C):
        #b*H*l*D 
        #offset = b * H*l*D + H * l * D + h * D
        # K[l:l+C, :D]
        mask = l*C+tl.arange(0, C)[None, :] < L
        k = tl.load(k_ptr + offsetK + tl.arange(0, D)[:, None] + tl.arange(0, C)[None, :]*D,
                    mask = mask)
        # V[l:l+C, :D]
        v = tl.load(v_ptr + offsetV + tl.arange(0, D)[:, None] + tl.arange(0, C)[None, :]*D,
                    mask = mask)
        # dot product
        x = (tl.dot(q, k)*scale).to(DTYPE_triton)
        x = tl.sum(x, axis=0)
        m = tl.maximum(last_m, tl.max(x)).to(DTYPE_triton)
        exp_m = last_d * tl.exp((last_m - m).to(tl.float32)).to(DTYPE_triton)
        exp_x = tl.exp((x - m).to(tl.float32)).to(DTYPE_triton)
        d = (exp_m + tl.sum(exp_x, axis=1))
        o_x = tl.sum(exp_x * v, axis=1)
        o_m = tl.sum(last_o * exp_m, axis=1)
        o = (tl.expand_dims(o_m + o_x, 1)/d).to(DTYPE_triton)

        offsetK += C*D
        offsetV += C*D
        last_m = m
        last_d = d
        last_o = o
    #tl.debug_barrier()
    # b * H * D // * o_stride0
    tl.store(o_ptr + b * o_stride0 + h * D + tl.arange(0, D)[:, None], o)

def flash_attn(q, k, v, B, L, H, D): 
    """
    Q: [B, H, D]
    K: [B, H, L, D]
    V: [B, H, L, D]

    O: [B, H, D]
    No mask is needed as everything is synced to length L
    """
    # either store transposed or transpose here
    #k = k.transpose(1, 2)
    #v = v.transpose(1, 2)
    # q doesn't need to be transposed because L=1
    o = torch.empty((B, H, D), device=q.device, dtype=DTYPE_torch)
    
    # 1D launch kernel across B, H, parallelize L & D
    # typical sizes B=1-100, H = 32, so ~32-3200 kernels
    grid = lambda META: (B, H)
    #print(B, L, H, D)
    #print(q.stride(0), k.stride(0), v.stride(0), o.stride(0))
    #print(B*H*D, B*H*L*D, B*H*L*D, B*H*D)
    #print(q.shape, k.shape, v.shape, o.shape)

    flash_attn_kernel2[grid](q, k, v, o, 1/math.sqrt(D), 
                            q.stride(0), k.stride(0), k.stride(1), o.stride(0), 
                            B, L, H, D, 16)
    return o #O doesn't need to be transposed either... since L=1

@triton.jit
def page_flash_attn_kernel(
        # Pointers to matrices
        p_ptr, q_ptr, k_ptr, v_ptr, o_ptr,
        scale, p_stride0, q_stride0, k_stride0, v_stride0, o_stride0,
        # Matrix dimensions
        P, B, L, H, D: tl.constexpr):
    """
    Pager: [B, L]       int16
    Q: [B, H, D]        dtype
    K_pages: [H, P, D]  dtype
    V_pages: [H, P, D]  dtype

    O: [B, H, D]        dtype
    No mask is needed as everything is synced to length L
    """
    # there are B*H programs, this kernel calculates across L*D
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    
    # Q[b, :] b*H*D  b*q_stride0
    q = tl.load(q_ptr + b*q_stride0 + h * D + tl.arange(0, D)[:, None])

    last_m = tl.full((1,1), float('-inf'), DTYPE_triton)
    m = tl.full((1,1), float('-inf'), DTYPE_triton)
    last_d = tl.zeros((1,1), DTYPE_triton)
    d = tl.zeros((1,1), DTYPE_triton)
    last_o = tl.zeros((D,1), DTYPE_triton)
    o = tl.zeros((D,1), DTYPE_triton)
    for l in range(0, L):
        #b*H*l*D 
        p = tl.load(p_ptr + b*p_stride0 + l) #[B, L]
        offsetK = p * D + h * k_stride0 #[H, P, D] 
        offsetV = p * D + h * v_stride0 #[H, P, D] 
        # K[l, :]
        k = tl.load(k_ptr + offsetK + tl.arange(0, D)[:, None])
        # V[l, :]
        v = tl.load(v_ptr + offsetV + tl.arange(0, D)[:, None])
        # dot product
        x = (tl.sum(q * k)*scale).to(tl.float16)
        m = tl.maximum(last_m, x)
        d = (last_d * tl.exp((last_m - m).to(tl.float32)) + tl.exp((x - m).to(tl.float32))).to(DTYPE_triton)
        o = (last_o * last_d*tl.exp((last_m - m).to(tl.float32))/d + tl.exp((x-m).to(tl.float32))/d * v).to(DTYPE_triton)

        last_m = m
        last_d = d
        last_o = o
    tl.debug_barrier()
    # b * H * D // * o_stride0
    tl.store(o_ptr + b * o_stride0 + h * D + tl.arange(0, D)[:, None], o)

@triton.jit
def page_flash_attn_kernel2(
        # Pointers to matrices
        p_ptr, q_ptr, k_ptr, v_ptr, o_ptr,
        scale, p_stride0, q_stride0, kv_stride0, kv_stride1, o_stride0,
        # Matrix dimensions
        P, B, L, H, D: tl.constexpr, C: tl.constexpr):
    """
    Pager: [B, L]       int16
    Q: [B, H, D]        dtype
    K_pages: [H, P, D]  dtype
    V_pages: [H, P, D]  dtype

    O: [B, H, D]        dtype
    No mask is needed as everything is synced to length L
    """
    # there are B*H programs, this kernel calculates across L*D
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    
    # Q[b, :] b*H*D  b*q_stride0
    q = tl.load(q_ptr + b*q_stride0 + h * D + tl.arange(0, D)[None, :] + tl.arange(0, 16)[:, None],
                mask = tl.arange(0, 16)[:, None] == 0)

    last_m = tl.full((1,1), float('-inf'), DTYPE_triton)
    m = tl.full((1,1), float('-inf'), DTYPE_triton)
    last_d = tl.zeros((1,1), DTYPE_triton)
    d = tl.zeros((1,1), DTYPE_triton)
    last_o = tl.zeros((D,1), DTYPE_triton)
    o = tl.zeros((D,1), DTYPE_triton)
    for l in range(0, L, C):
        #b*H*l*D
        new_L = l + tl.arange(0, C) #[None, C]
        new_L = tl.max_contiguous(tl.multiple_of(new_L, C), C)[None, :]
        mask = new_L < L
        p = tl.load(p_ptr + b*p_stride0 + new_L, mask = mask) #[B, L]
        offsetKV = p * kv_stride1 + h * kv_stride0 #[H, P, D]
        # K[l:l+C, :D]
        off_k = k_ptr + offsetKV + tl.arange(0, D)[:, None] #[D, C]
        # tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
        #off_k = tl.max_contiguous(tl.multiple_of(off_k, D), D)
        k = tl.load(off_k, mask = mask)
        # V[l:l+C, :D]
        off_v = v_ptr + offsetKV + tl.arange(0, D)[:, None]
        #off_v = tl.max_contiguous(tl.multiple_of(off_v, D), D)
        v = tl.load(off_v, mask = mask)
        # dot product
        x = (tl.dot(q, k)*scale).to(DTYPE_triton)
        x = tl.sum(x, axis=0)
        m = tl.maximum(last_m, tl.max(x)).to(DTYPE_triton)
        exp_m = last_d * tl.exp((last_m - m).to(tl.float32)).to(DTYPE_triton)
        exp_x = tl.exp((x - m).to(tl.float32)).to(DTYPE_triton)
        d = (exp_m + tl.sum(exp_x, axis=1))
        o_x = tl.sum(exp_x * v, axis=1)
        o_m = tl.sum(last_o * exp_m, axis=1)
        o = (tl.expand_dims(o_m + o_x, 1)/d).to(DTYPE_triton)

        last_m = m
        last_d = d
        last_o = o
    #tl.debug_barrier()
    # b * H * D // * o_stride0
    tl.store(o_ptr + b * o_stride0 + h * D + tl.arange(0, D)[:, None], o)

def page_flash_attn(p, q, k, v, P, B, L, H, D): 
    """
    Pager: [B, L]       uint16
    Q: [B, H, D]        dtype
    K_pages: [H, P, D]  dtype
    V_pages: [H, P, D]  dtype

    O: [B, H, D]        dtype
    No mask is needed as everything is synced to length L
    """
    # either store transposed or transpose here
    #k = k.transpose(1, 2)
    #v = v.transpose(1, 2)
    # q doesn't need to be transposed because L=1
    o = torch.empty((B, H, D), device=q.device, dtype=DTYPE_torch)
    # 1D launch kernel across B, H, parallelize L & D
    # typical sizes B=1-100, H = 32, so ~32-3200 kernels
    grid = lambda META: (B, H)
    #print(B, L, H, D)
    #print(q.stride(0), k.stride(0), v.stride(0), o.stride(0))
    #print(B*H*D, B*H*L*D, B*H*L*D, B*H*D)

    page_flash_attn_kernel2[grid](p, q, k, v, o, 1/math.sqrt(D), 
                            p.stride(0), q.stride(0), k.stride(0), k.stride(1), o.stride(0), 
                            P, B, L, H, D, 32)
    return o #O doesn't need to be transposed either... since L=1

@triton.jit
def page_flash_attn_kernel3(
        # Pointers to matrices
        p_ptr, q_ptr, k_ptr, v_ptr, o_ptr,
        m_ptr, d_ptr,
        scale, p_stride0, q_stride0, kv_stride0, kv_stride1, 
        o_stride0, m_stride0, offset,
        # Matrix dimensions
        P, B, L, H, D: tl.constexpr, C: tl.constexpr):
    """
    Pager: [B, L]       int16
    Q: [B, H, D]        dtype
    K_pages: [H, P, D]  dtype
    V_pages: [H, P, D]  dtype

    O: [B, H, D]        dtype
    No mask is needed as everything is synced to length L
    """
    # there are B*H programs, this kernel calculates across L*D
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    
    # Q[b, :] b*H*D  b*q_stride0
    q = tl.load(q_ptr + b*q_stride0 + h * D + tl.arange(0, D)[None, :] + tl.arange(0, 16)[:, None],
                mask = tl.arange(0, 16)[:, None] == 0)
    m_ptr = m_ptr + b*m_stride0 + h
    d_ptr = d_ptr + b*m_stride0 + h

    #last_m = tl.full((1,1), float('-inf'), DTYPE_triton)
    last_m = tl.load(m_ptr)[None, None].to(DTYPE_triton)
    m = tl.full((1,1), float('-inf'), DTYPE_triton)
    #last_d = tl.zeros((1,1), DTYPE_triton)
    last_d = tl.load(d_ptr)[None, None].to(DTYPE_triton)
    d = tl.zeros((1,1), DTYPE_triton)
    last_o = tl.zeros((D,1), DTYPE_triton)
    o = tl.zeros((D,1), DTYPE_triton)
    for l in range(offset, L, C):
        #b*H*l*D
        l_range = l+tl.arange(0, C)[None, :]
        mask = l_range < L
        p = tl.load(p_ptr + b*p_stride0 + l_range, mask = mask) #[B, L]
        offsetKV = p * kv_stride1 + h * kv_stride0 #[H, P, D]
        # K[l:l+C, :D]
        k = tl.load(k_ptr + offsetKV + tl.arange(0, D)[:, None], mask = mask)
        # V[l:l+C, :D]
        v = tl.load(v_ptr + offsetKV + tl.arange(0, D)[:, None], mask = mask)
        # dot product
        x = (tl.dot(q, k)*scale).to(DTYPE_triton)
        x = tl.sum(x, axis=0)
        m = tl.maximum(last_m, tl.max(x)).to(DTYPE_triton)
        exp_m = last_d * tl.exp((last_m - m).to(tl.float32)).to(DTYPE_triton)
        exp_x = tl.exp((x - m).to(tl.float32)).to(DTYPE_triton)
        d = (exp_m + tl.sum(exp_x, axis=1))
        o_x = tl.sum(exp_x * v, axis=1)
        o_m = tl.sum(last_o * exp_m, axis=1)
        o = (tl.expand_dims(o_m + o_x, 1)/d).to(DTYPE_triton)

        last_m = m
        last_d = d
        last_o = o
    #tl.debug_barrier()
    # b * H * D // * o_stride0
    tl.store(o_ptr + b * o_stride0 + h * D + tl.arange(0, D)[:, None], o)

@triton.jit
def chunk_attn_kernel(
        # Pointers to matrices
        p_ptr, q_ptr, k_ptr, v_ptr, o_ptr,
        m_ptr, d_ptr,
        scale, p_stride0, q_stride0, kv_stride0, kv_stride1, 
        o_stride0, m_stride0,
        # Matrix dimensions
        P, B, L, H, D: tl.constexpr, C: tl.constexpr, B_BLOCK: tl.constexpr):
    """
    Pager: [B, L]       int16
    Q: [B, H, D]        dtype
    K_pages: [H, P, D]  dtype
    V_pages: [H, P, D]  dtype

    O: [B, H, D]        dtype
    No mask is needed as everything is synced to length L
    """
    # there are H programs, this kernel calculates across B*L*D
    b_pid = tl.program_id(axis=0)
    h = tl.program_id(axis=0)
    b = b_pid * B_BLOCK + tl.arange(0, B_BLOCK)
    b_mask = b < B
    
    # Q[b, :D] => [B', D]
    q = tl.load(q_ptr + b[:, None]*q_stride0 + h * D + tl.arange(0, D)[None, :],
                mask = b[:, None] < B)

    last_m = tl.full((1,B_BLOCK), float('-inf'), DTYPE_triton)
    m = tl.full((1,B_BLOCK), float('-inf'), DTYPE_triton)
    last_d = tl.zeros((1,B_BLOCK), DTYPE_triton)
    d = tl.zeros((1,B_BLOCK), DTYPE_triton)
    last_o = tl.zeros((D,B_BLOCK), DTYPE_triton)
    o = tl.zeros((D,B_BLOCK), DTYPE_triton)
    for l in range(0, L, C):
        #b*H*l*D
        l = tl.multiple_of(l, C)
        l_range = l+tl.arange(0, C)[None, :]
        mask = l_range < L
        offsetKV = l_range * kv_stride1 + h * kv_stride0 #[H, P, D] -> [1, C']
        # K[l:l+C, :D] => [D, C']
        k = tl.load(k_ptr + offsetKV + tl.arange(0, D)[:, None], mask = mask)
        # V[l:l+C, :D] => [D, C']
        v = tl.load(v_ptr + offsetKV + tl.arange(0, D)[:, None], mask = mask)
        #tl.static_print(k, v)
        # dot product
        x = (tl.dot(q, k)*scale).to(DTYPE_triton) #[B', C']
        #m = tl.maximum(last_m, tl.max(x, axis=1)).to(DTYPE_triton) #[1, B']
        #tl.static_print(x, k, v, p, m)

        # [1, B'], [C, B']
        exp_m = last_d * tl.exp((last_m - m).to(tl.float32)).to(DTYPE_triton)
        exp_x = tl.exp((tl.trans(x) - m).to(tl.float32)).to(DTYPE_triton)
        #tl.static_print(exp_m, exp_x)
        # [1, B'] + [1, B'] = [1, B']
        d = (exp_m + tl.expand_dims(tl.sum(exp_x, axis=0), 0))
        tl.static_print(d)
        #o_x = tl.dot(v, exp_x) #[D, C]x[C, B'] = [D, B']
        o_m = last_o * exp_m #[D, B']
        #tl.static_print(o_x, o_m)
        o = ((o_m)/d).to(DTYPE_triton)

        last_m = m
        last_d = d
        last_o = o
    tl.debug_barrier()
    # b * H * D // * o_stride0
    #tl.store(m_ptr + b[None, :] * m_stride0 + h, m)
    #tl.store(d_ptr + b[None, :] * m_stride0 + h, d)
    tl.store(o_ptr + b[None, :] * o_stride0 + h * D + tl.arange(0, D)[:, None], o,
             mask = b[None, :] < B)

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
# @triton.autotune(
#    configs=[
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
#        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
#        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
#        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=7, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=7, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=6, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=5, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=6, num_warps=4),
#    ],
#    key=['N_CTX'],
# )


@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, L, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H,  #
              N_CTX: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_offs = off_hz * N_CTX + offs_m
    tl.store(M + m_offs, m_i)
    tl.store(L + m_offs, l_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def triton_attn(q, k, v, causal, sm_scale):
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    assert Lq == Lk and (Lk == Lv or v.dtype == torch.float8_e5m2)
    assert Lk in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    BLOCK_M = 128
    BLOCK_N = 64 if Lk <= 64 else 32
    num_stages = 4 if Lk <= 64 else 3
    num_warps = 4
    num_stages -=1
    stage = 3 if causal else 1
    # Tuning for H100
    if torch.cuda.get_device_capability()[0] == 9:
        num_warps = 8
        num_stages = 7 if Lk >= 64 else 3
        if v.dtype == torch.float8_e5m2:
            if Lk < 256:
                BLOCK_M = 64 if not causal else 128
                BLOCK_N = 128
                num_stages = 3 if Lk == 128 else 4
                num_warps = 4
            else:
                BLOCK_M = 128
                BLOCK_N = 128
                num_stages = 3
                num_warps = 8
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    _attn_fwd[grid](
        q, k, v, sm_scale, M, L, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        q.shape[0], q.shape[1],  #
        N_CTX=q.shape[2],  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        BLOCK_DMODEL=Lk,  #
        STAGE=stage,  #
        num_warps=num_warps,  #
        num_stages=num_stages  #
    )
    
    return o

def chunk_attn(p, q, k, v, P, B, L, H, D, offset): 
    """
    Pager: [B, L]       uint16
    Q: [B, H, D]        dtype
    K_pages: [H, P, D]  dtype
    V_pages: [H, P, D]  dtype

    O: [B, H, D]        dtype
    No mask is needed as everything is synced to length L
    """
    q_T = q.transpose(0, 1).contiguous().view(1, H, B, D)
    k_T = k.view(1, H, P, D)[:, :, :offset, :]
    v_T = v.view(1, H, P, D)[:, :, :offset, :]
    # q doesn't need to be transposed because L=1

    # NO PAGING IN THE CHUNKING PART!
    # shape constraints
    Lq, Lk, Lv = q_T.shape[-1], k_T.shape[-1], v_T.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    assert Lq == Lk and (Lk == Lv or v_T.dtype == torch.float8_e5m2)
    assert Lk in {16, 32, 64, 128, 256}
    o = torch.empty_like(q_T)
    BLOCK_M = 128
    BLOCK_N = 64 if Lk <= 64 else 32
    num_stages = 4 if Lk <= 64 else 3
    num_warps = 4
    num_stages -=1
    stage = 1 #if not causal else 3
    # Tuning for H100
    if torch.cuda.get_device_capability()[0] == 9:
        num_warps = 8
        num_stages = 7 if Lk >= 64 else 3
        if v.dtype == torch.float8_e5m2:
            if Lk < 256:
                BLOCK_M = 64 #if not causal else 128
                BLOCK_N = 128
                num_stages = 3 if Lk == 128 else 4
                num_warps = 4
            else:
                BLOCK_M = 128
                BLOCK_N = 128
                num_stages = 3
                num_warps = 8
    grid = (triton.cdiv(q_T.shape[2], BLOCK_M), q_T.shape[0] * q_T.shape[1], 1)
    M = torch.empty((q_T.shape[0], q_T.shape[1], q_T.shape[2]),
                    device=q.device, dtype=torch.float32)
    D_mat = torch.empty((q_T.shape[0], q_T.shape[1], q_T.shape[2]),
                    device=q.device, dtype=torch.float32)

    _attn_fwd[grid](
        q_T, k_T, v_T, 1/math.sqrt(D), M, D_mat, o,  #
        q_T.stride(0), q_T.stride(1), q_T.stride(2), q_T.stride(3),  #
        k_T.stride(0), k_T.stride(1), k_T.stride(2), k_T.stride(3),  #
        v_T.stride(0), v_T.stride(1), v_T.stride(2), v_T.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        q_T.shape[0], q_T.shape[1],  #
        N_CTX=q.shape[2],  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        BLOCK_DMODEL=Lk,  #
        STAGE=stage,  #
        num_warps=num_warps,  #
        num_stages=num_stages  #
    )

    # Q = (B, H, S, D) (1, H, B, D)
    # O = (B, H, S, D) (1, H, B, D)
    # M = L = (B, H, S) (1, H, B)
    #o = torch.empty((B, H, D), device=q.device, dtype=DTYPE_torch)
    #m = torch.empty((B, H), device=q.device, dtype=DTYPE_torch)
    #d = torch.empty((B, H), device=q.device, dtype=DTYPE_torch)
    o = o.to(DTYPE_torch).reshape(H, B, D).transpose(0, 1).contiguous()
    m = M.to(DTYPE_torch).reshape(H, B).transpose(0, 1).contiguous()
    d = D_mat.to(DTYPE_torch).reshape(H, B).transpose(0, 1).contiguous()

    grid = lambda META: (B, H)
    #print(L, offset)
    page_flash_attn_kernel3[grid](p, q, k, v, o, m, d, 1/math.sqrt(D), 
                            p.stride(0), q.stride(0), k.stride(0), k.stride(1), 
                            o.stride(0), m.stride(0), offset,
                            P, B, L, H, D, 32)
    return o #O doesn't need to be transposed either... since L=1