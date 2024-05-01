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
        #tl.static_print(b_w.shape, b_s0.shape, b_s1.shape)
        b_MSB0 = ((b_w0>>4).to(DTYPE_triton) * b_s0)
        b_LSB1 = (((b_w0<<28)>>28).to(DTYPE_triton) * b_s1)
        #tl.static_print(b_MSB.shape, b_LSB.shape)

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
    c = torch.zeros((M, N), device=a.device, dtype=DTYPE_torch)
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
    for l in range(0, L//C):
        #b*H*l*D
        new_L = l*C+tl.arange(0, C)[None, :]
        mask = new_L < L
        p = tl.load(p_ptr + b*p_stride0 + new_L, mask = mask) #[B, L]
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
    tl.debug_barrier()
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
                            P, B, L, H, D, 16)
    return o #O doesn't need to be transposed either... since L=1
