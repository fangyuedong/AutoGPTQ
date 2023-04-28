import math
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.cuda.amp import custom_bwd, custom_fwd
from logging import getLogger

logger = getLogger(__name__)

try:
    import triton
    import triton.language as tl
    from .triton_utils import custom_autotune

    # code based https://github.com/fpgaminer/GPTQ-triton

    @custom_autotune.autotune(
        configs=[
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=4,
                num_warps=4
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=4,
                num_warps=4
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=4,
                num_warps=4
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=4,
                num_warps=4
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=4,
                num_warps=4
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                num_stages=2,
                num_warps=8
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                num_stages=3,
                num_warps=8
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                num_stages=2,
                num_warps=4
            ),
        ],
        key=['M', 'N', 'K'],
        nearest_power_of_two=True,
        prune_configs_by={
            'early_config_prune': custom_autotune.matmul248_kernel_config_pruner,
            'perf_model': None,
            'top_k': None,
        },
    )
    @triton.jit
    def matmul_248_kernel(
        a_ptr, b_ptr, c_ptr,
        scales_ptr, zeros_ptr, g_ptr,
        M, N, K,
        bits, maxq,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_scales, stride_zeros,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
    ):
        """
        Compute the matrix multiplication C = A x B.
        A is of shape (M, K) float16
        B is of shape (K//8, N) int32
        C is of shape (M, N) float16
        scales is of shape (G, N) float16
        zeros is of shape (G, N) float16
        g_ptr is of shape (K) int32
        """
        infearure_per_bits = 32 // bits

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_mask = (offs_am[:, None] < M)
        # b_ptrs is set up such that it repeats elements along the K axis 8 times
        b_ptrs = b_ptr + (
            (offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn
        )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        g_ptrs = g_ptr + offs_k
        # shifter is used to extract the N bits of each element in the 32-bit word from B
        scales_ptrs = scales_ptr + offs_bn[None, :]
        zeros_ptrs = zeros_ptr + (offs_bn[None, :] // infearure_per_bits)

        shifter = (offs_k % infearure_per_bits) * bits
        zeros_shifter = (offs_bn % infearure_per_bits) * bits
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, num_pid_k):
            g_idx = tl.load(g_ptrs)

            # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
            scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

            zeros = (zeros >> zeros_shifter[None, :]) & maxq
            zeros = (zeros + 1)

            a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
            b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

            # Now we need to unpack b (which is N-bit values) into 32-bit values
            b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
            b = (b - zeros) * scales  # Scale and shift

            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
            g_ptrs += BLOCK_SIZE_K

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


    @custom_autotune.autotune(configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8},
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
            num_stages=2,
            num_warps=8
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
            num_stages=3,
            num_warps=8
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=2,
            num_warps=4
        ),
    ],
        key=['M', 'N', 'K'],
        nearest_power_of_two=True
    )
    @triton.jit
    def transpose_matmul_248_kernel(
        a_ptr, b_ptr, c_ptr,
        scales_ptr, zeros_ptr, g_ptr,
        M, N, K,
        bits, maxq,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_scales, stride_zeros,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
    ):
        """
        Compute the matrix multiplication C = A x B.
        A is of shape (M, N) float16
        B is of shape (K//8, N) int32
        C is of shape (M, K) float16
        scales is of shape (G, N) float16
        zeros is of shape (G, N) float16
        g_ptr is of shape (K) int32
        """
        infearure_per_bits = 32 // bits

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_k
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_k = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_n[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        a_mask = (offs_am[:, None] < M)
        # b_ptrs is set up such that it repeats elements along the K axis 8 times
        b_ptrs = b_ptr + (
            (offs_bk[:, None] // infearure_per_bits) * stride_bk + offs_n[None, :] * stride_bn
        )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        g_ptrs = g_ptr + offs_bk
        g_idx = tl.load(g_ptrs)

        # shifter is used to extract the N bits of each element in the 32-bit word from B
        scales_ptrs = scales_ptr + offs_n[None, :] + g_idx[:, None] * stride_scales
        zeros_ptrs = zeros_ptr + (offs_n[None, :] // infearure_per_bits) + g_idx[:, None] * stride_zeros

        shifter = (offs_bk % infearure_per_bits) * bits
        zeros_shifter = (offs_n % infearure_per_bits) * bits
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

        for k in range(0, num_pid_n):
            # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
            scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

            zeros = (zeros >> zeros_shifter[None, :]) & maxq
            zeros = (zeros + 1)

            a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
            b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

            # Now we need to unpack b (which is N-bit values) into 32-bit values
            b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
            b = (b - zeros) * scales  # Scale and shift
            b = tl.trans(b)

            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_N
            b_ptrs += BLOCK_SIZE_N
            scales_ptrs += BLOCK_SIZE_N
            zeros_ptrs += (BLOCK_SIZE_N // infearure_per_bits)

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bk[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bk[None, :] < K)
        tl.store(c_ptrs, accumulator, mask=c_mask)
except ImportError:
    logger.warning('triton not installed.')
    raise


def matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output = torch.empty((input.shape[0], qweight.shape[1]), device='cuda', dtype=torch.float16)
        grid = lambda META: (
            triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(qweight.shape[1], META['BLOCK_SIZE_N']),
        )
        matmul_248_kernel[grid](
            input, qweight, output,
            scales, qzeros, g_idx,
            input.shape[0], qweight.shape[1], input.shape[1],
            bits, maxq,
            input.stride(0), input.stride(1),
            qweight.stride(0), qweight.stride(1),
            output.stride(0), output.stride(1),
            scales.stride(0), qzeros.stride(0)
        )
        return output


def transpose_matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output_dim = (qweight.shape[0] * 32) // bits
        output = torch.empty((input.shape[0], output_dim), device='cuda', dtype=torch.float16)
        grid = lambda META: (
            triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(output_dim, META['BLOCK_SIZE_K']),)
        transpose_matmul_248_kernel[grid](
            input, qweight, output,
            scales, qzeros, g_idx,
            input.shape[0], qweight.shape[1], output_dim,
            bits, maxq,
            input.stride(0), input.stride(1),
            qweight.stride(0), qweight.stride(1),
            output.stride(0), output.stride(1),
            scales.stride(0), qzeros.stride(0)
        )
        return output


class QuantLinearFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = transpose_matmul248(grad_output, qweight, scales, qzeros, g_idx, bits, maxq)
        return grad_input, None, None, None, None, None, None


class QuantLinear(nn.Module):
    def __init__(
        self,
        bits,
        groupsize,
        infeatures,
        outfeatures,
        bias
    ):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        if infeatures % 256 != 0 or outfeatures % 256 != 0:
            raise NotImplementedError("in_feature or out_feature must be divisible by 256.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.groupsize = groupsize if groupsize != -1 else infeatures
        self.maxq = 2 ** self.bits - 1

        self.register_buffer(
            'qweight',
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32)
        )
        self.register_buffer(
            'qzeros',
            torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures // 32 * self.bits), dtype=torch.int32)
        )
        self.register_buffer(
            'scales',
            torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16)
        )
        self.register_buffer(
            'g_idx',
            torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32)
        )
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def pack(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()
    
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (
                        W[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        out = QuantLinearFunction.apply(
            x.reshape(-1, x.shape[-1]),
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.bits,
            self.maxq
        )
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out


def autotune_warmup_linear(model, transpose=False, seqlen=2048):
    """
    Pre-tunes the quantized kernel
    """
    from tqdm import tqdm

    kn_values = {}

    for _, m in model.named_modules():
        if not isinstance(m, QuantLinear):
            continue

        k = m.infeatures
        n = m.outfeatures

        if (k, n) not in kn_values:
            kn_values[(k, n)] = (m.qweight.cuda(), m.scales.cuda(), m.qzeros.cuda(), m.g_idx.cuda(), m.bits, m.maxq)

    logger.info(f'Found {len(kn_values)} unique KN Linear values.')
    logger.info('Warming up autotune cache ...')
    with torch.no_grad():
        for m in tqdm(range(0, math.ceil(math.log2(seqlen)) + 1)):
            m = 2 ** m
            for (k, n), (qweight, scales, qzeros, g_idx, bits, maxq) in kn_values.items():
                a = torch.randn(m, k, dtype=torch.float16, device='cuda')
                matmul248(a, qweight, scales, qzeros, g_idx, bits, maxq)
                if transpose:
                    a = torch.randn(m, n, dtype=torch.float16, device='cuda')
                    transpose_matmul248(a, qweight, scales, qzeros, g_idx, bits, maxq)
    del kn_values


__all__ = [
    "QuantLinear",
    "autotune_warmup_linear"
]
