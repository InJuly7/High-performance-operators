import time
from functools import partial
from typing import Optional
import onnx.helper as helper
import onnxruntime as ort
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.7"
os.environ["MAX_JOBS"] = "8"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


class Config:
    def __init__(self):
        self.dim1 = [4096, 8192]  # [4096,8192]
        self.dim2 = [256, 512, 1024, 2048, 4096, 8192]  # [256,512,1024,2048,4096,8192]
        self.kernels = {
            "rms_norm_f32": 0.0,
            "rms_norm_f32x4": 0.0,
            "rms_norm_f16": 0.0,
            "rms_norm_f16x2": 0.0,
            "rms_norm_f16x8": 0.0,
            "rms_norm_f16_pack": 0.0,
            "torch_rms_norm": 0.0,
        }  # kernel name : time in ms
        self.torch_out: torch.Tensor = None  # 用于存储torch的输出结果, 作为对比基准
        self.onnx_out: torch.Tensor = None  # 用于存储onnxruntime的输出结果, 作为对比基准
        self.tensors = [None] * len(self.kernels)
        self.dtypes = {"fp32": torch.float32, "fp16": torch.float16}
        self.compare = False
        self.iter = 1000
        self.warmup = 10
        self.device = "cuda:0"  # "cpu"
        self.rtol_fp16 = 1e-03
        self.rtol_fp32 = 1e-05
        self.atol_fp16 = 1e-05
        self.atol_fp32 = 1e-08
        # Load the CUDA kernel as a python module
        self.lib = load(
            name="rms_norm_lib",
            sources=[
                "./RMSNorms/bind.cpp",
                "./RMSNorms/rms_norm_f32.cu",
                "./RMSNorms/rms_norm_f32x4.cu",
                "./RMSNorms/rms_norm_f16.cu",
                "./RMSNorms/rms_norm_f16x2.cu",
                "./RMSNorms/rms_norm_f16x8.cu",
                "./RMSNorms/rms_norm_f16_pack.cu",
            ],  # 这里可以根据需要添加更多的源文件
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ],
            extra_cflags=["-std=c++17"],
        )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


"""
@func: 使用torch框架实现rms-norm, 并测试性能
@param: A : 输入Tensor
@param: g : 缩放参数
@param: B : 输出Tensor
@param: shape : 形状
@param: config: 配置 
"""


def rms_norm_torch(
    A: torch.Tensor = None,
    B: torch.Tensor = None,
    C: torch.Tensor = None,
    shape: list = [4096, 256],
    config: Config = None,
):
    assert A is not None, "Input tensor A cannot be None"
    assert B is not None, "Weight tensor B cannot be None"
    assert C is not None, "Output tensor C cannot be None"

    rms_norm = LlamaRMSNorm(hidden_size=shape[1], eps=1e-6).to(config.device).to(A.dtype)
    # 加载预训练权重
    rms_norm.weight.data = B
    rms_norm.eval()
    # warmup
    for _ in range(config.warmup):
        C = rms_norm(A)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    # iters
    for _ in range(config.iter):
        C = rms_norm(A)
    end_event.record()
    end_event.synchronize()
    # 计算平均时间
    total_time_ms = start_event.elapsed_time(end_event)
    mean_time = total_time_ms / config.iter
    # 展平, 从计算图中分离, 转cpu, 转numpy, 转list
    # 仅显示前三个元素
    out_val = C.flatten().detach().cpu().numpy().tolist()[:3]
    # 保留8位小数
    out_val = [round(v, 8) for v in out_val]
    config.kernels["torch_rms_norm"] = mean_time
    config.tensors[list(config.kernels.keys()).index("torch_rms_norm")] = out_val
    config.torch_out = C


"""
@func: 使用onnx.helper构建rms_norm算子, 使用onnxruntime库运行
@param: A: 输入tensor
@param: A_str: 输入tensor name
@param: B: 输出tensor
@param: B_str: 输出tensor name
@param: shape: 形状
@param: config: 配置
"""


def rms_norm_onnx(
    A: torch.Tensor = None,
    A_str: str = "input",
    B: torch.Tensor = None,
    B_str: str = "output",
    shape: list = [4096, 512],
    config: Config = None,
):
    pass


"""
@func: 运行基准测试
@param: perf_func : callable (可调用的kenrel函数)
@param: A : 输入tensor
@param: B : 输出tensor
@param: kernel : 核函数名称
@param: show_all : 是否显示全部输出
@param: config : 配置文件
"""


def run_benchmark(
    perf_func: callable,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    kernel: str = None,
    config: Config = None,
):
    # 初始化 输出
    if C is not None:
        C.fill_(0)
    # warmup
    for _ in range(config.warmup):
        perf_func(A, B, C)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    # iters
    for i in range(config.iter):
        perf_func(A, B, C)
    end_event.record()
    end_event.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)
    mean_time = total_time_ms / config.iter
    # 展平, 从计算图中分离, 转cpu, 转numpy, 转list
    # 仅显示前三个元素
    out_val = C.flatten().detach().cpu().numpy().tolist()[:3]
    # 保留8位小数
    out_val = [round(v, 8) for v in out_val]

    config.kernels[kernel] = mean_time
    config.tensors[list(config.kernels.keys()).index(kernel)] = out_val
    if config.compare:
        if compare_tensor(C, config.torch_out, config=config):
            print(f"{kernel} Results RIGHT!")


"""
@func: 创建一个 torch Tensor
@param: shape: 形状
@param: dtype: 数据类型
@param: ndim: 维度数量
@param: device: 设备名称
@return: torch.Tensor
"""


def create_tensor(shape, dtype=torch.float32, ndim=2, device="cpu"):
    print(f"device = {device}, dtype = {dtype}, shape = {shape}, ndim = {ndim}")
    if ndim == 1:
        return torch.randn((shape[0]), device=device).to(dtype).contiguous()
    elif ndim == 2:
        return torch.randn((shape[0], shape[1]), device=device).to(dtype).contiguous()
    elif ndim == 3:
        return torch.randn((shape[0], shape[1], shape[2]), device=device).to(dtype).contiguous()
    else:
        raise ValueError("Unsupported ndim")


"""
@func: 比较两个tensor是否相等
@param: A : 第一个tensor
@param: B : 第二个tensor
@return: bool
"""


# ∣A − B∣ ≤ atol + (rtol × ∣B∣)
def compare_tensor(A: torch.Tensor, B: torch.Tensor, config: Config = None) -> bool:
    if config is None:
        config = Config()  # 使用默认值

    # 检查形状
    if A.shape != B.shape:
        print(f"Shape mismatch: {A.shape} vs {B.shape}")
        return False

    # 使用 torch.isclose 生成逐元素的对比掩码, is_close 是一个全是 True/False 的 Tensor
    if A.dtype == torch.float16:
        is_close = torch.isclose(A, B, rtol=config.rtol_fp16, atol=config.atol_fp16)
    elif A.dtype == torch.float32:
        is_close = torch.isclose(A, B, rtol=config.rtol_fp32, atol=config.atol_fp32)

    # 如果全部都 Close，则通过
    if torch.all(is_close):
        return True

    # 定位错误
    print("Tensor values are not close enough.")

    # 找到不一致的位置 (False 的位置)
    mismatch_indices = torch.nonzero(~is_close, as_tuple=False)
    num_mismatches = mismatch_indices.shape[0]
    total_elements = A.numel()

    print(f"   Mismatched elements: {num_mismatches} / {total_elements} ({(num_mismatches/total_elements)*100:.2f}%)")

    # 计算最大绝对误差
    diff = torch.abs(A - B)
    max_diff = torch.max(diff)
    print(f"   Max absolute difference: {max_diff.item()}")

    # 4. 打印前 N 个具体的错误位置供调试
    print("\n   --- First 5 Mismatches ---")
    for i in range(min(5, num_mismatches)):
        idx = mismatch_indices[i]  # 获取由维度组成的索引，如 [0, 2, 1]

        # 将 tensor 索引转为 tuple 以便用于访问
        idx_tuple = tuple(idx.tolist())

        val_a = A[idx_tuple].item()
        val_b = B[idx_tuple].item()
        abs_err = abs(val_a - val_b)

        print(f"   Index {idx_tuple}:")
        print(f"     A: {val_a}")
        print(f"     B: {val_b}")
        print(f"     Diff: {abs_err}")

    return False


"""
@func: 重置config.tensor, 每个kernel的运行时间
"""


def clear_status(config: Config = None):
    config.tensors = [None] * len(config.kernels)
    for k, v in config.kernels.items():
        config.kernels[k] = 0.0


"""
@func: 打印性能表格
@param: 输入shape : list
@param: 输出shape : list
@param: 数据类型: str
@param: {kernel_name : time_in_ms}: dict
@param: tensors: list
@param: 描述信息: str
@return: None
"""


def print_perf(
    input_shape: list,
    output_shape: list,
    config: Config = None,
    info: str = "",
    kernels: list = None,
):
    print(f"{'-'*100}")
    if info != "":
        left_space = (100 - len(info)) // 2
        print(" " * left_space + f"{info}")
    shape_info = f"S={input_shape[0]}, H={input_shape[1]}"
    left_space = (100 - len(shape_info)) // 2
    print(" " * left_space + f"{shape_info}")
    print("-" * 100)
    for val, (kernel, times) in zip(config.tensors, config.kernels.items()):
        if val is None:
            continue
        if kernels is not None and kernel not in kernels:
            continue
        # 格式化输出, 12宽度左对齐
        out_val = [f"{v:<12}" for v in val]
        # 24宽度右对齐
        print(f"{kernel:>24}: {out_val}, time:{times:.8f}ms")
    print(f"{'-'*100}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    config = Config()
    lib = config.lib
    S = config.dim1
    H = config.dim2
    print(f"S={S}, H={H}")
    for dim1 in S:
        for dim2 in H:
            info = f"Benchmarking RMSNorm with shape ({dim1}, {dim2}), data type fp32"
            A = create_tensor(shape=[dim1, dim2], dtype=config.dtypes["fp32"], ndim=2, device="cuda")
            B = create_tensor(shape=[dim2], dtype=config.dtypes["fp32"], ndim=1, device="cuda")
            C = create_tensor(shape=[dim1, dim2], dtype=config.dtypes["fp32"], ndim=2, device="cuda")

            # Benchmark PyTorch rms_norm
            rms_norm_torch(A, B, C, shape=[dim1, dim2], config=config)

            # Benchmark ONNX rms_norm
            # rms_norm_onnx(A, "input", B, "output", shape=[dim1, dim2], config=config)

            # Benchmark custom rms_norm kernel
            if dim2 >= 32 and dim2 <= 1024:
                run_benchmark(lib.rms_norm_f32, A, B, C, "rms_norm_f32", config=config)

            if dim2 >= 128 and dim2 <= 4096:
                run_benchmark(lib.rms_norm_f32x4, A, B, C, "rms_norm_f32x4", config=config)
            print_perf(
                input_shape=[dim1, dim2],
                output_shape=[dim1, dim2],
                config=config,
                info=info,
                kernels=[
                    "torch_rms_norm",
                    "rms_norm_f32",
                    "rms_norm_f32x4",
                ],
            )
            clear_status(config)

    for dim1 in S:
        for dim2 in H:
            info = f"Benchmarking RMSNorm with shape ({dim1}, {dim2}), data type fp16"
            A = create_tensor(shape=[dim1, dim2], dtype=config.dtypes["fp16"], ndim=2, device="cuda")
            B = create_tensor(shape=[dim2], dtype=config.dtypes["fp16"], ndim=1, device="cuda")
            C = create_tensor(shape=[dim1, dim2], dtype=config.dtypes["fp16"], ndim=2, device="cuda")

            # Benchmark PyTorch rms_norm
            rms_norm_torch(A, B, C, shape=[dim1, dim2], config=config)

            # Benchmark ONNX rms_norm
            # rms_norm_onnx(A, "input", B, "output", shape=[dim1, dim2], config=config)

            # Benchmark custom rms_norm kernel
            if dim2 >= 32 and dim2 <= 1024:
                run_benchmark(lib.rms_norm_f16, A, B, C, "rms_norm_f16", config=config)
            if dim2 >= 64 and dim2 <= 2048:
                run_benchmark(lib.rms_norm_f16x2, A, B, C, "rms_norm_f16x2", config=config)
            if dim2 >= 256 and dim2 <= 8192:
                run_benchmark(lib.rms_norm_f16x8, A, B, C, "rms_norm_f16x8", config=config)
                run_benchmark(lib.rms_norm_f16_pack, A, B, C, "rms_norm_f16_pack", config=config)
            print_perf(
                input_shape=[dim1, dim2],
                output_shape=[dim1, dim2],
                config=config,
                info=info,
                kernels=[
                    "torch_rms_norm",
                    "rms_norm_f16",
                    "rms_norm_f16x2",
                    "rms_norm_f16x8",
                    "rms_norm_f16_pack",
                ],
            )
            clear_status(config)
