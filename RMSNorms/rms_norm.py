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
        self.dim1 = [4096]  # [4096,8192]
        self.dim2 = [256]  # [256,512,1024,2048,4096,8192]
        self.kernels = {
            "rms_norm_f32": 0.0,
            "rms_norm_f32x4": 0.0,
            # "rms_norm_f16": 0.0,
            # "rms_norm_f16x2": 0.0,
            # "rms_norm_f16x8": 0.0,
            # "rms_norm_f16_pack": 0.0,
            "torch_rms_norm": 0.0,
        }  # kernel name : time in ms
        self.torch_out: torch.Tensor = None  # 用于存储torch的输出结果, 作为对比基准
        self.onnx_out: torch.Tensor = None  # 用于存储onnxruntime的输出结果, 作为对比基准
        self.tensors = [None] * len(self.kernels)
        self.dtypes = {"fp32": torch.float32, "fp16": torch.float16}
        self.compare = True
        self.iter = 1
        self.warmup = 0
        self.device = "cuda:0"  # "cpu"
        self.rtol = 1e-05
        self.atol = 1e-08
        # Load the CUDA kernel as a python module
        self.lib = load(
            name="rms_norm.lib",
            sources=[
                "./RMSNorms/bind.cpp",
                "./RMSNorms/rms_norm_f32.cu",
                "./RMSNorms/rms_norm_f32x4.cu",
                # "./RMSNorms/rms_norm_f16.cu",
                # "./RMSNorms/rms_norm_f16x2.cu",
                # "./RMSNorms/rms_norm_f16x8.cu",
                # "./RMSNorms/rms_norm_f16_pack.cu",
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
    shape: list = [4096, 256],
    config: Config = None,
):
    assert A is not None, "Input tensor A cannot be None"
    assert B is not None, "Output tensor B cannot be None"

    # warmup
    for _ in range(config.warmup):
        nn.rms_norm.softmax(A, dim=1, out=B)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    # iters
    for _ in range(config.iter):
        torch.softmax(A, dim=1, out=B)
    end_event.record()
    end_event.synchronize()
    # 计算平均时间
    total_time_ms = start_event.elapsed_time(end_event)
    mean_time = total_time_ms / config.iter

    # 展平, 从计算图中分离, 转cpu, 转numpy, 转list
    # 仅显示前三个元素
    out_val = B.flatten().detach().cpu().numpy().tolist()[:3]
    # 保留8位小数
    out_val = [round(v, 8) for v in out_val]
    config.kernels["torch_safe_softmax"] = mean_time
    config.tensors[list(config.kernels.keys()).index("torch_safe_softmax")] = out_val
    config.torch_out = B


"""
@func: 使用onnx.helper构建softmax算子, 使用onnxruntime库运行
@param: A: 输入tensor
@param: A_str: 输入tensor name
@param: B: 输出tensor
@param: B_str: 输出tensor name
@param: shape: 形状
@param: config: 配置
"""


def softmax_onnx(
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
    kernel: str = None,
    config: Config = None,
):
    # 初始化 输出
    if B is not None:
        B.fill_(0)
    # warmup
    for _ in range(config.warmup):
        perf_func(A, B)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    # iters
    for i in range(config.iter):
        perf_func(A, B)
    end_event.record()
    end_event.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)
    mean_time = total_time_ms / config.iter

    # 展平, 从计算图中分离, 转cpu, 转numpy, 转list
    # 仅显示前三个元素
    out_val = B.flatten().detach().cpu().numpy().tolist()[:3]
    # 保留8位小数
    out_val = [round(v, 8) for v in out_val]

    config.kernels[kernel] = mean_time
    config.tensors[list(config.kernels.keys()).index(kernel)] = out_val
    if config.compare:
        if compare_tensor(B, config.torch_out, config=config):
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
    if ndim == 2:
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
    if A.shape != B.shape:
        print(f"Shape mismatch: {A.shape} vs {B.shape}")
        return False
    if not torch.allclose(A, B, rtol=config.rtol, atol=config.atol):
        print("Tensor values are not close enough.")
        return False
    return True


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
            info = f"Benchmarking Safe-Softmax with shape ({dim1}, {dim2}), data type fp32"
            A = create_tensor(shape=[dim1, dim2], dtype=config.dtypes["fp32"], device="cuda")
            B = create_tensor(shape=[dim1, dim2], dtype=config.dtypes["fp32"], device="cuda")

            # Benchmark PyTorch safe-softmax
            rms_norm_torch(A, B, shape=[dim1, dim2], config=config)

            # Benchmark ONNX safe-softmax
            # softmax_onnx(A, "input", B, "output", shape=[dim1, dim2], config=config)

            # Benchmark custom safe-softmax kernel
            if dim2 <= 1024:
                run_benchmark(lib.rms_norm_f32, A, B, "rms_norm_f32", config=config)

            if dim2 >= 128 and dim2 <= 4096:
                run_benchmark(lib.rms_norm_f32x4, A, B, "rms_norm_f32x4", config=config)
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

    # for dim1 in S:
    #     for dim2 in H:
    #         info = f"Benchmarking Safe-Softmax with shape ({dim1}, {dim2}), data type fp16"
    #         A = create_tensor(shape=[dim1, dim2], dtype=config.dtypes["fp16"], device="cuda")
    #         B = create_tensor(shape=[dim1, dim2], dtype=config.dtypes["fp16"], device="cuda")

    #         # Benchmark PyTorch safe-softmax
    #         rms_norm_torch(A, B, shape=[dim1, dim2], config=config)

    #         # Benchmark ONNX safe-softmax
    #         # softmax_onnx(A, "input", B, "output", shape=[dim1, dim2], config=config)

    #         # Benchmark custom safe-softmax kernel
    #         if dim2 >= 32 and dim2 <= 1024:
    #             run_benchmark(lib.rms_norm_f16, A, B, "safe_softmax_f16", config=config)
    #         if dim2 >= 256 and dim2 <= 8192:
    #             run_benchmark(lib.rms_norm_f16x8, A, B, "safe_softmax_f16x8", config=config)
    #             run_benchmark(lib.rms_norm_f16_pack, A, B, "safe_softmax_f16_pack", config=config)
    #         print_perf(
    #             input_shape=[dim1, dim2],
    #             output_shape=[dim1, dim2],
    #             config=config,
    #             info=info,
    #             kernels=[
    #                 "torch_safe_softmax",
    #                 "safe_softmax_f16",
    #                 "safe_softmax_f16x8",
    #                 "safe_softmax_f16_pack",
    #                 "online_softmax_f16x8",
    #             ],
    #         )
