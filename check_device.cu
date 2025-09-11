#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

// 辅助函数：根据计算能力获取每个SM的CUDA核心数
inline int _ConvertSMVer2Cores(int major, int minor) {
    // 定义每个SM的CUDA核心数（基于计算能力）
    typedef struct {
        int SM;  // 0xMm (M = SM主版本号, m = SM次版本号)
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192},  // Kepler Generation (SM 3.0) GK10x class
                                       {0x32, 192},  // Kepler Generation (SM 3.2) GK10x class
                                       {0x35, 192},  // Kepler Generation (SM 3.5) GK11x class
                                       {0x37, 192},  // Kepler Generation (SM 3.7) GK21x class
                                       {0x50, 128},  // Maxwell Generation (SM 5.0) GM10x class
                                       {0x52, 128},  // Maxwell Generation (SM 5.2) GM20x class
                                       {0x53, 128},  // Maxwell Generation (SM 5.3) GM20x class
                                       {0x60, 64},   // Pascal Generation (SM 6.0) GP10x class
                                       {0x61, 128},  // Pascal Generation (SM 6.1) GP10x class
                                       {0x62, 128},  // Pascal Generation (SM 6.2) GP10x class
                                       {0x70, 64},   // Volta Generation (SM 7.0) GV10x class
                                       {0x72, 64},   // Volta Generation (SM 7.2) GV11b class
                                       {0x75, 64},   // Turing Generation (SM 7.5) TU10x class
                                       {0x80, 64},   // Ampere Generation (SM 8.0) GA10x class
                                       {0x86, 128},  // Ampere Generation (SM 8.6) GA10x class
                                       {0x87, 128},  // Ampere Generation (SM 8.7) GA10x class
                                       {0x89, 128},  // Ada Lovelace Generation (SM 8.9) AD10x class
                                       {0x90, 128},  // Hopper Generation (SM 9.0) GH100 class
                                       {-1, -1}};

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }

    // 如果没有找到匹配的架构，返回估算值
    std::cout << "  警告: 未知的计算能力 " << major << "." << minor << ", 使用默认核心数估算" << std::endl;
    return 64;  // 默认估算值
}

int main() {
    // 获取CUDA设备数量
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    // 检查CUDA是否可用
    if (error != cudaSuccess) {
        std::cout << "CUDA错误: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (deviceCount == 0) {
        std::cout << "没有找到支持CUDA的设备!" << std::endl;
        return -1;
    }

    std::cout << "=== CUDA设备信息查询 ===" << std::endl;
    std::cout << "检测到 " << deviceCount << " 个CUDA设备" << std::endl << std::endl;

    // 遍历所有设备
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "设备 [" << device << "]: " << deviceProp.name << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        // === 基本信息 ===
        std::cout << "【基本信息】" << std::endl;
        std::cout << "  设备名称: " << deviceProp.name << std::endl;
        std::cout << "  计算能力: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  PCI总线ID: " << deviceProp.pciBusID << std::endl;
        std::cout << "  PCI设备ID: " << deviceProp.pciDeviceID << std::endl;
        std::cout << "  PCI域ID: " << deviceProp.pciDomainID << std::endl;

        // === 内存信息 ===
        std::cout << "\n【内存信息】" << std::endl;
        std::cout << "  全局内存总量: " << std::fixed << std::setprecision(2) << deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB"
                  << std::endl;
        std::cout << "  常量内存总量: " << deviceProp.totalConstMem / 1024 << " KB" << std::endl;
        std::cout << "  共享内存/块: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  共享内存/SM: " << deviceProp.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
        std::cout << "  L2缓存大小: " << deviceProp.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  内存总线位宽: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        std::cout << "  内存时钟频率: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;

        // 计算内存带宽
        float memBandwidth = 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6;
        std::cout << "  理论内存带宽: " << std::fixed << std::setprecision(1) << memBandwidth << " GB/s" << std::endl;

        // === 计算单元信息 ===
        std::cout << "\n【计算单元信息】" << std::endl;
        std::cout << "  流多处理器(SM)数量: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  CUDA核心数量/SM: " << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << std::endl;
        std::cout << "  总CUDA核心数: " << deviceProp.multiProcessorCount * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << std::endl;
        std::cout << "  基础时钟频率: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;

        // === 线程和块信息 ===
        std::cout << "\n【线程和块信息】" << std::endl;
        std::cout << "  每块最大线程数: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  每SM最大线程数: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  最大网格尺寸: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")"
                  << std::endl;
        std::cout << "  最大块尺寸: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2]
                  << ")" << std::endl;

        // === 寄存器和Warp信息 ===
        std::cout << "\n【寄存器和Warp信息】" << std::endl;
        std::cout << "  每块32位寄存器数: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  每SM 32位寄存器数: " << deviceProp.regsPerMultiprocessor << std::endl;
        std::cout << "  Warp大小: " << deviceProp.warpSize << std::endl;

        // === 功能特性 ===
        std::cout << "\n【功能特性】" << std::endl;
        std::cout << "  统一寻址: " << (deviceProp.unifiedAddressing ? "支持" : "不支持") << std::endl;
        std::cout << "  并发内核执行: " << (deviceProp.concurrentKernels ? "支持" : "不支持") << std::endl;
        std::cout << "  ECC内存支持: " << (deviceProp.ECCEnabled ? "启用" : "禁用") << std::endl;
        std::cout << "  异步引擎数量: " << deviceProp.asyncEngineCount << std::endl;
        std::cout << "  内存映射主机内存: " << (deviceProp.canMapHostMemory ? "支持" : "不支持") << std::endl;
        std::cout << "  计算模式: ";
        switch (deviceProp.computeMode) {
            case cudaComputeModeDefault:
                std::cout << "默认模式(多线程)" << std::endl;
                break;
            case cudaComputeModeExclusive:
                std::cout << "独占模式(单线程)" << std::endl;
                break;
            case cudaComputeModeProhibited:
                std::cout << "禁止模式" << std::endl;
                break;
            case cudaComputeModeExclusiveProcess:
                std::cout << "独占进程模式" << std::endl;
                break;
            default:
                std::cout << "未知模式" << std::endl;
        }

        // === 纹理和表面内存 ===
        std::cout << "\n【纹理和表面内存】" << std::endl;
        std::cout << "  1D纹理最大宽度: " << deviceProp.maxTexture1D << std::endl;
        std::cout << "  2D纹理最大尺寸: " << deviceProp.maxTexture2D[0] << " x " << deviceProp.maxTexture2D[1] << std::endl;
        std::cout << "  3D纹理最大尺寸: " << deviceProp.maxTexture3D[0] << " x " << deviceProp.maxTexture3D[1] << " x " << deviceProp.maxTexture3D[2]
                  << std::endl;
        std::cout << "  表面内存对齐: " << deviceProp.surfaceAlignment << " bytes" << std::endl;
        std::cout << "  纹理内存对齐: " << deviceProp.textureAlignment << " bytes" << std::endl;

        // 获取当前内存使用情况
        size_t freeMem, totalMem;
        cudaSetDevice(device);
        cudaMemGetInfo(&freeMem, &totalMem);
        std::cout << "\n【当前内存使用】" << std::endl;
        std::cout << "  已用内存: " << std::fixed << std::setprecision(2) << (totalMem - freeMem) / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  可用内存: " << std::fixed << std::setprecision(2) << freeMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  使用率: " << std::fixed << std::setprecision(1) << ((totalMem - freeMem) * 100.0 / totalMem) << "%" << std::endl;

        std::cout << std::endl;
    }

    // 显示CUDA Runtime版本
    int runtimeVersion, driverVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);

    std::cout << "【CUDA版本信息】" << std::endl;
    std::cout << "CUDA Runtime版本: " << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;
    std::cout << "CUDA Driver版本: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << std::endl;

    return 0;
}
