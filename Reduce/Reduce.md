### 线程组织结构
设输入向量为 $\vec{A} = [a_0, a_1, ..., a_{N-1}]$，其中 $N$ 为总元素数。 输出向量 $\vec{B} = [b_0, b_1, ..., b_{\frac{N}{BN}}]$, 其中$BN$ 为块大小

对于第 $i$ 个线程块（$i = 0, 1, ..., \frac{N}{\text{BN}}$）：
- 处理输入向量：$\vec{A}_i = [a_{\text{BN} \cdot i}, a_{\text{BN} \cdot i + 1}, ..., a_{\text{BN} \cdot (i+1) - 1}]$
- 处理输出向量: $\vec{B}_i = b_i$


### Reduce_v0_baseline.cu

**线程索引定义**
- 块索引：$x = \text{blockIdx.x}$
- 线程索引：$i = \text{threadIdx.x}$
- 块大小：$BN = \text{blockDim.x}$

**GMem -> SMem**（内存复制阶段）
每个线程 $i$ 的搬运任务：
$$\text{SMem}[i] = \text{vec\_A}[x \cdot BN + i]$$

其中线程 $i$ 搬运全局内存中第 $(x \cdot BN + i)$ 个元素到共享内存的第 $i$ 个位置。

**计算任务**（Reduction阶段）

在第 $k$ 轮迭代中（$k = 0, 1, 2, \ldots, \log_2(N)-1$）：

**步长定义：** $s_k = 2^k$（从1开始，每轮翻倍）

**活跃线程条件：** $i \bmod (2 \times s_k) = 0$

**计算操作：**  
当条件满足时，线程 $i$ 执行：  
$$SMem[i]←SMem[i]+SMem[i+sk]$$

**具体展开：**

- **第0轮：** $s_0 = 1$，线程条件 $i \bmod 2 = 0$（即偶数线程）
    
    - 线程0: $\text{SMem}[0] \leftarrow \text{SMem}[0] + \text{SMem}[1]$
    - 线程2: $\text{SMem}[2] \leftarrow \text{SMem}[2] + \text{SMem}[3]$
    - 线程4: $\text{SMem}[4] \leftarrow \text{SMem}[4] + \text{SMem}[5]$
    - ...
- **第1轮：** $s_1 = 2$，线程条件 $i \bmod 4 = 0$
    
    - 线程0: $\text{SMem}[0] \leftarrow \text{SMem}[0] + \text{SMem}[2]$
    - 线程4: $\text{SMem}[4] \leftarrow \text{SMem}[4] + \text{SMem}[6]$
    - ...
- **第2轮：** $s_2 = 4$，线程条件 $i \bmod 8 = 0$
    
    - 线程0: $\text{SMem}[0] \leftarrow \text{SMem}[0] + \text{SMem}[4]$
    - 线程8: $\text{SMem}[8] \leftarrow \text{SMem}[8] + \text{SMem}[12]$
    - ...
- **第k轮：** $s_k = 2^k$，线程条件 $i \bmod (2^{k+1}) = 0$
    
    - 活跃线程执行：$\text{SMem}[i] \leftarrow \text{SMem}[i] + \text{SMem}[i + 2^k]$






#### 造成 warp divergence 的原因

**活跃线程模式：**

对于步长 $s = 2^k$，活跃条件：$\text{tid} \bmod 2^{k+1} = 0$

在一个warp中（线程ID: 0-31），活跃线程的分布：

| 迭代轮次 | 步长s | 活跃线程ID | 活跃比例 |
|---------|-------|------------|----------|
| k=0 | s=1 | 0,2,4,6,...,30 | 16/32 = 50% |
| k=1 | s=2 | 0,4,8,12,...,28 | 8/32 = 25% |
| k=2 | s=4 | 0,8,16,24 | 4/32 = 12.5% |
| k=3 | s=8 | 0,16 | 2/32 = 6.25% |


### Reduce_v1_no_divergence_branch.cu

**步长定义**
在第 $k$ 轮迭代中（$k = 0, 1, 2, \ldots, \log_2(N)-1$）：$s_k = 2^k$

**活跃线程条件**
$$\text{index} = 2 \times s_k \times \text{threadIdx.x} < N$$
即：$$\text{threadIdx.x} < \frac{N}{2 \times s_k} = \frac{N}{2^{k+1}}$$

**计算操作**
当条件满足时，线程执行：
$$\text{SMem}[\text{index}] \leftarrow \text{SMem}[\text{index}] + \text{SMem}[\text{index} + s_k]$$

**具体展开（N=256的例子）**

**第0轮** ($k = 0, s_0 = 1$)
- **活跃线程**：$\text{threadIdx.x} < 128$ （线程0-127）
- **计算操作**：
  - 线程0：$\text{SMem}[0] += \text{SMem}[1]$
  - 线程1：$\text{SMem}[2] += \text{SMem}[3]$
  - 线程2：$\text{SMem}[4] += \text{SMem}[5]$
  - ...
  - 线程127：$\text{SMem}[254] += \text{SMem}[255]$

**第1轮** ($k=1, s_1=2$)
- **活跃线程**：$\text{threadIdx.x} < 64$ （线程0-63）
- **计算操作**：
  - 线程0：$\text{SMem}[0] += \text{SMem}[2]$
  - 线程1：$\text{SMem}[4] += \text{SMem}[6]$
  - 线程2：$\text{SMem}[8] += \text{SMem}[10]$
  - ...

**第2轮** ($k=2, s_2=4$)
- **活跃线程**：$\text{threadIdx.x} < 32$ （线程0-31）
- **计算操作**：
  - 线程0：$\text{SMem}[0] += \text{SMem}[4]$
  - 线程1：$\text{SMem}[8] += \text{SMem}[12]$
  - ...

**Warp级别的分析（8个warp，每个warp 32线程）**

| 轮次  | 活跃warp   | 活跃线程范围 | Divergence? |
| --- | -------- | ------ | ----------- |
| 0   | warp 0-3 | 0-127  | ❌ 无         |
| 1   | warp 0-1 | 0-63   | ❌ 无         |
| 2   | warp 0   | 0-31   | ❌ 无         |
| 3   | warp 0   | 0-15   | ⚠️ 有        |
| 4   | warp 0   | 0-7    | ⚠️ 有        |
| 5-7 | warp 0   | 少量线程   | ⚠️ 有        |




| 迭代  | s值    | 活跃线程范围            | 操作模式                                       | 活跃线程数 |
| --- | ----- | ----------------- | ------------------------------------------ | ----- |
| 1   | s=1   | tid=0,1,2,...,127 | SMem[0]+=SMem[1], SMem[2]+=SMem[3], ...    | 128   |
| 2   | s=2   | tid=0,1,2,...,63  | SMem[0]+=SMem[2], SMem[4]+=SMem[6], ...    | 64    |
| 3   | s=4   | tid=0,1,2,...,31  | SMem[0]+=SMem[4], SMem[8]+=SMem[12], ...   | 32    |
| 4   | s=8   | tid=0,1,2,...,15  | SMem[0]+=SMem[8], SMem[16]+=SMem[24], ...  | 16    |
| 5   | s=16  | tid=0,1,2,...,7   | SMem[0]+=SMem[16], SMem[32]+=SMem[48], ... | 8     |
| 6   | s=32  | tid=0,1,2,3       | SMem[0]+=SMem[32], SMem[64]+=SMem[96], ... | 4     |
| 7   | s=64  | tid=0,1           | SMem[0]+=SMem[64], SMem[128]+=SMem[192]    | 2     |
| 8   | s=128 | tid=0             | SMem[0]+=SMem[128]                         | 1     |

#### 造成 bank conflict 的原因

- BN = 256
- Shared Memory有32个Bank (Bank 0-31)
- 每个Bank宽度32位(存储一个int/float)
- Bank ID = (地址 / 4) % 32

**迭代1 (s=1) 2-way Bank Conflict**

| Warp 0中的线程 | threadIdx.x | index = 2\*s\*tid | 访问地址               | Bank ID | 冲突组        |
| ---------- | ----------- | ----------------- | ------------------ | ------- | ---------- |
| T0         | 0           | 0                 | SMem[0], SMem[1]   | 0, 1    | 组1         |
| T1         | 1           | 2                 | SMem[2], SMem[3]   | 2, 3    | 组2         |
| T2         | 2           | 4                 | SMem[4], SMem[5]   | 4, 5    | 组3         |
| ...        | ...         | ...               | ...                | ...     | ...        |
| T15        | 15          | 30                | SMem[30], SMem[31] | 30, 31  | 组16        |
| T16        | 16          | 32                | SMem[32], SMem[33] | 0, 1    | **组1冲突!**  |
| T17        | 17          | 34                | SMem[34], SMem[35] | 2, 3    | **组2冲突!**  |
| ...        | ...         | ...               | ...                | ...     | ...        |
| T31        | 31          | 62                | SMem[62], SMem[63] | 30, 31  | **组16冲突!** |

**结果：每个Bank被2个线程同时访问 → 2-way Bank Conflict**

**迭代2 (s=2) - 4-way Bank Conflict**

| Warp 0中的线程 | threadIdx.x | index = 4\*tid | 访问地址               | Bank ID  | 冲突分析       |
| ---------- | ----------- | ------------- | ------------------ | -------- | ---------- |
| T0         | 0           | 0             | SMem[0], SMem[2]   | 0, 2     | -          |
| T1         | 1           | 4             | SMem[4], SMem[6]   | 4, 6     | -          |
| T2         | 2           | 8             | SMem[8], SMem[10]  | 8, 10    | -          |
| T3         | 3           | 12            | SMem[12], SMem[14] | 12, 14   | -          |
| T4         | 4           | 16            | SMem[16], SMem[18] | 16, 18   | -          |
| T5         | 5           | 20            | SMem[20], SMem[22] | 20, 22   | -          |
| T6         | 6           | 24            | SMem[24], SMem[26] | 24, 26   | -          |
| T7         | 7           | 28            | SMem[28], SMem[30] | 28, 30   | -          |
| T8         | 8           | 32            | SMem[32], SMem[34] | **0, 2** | **与T0冲突!** |
| T9         | 9           | 36            | SMem[36], SMem[38] | **4, 6** | **与T1冲突!** |
| ...        | ...         | ...           | ...                | ...      | ...        |

**结果：每个Bank被4个线程访问 → 4-way Bank Conflict**

**迭代3 (s=4) - 8-way Bank Conflict**

| 线程组 | 访问的Bank ID | 冲突程度 |
|--------|---------------|----------|
| T0,T8,T16,T24 | Bank 0 | 4-way conflict |
| T1,T9,T17,T25 | Bank 4 | 4-way conflict |
| T2,T10,T18,T26 | Bank 8 | 4-way conflict |
| T3,T11,T19,T27 | Bank 12 | 4-way conflict |
| T4,T12,T20,T28 | Bank 16 | 4-way conflict |
| T5,T13,T21,T29 | Bank 20 | 4-way conflict |
| T6,T14,T22,T30 | Bank 24 | 4-way conflict |
| T7,T15,T23,T31 | Bank 28 | 4-way conflict |

**迭代4及以后的Bank Conflict**

| 迭代  | 步长s | 活跃线程数 | Bank Conflict程度 | 原因      |
| --- | --- | ----- | --------------- | ------- |
| 4   | 8   | 16    | 无冲突             | 访问间隔足够大 |
| 5   | 16  | 8     | 无冲突             | 访问间隔更大  |
| 6   | 32  | 4     | 无冲突             | 访问间隔很大  |
| 7+  | 64+ | ≤2    | 无冲突             | 线程很少    |

### Reduce_v2_no_bank_confict

**计算任务**（Reduction阶段）
在第 $k$ 轮迭代中（$k = 0, 1, 2, \ldots, \log_2(N)-1$）：

**步长定义：** $s_k = \frac{N}{2^{k+1}}$

**活跃线程条件** $i < s_k$

**计算操作：**
当条件满足时，线程 $i$ 执行：
$$\text{SMem}[i] \leftarrow \text{SMem}[i] + \text{SMem}[i + s_k]$$

**具体展开：**
- 第0轮：$s_0 = N/2$，线程 $i < N/2$ 计算 $\text{SMem}[i] \leftarrow \text{SMem}[i] + \text{SMem}[i + N/2]$
- 第1轮：$s_1 = N/4$，线程 $i < N/4$ 计算 $\text{SMem}[i] \leftarrow \text{SMem}[i] + \text{SMem}[i + N/4]$
- 第k轮：$s_k = N/2^{k+1}$，线程 $i < N/2^{k+1}$ 计算 $\text{SMem}[i] \leftarrow \text{SMem}[i] + \text{SMem}[i + N/2^{k+1}]$

**分析 bank conflict**

| 迭代轮次    | 步长(s) | 活跃线程数 | 活跃Warp数 | 访问模式示例                                             | Bank冲突情况                                        |
| ------- | ----- | ----- | ------- | -------------------------------------------------- | ----------------------------------------------- |
| **第1轮** | 128   | 128   | 4       | tid=0访问[0,128]<br>tid=1访问[1,129]<br>tid=2访问[2,130] | **无冲突**<br>stride=128, 128%32=0<br>相邻线程访问不同bank |
| **第2轮** | 64    | 64    | 2       | tid=0访问[0,64]<br>tid=1访问[1,65]<br>tid=2访问[2,66]    | **无冲突**<br>stride=64, 64%32=0<br>相邻线程访问不同bank   |
| **第3轮** | 32    | 32    | 1       | tid=0访问[0,32]<br>tid=1访问[1,33]<br>tid=2访问[2,34]    | **无冲突**<br>stride=32, 32%32=0<br>相邻线程访问不同bank   |

### Reduce_v3_add_during_load

**功能**: 归约求和核函数，在加载时进行第一步归约操作
**符号定义**:
- 块索引: `blockIdx.x` 
- 线程索引: `threadIdx.x` (tid)
- SMem: 共享内存 `SMem[THREAD_PER_BLOCK]`
- GMem: 全局内存 `vec_A`, `vec_B`
- 一维输入: `vec_A` (待归约数组), `vec_B` (结果数组)

**GMem -> SMem 每个线程的搬运任务**

**数学描述**:
- 每个线程 $tid$ 负责加载并归约两个相邻的数据块
- $A_{start} = vec\_A + blockIdx.x \times blockDim.x \times 2$
- 线程 $tid$ 处理全局内存位置: $A_{start}[tid]$ 和 $A_{start}[tid + blockDim.x]$

**具体操作**:
$$SMem[tid] = A_{start}[tid] + A_{start}[tid + blockDim.x]$$

| 线程ID | 加载位置1 | 加载位置2 | 存储到SMem |
|--------|-----------|-----------|------------|
| 0 | A_start[0] | A_start[256] | SMem[0] |
| 1 | A_start[1] | A_start[257] | SMem[1] |
| 255 | A_start[255] | A_start[511] | SMem[255] |

**计算任务**

**活跃线程条件**: $threadIdx.x < s$，其中 $s \in \{128, 64, 32, 16, 8, 4, 2, 1\}$

每个活跃线程 $tid$ 执行: 
$$SMem[tid] \leftarrow SMem[tid] + SMem[tid + s]$$

其中 $s = \frac{blockDim.x}{2^{iteration}}$，$iteration \in \{1, 2, 3, ..., \log_2(blockDim.x)\}$

**迭代过程表格 (8 Warp, 256线程)**

| 迭代轮次 | $s$值 | 活跃线程范围 | 活跃Warp数 | 操作描述 |
|----------|-------|--------------|------------|----------|
| 1 | 128 | $tid \in [0, 127]$ | 4个 | $SMem[tid] += SMem[tid + 128]$ |
| 2 | 64 | $tid \in [0, 63]$ | 2个 | $SMem[tid] += SMem[tid + 64]$ |
| 3 | 32 | $tid \in [0, 31]$ | 1个 | $SMem[tid] += SMem[tid + 32]$ |
| 4 | 16 | $tid \in [0, 15]$ | 1个 | $SMem[tid] += SMem[tid + 16]$ |
| 5 | 8 | $tid \in [0, 7]$ | 1个 | $SMem[tid] += SMem[tid + 8]$ |
| 6 | 4 | $tid \in [0, 3]$ | 1个 | $SMem[tid] += SMem[tid + 4]$ |
| 7 | 2 | $tid \in [0, 1]$ | 1个 | $SMem[tid] += SMem[tid + 2]$ |
| 8 | 1 | $tid = 0$ | 1个 | $SMem[0] += SMem[1]$ |

**第1轮** ($s=128$):
- 活跃线程: $tid \in [0, 127]$
- 操作: $\forall tid \in [0, 127]: SMem[tid] \leftarrow SMem[tid] + SMem[tid + 128]$

**第3轮** ($s=32$):
- 活跃线程: $tid \in [0, 31]$ (仅Warp 0活跃)
- 操作: $\forall tid \in [0, 31]: SMem[tid] \leftarrow SMem[tid] + SMem[tid + 32]$

**第8轮** ($s=1$):
- 活跃线程: $tid = 0$ (仅1个线程活跃)
- 操作: $SMem[0] \leftarrow SMem[0] + SMem[1]$


**条件**: $threadIdx.x = 0$
**操作**: 
$$vec\_B[blockIdx.x] = SMem[0]$$

**数学表示**:
最终结果为当前块处理的 $2 \times blockDim.x$ 个元素的和:
$$vec\_B[blockIdx.x] = \sum_{i=0}^{2 \times blockDim.x - 1} vec\_A[blockIdx.x \times 2 \times blockDim.x + i]$$



### Reduce_v4_unroll_last_warp

**计算过程与 Reduce_v3_add_during_load 相同**
#### warp Reduce volatile 保证内存访问正确性
**Warp 内的隐式同步 为什么需要 volatile？**
```cuda
// volatile 告诉编译器不要对被修饰的变量进行优化，每次访问都必须从内存中重新读取
__device__ void warpReduce4(volatile float *cache) {
    cache[threadIdx.x] += cache[threadIdx.x + 32];  // 步骤1: 32+0, 33+1, ..., 63+31
    cache[threadIdx.x] += cache[threadIdx.x + 16];  // if(tid < 16) 步骤2: 只有前16个线程有效
    cache[threadIdx.x] += cache[threadIdx.x + 8];   // if(tid < 8)  步骤3: 只有前8个线程有效
    cache[threadIdx.x] += cache[threadIdx.x + 4];
    cache[threadIdx.x] += cache[threadIdx.x + 2];
    cache[threadIdx.x] += cache[threadIdx.x + 1];
}
```

**没有 volatile 编译器会进行以下优化：**
```cuda
// 编译器可能优化成这样（错误！）
float temp = cache[threadIdx.x];
temp += cache[threadIdx.x + 32];
temp += cache[threadIdx.x + 16];
temp += cache[threadIdx.x + 8];
temp += cache[threadIdx.x + 4];
temp += cache[threadIdx.x + 2];
temp += cache[threadIdx.x + 1];
cache[threadIdx.x] = temp;
```
**这样优化后，计算后的结果是写入到寄存器中而不是缓存中, 丢失了线程之间的数据依赖**

**warp内同步**：warp内所有线程在同一条指令上执行，天然同步
**内存可见性**：`volatile` 确保每次访问都从内存读取最新值
**无需显式同步**：不需要 `__syncthreads()` `__syncwarp()`，因为warp内天然同步

**`volatile` 确保了在warp天然同步的基础上，内存访问的正确性！**

### Reduce_v5_completely_unroll

### Reduce_v7_shuffle

**GMem -> Reg**
每个线程 `i` 的加载任务：
- 从全局内存 `vec_A` 中加载 `NUM_PER_THREAD` 个元素
- 加载地址：`A_start + iter × blockSize + i`，其中 `iter ∈ [0, NUM_PER_THREAD)`
- 线程 `i` 负责加载索引为 `{i, i+blockSize, i+2×blockSize, ..., i+(NUM_PER_THREAD-1)×blockSize}` 的元素

**Reg -> SMem**
- `warpLevelSums[32]`: 存储每个 warp 的归约结果
- 每个 warp 的第一个线程（`laneId == 0`）将其 warp 的归约结果写入共享内存


**第一阶段：线程级归约**
活跃线程条件：所有线程 `i ∈ [0, blockSize)`

每个线程计算：
$$sum_i = \sum_{k=0}^{NUM\_PER\_THREAD-1} A[x \times blockSize \times NUM\_PER\_THREAD + k \times blockSize + i]$$

**第二阶段：Warp内归约（Shuffle）**

**Warp Shuffle 归约过程**

| 迭代步骤 | 条件检查 | Shuffle距离 | 操作描述 | 活跃线程范围 |
|---------|----------|-------------|----------|-------------|
| 1 | `blockSize >= 32` | 16 | `sum[i] += sum[i+16]` | `laneId ∈ [0,15]` |
| 2 | `blockSize >= 16` | 8  | `sum[i] += sum[i+8]`  | `laneId ∈ [0,7]` |
| 3 | `blockSize >= 8`  | 4  | `sum[i] += sum[i+4]`  | `laneId ∈ [0,3]` |
| 4 | `blockSize >= 4`  | 2  | `sum[i] += sum[i+2]`  | `laneId ∈ [0,1]` |
| 5 | `blockSize >= 2`  | 1  | `sum[i] += sum[i+1]`  | `laneId = 0` |

其中 `laneId = i % 32`，`warpId = i / 32`

**第三阶段：Warp间归约**

活跃线程条件：
- 存储warp结果：`laneId == 0` 的线程将warp归约结果存入 `warpLevelSums[warpId]`
- 最终归约：`warpId == 0` 的线程对所有warp结果进行最终归约


第一个warp中的线程重新加载数据：
$$sum_i = \begin{cases} 
warpLevelSums[i] & \text{if } i < \frac{blockSize}{32} \\
0 & \text{otherwise}
\end{cases}$$

然后执行 `warp_shfl_Reduce<blockSize/32>(sum)` 