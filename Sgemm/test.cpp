template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_kernel(float *a, float *b, float *c, const int M, const int N, const int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];

    float r_load_a[TM / 2];  // 4
    float r_load_b[TN / 2];  // 4
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid / 2;  // tid / 2，(0,1,2,...,128)

    int load_a_smem_k = (tid & 1) << 2;  // (0,4)

    int load_b_smem_k = tid / 32;  // 0~8

    int load_b_smem_n = (tid & 31) << 2;  // (0,4,8,12,...,124)

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    if (load_a_gmem_m >= M || load_b_gmem_n >= N) return;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

 layer_0  b0
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];  // e.g layer_4  b0
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];  // e.g layer_8  b0
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];  // e.g layer_12 b0

        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {

            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);

            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);
            // conclusion: still have some bank conflicts, need 4 memory issues.

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}