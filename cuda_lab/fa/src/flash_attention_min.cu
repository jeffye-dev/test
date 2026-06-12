#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// 简化分块注意力：每个 block 处理一个 query 行，演示在线 softmax + 分块
// 形状：Q,K,V 均为 [N, D]，输出 O 为 [N, D]。不使用 Tensor Core。
#define D 64            // head_dim，取小值便于阅读
#define BK 64           // K/V 分块大小

__global__ void flash_attn_min(const float* Q, const float* K,
                               const float* V, float* O, int N) {
    int row = blockIdx.x;              // 一个 block 负责一个 query 行（决策五：块间独立）
    int t = threadIdx.x;              // 线程负责输出维度 d = t（假设 blockDim.x == D）
    if (row >= N) return;

    __shared__ float Ks[BK * D];      // 决策三：K/V 块进共享内存
    __shared__ float Vs[BK * D];
    __shared__ float q[D];            // 当前 query 行
    __shared__ float scores[BK];      // 当前块的分数
    __shared__ float reduce[BK];      // 规约暂存

    q[t] = Q[row * D + t];
    __syncthreads();

    float m = -INFINITY;              // 运行最大值（在线 softmax）
    float l = 0.0f;                   // 运行分母
    float acc = 0.0f;                 // 该线程负责维度 d=t 的输出累加器

    for (int kb = 0; kb < N; kb += BK) {
        int bn = min(BK, N - kb);
        // 合并加载 K/V 块到共享内存（决策三 + 第 06 篇合并访问）
        for (int j = 0; j < bn; ++j) {
            Ks[j * D + t] = K[(kb + j) * D + t];
            Vs[j * D + t] = V[(kb + j) * D + t];
        }
        __syncthreads();              // 第 07 篇：写完共享内存再读

        // 算分数 scores[j] = dot(q, K_j)。用线程 0..bn-1 各算一行点积
        if (t < bn) {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s += q[d] * Ks[t * D + d];
            scores[t] = s / sqrtf((float)D);
        }
        __syncthreads();

        // 块内最大值（简单串行规约，教学版；生产版用 warp shuffle，第 03 篇）
        float m_blk = -INFINITY;
        for (int j = 0; j < bn; ++j) m_blk = fmaxf(m_blk, scores[j]);

        float m_new = fmaxf(m, m_blk);
        float corr = expf(m - m_new);     // 决策二：最大值变化的校正因子
        float l_new = l * corr;
        acc *= corr;                      // 校正已累加输出

        for (int j = 0; j < bn; ++j) {
            float p = expf(scores[j] - m_new);
            l_new += p;
            acc += p * Vs[j * D + t];      // 累加到该线程负责的维度
        }
        m = m_new;
        l = l_new;
        __syncthreads();
    }

    O[row * D + t] = acc / l;             // 归一化写回
}

// 朴素参考实现（CPU），用于对拍
static void attn_ref(const float* Q, const float* K, const float* V,
                     float* O, int N) {
    float* s = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        float mx = -INFINITY;
        for (int j = 0; j < N; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += Q[i * D + d] * K[j * D + d];
            s[j] = dot / sqrtf((float)D);
            mx = fmaxf(mx, s[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) { s[j] = expf(s[j] - mx); sum += s[j]; }
        for (int d = 0; d < D; ++d) {
            float o = 0.0f;
            for (int j = 0; j < N; ++j) o += s[j] * V[j * D + d];
            O[i * D + d] = o / sum;
        }
    }
    free(s);
}

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA err %s\n", cudaGetErrorString(e)); exit(1);} } while(0)

int main() {
    const int N = 256;
    size_t bytes = (size_t)N * D * sizeof(float);
    float *hQ=(float*)malloc(bytes), *hK=(float*)malloc(bytes),
          *hV=(float*)malloc(bytes), *hO=(float*)malloc(bytes),
          *ref=(float*)malloc(bytes);
    srand(0);
    for (int i = 0; i < N * D; ++i) {
        hQ[i] = (rand() % 1000) / 1000.0f;
        hK[i] = (rand() % 1000) / 1000.0f;
        hV[i] = (rand() % 1000) / 1000.0f;
    }

    float *dQ,*dK,*dV,*dO;
    CK(cudaMalloc(&dQ,bytes)); CK(cudaMalloc(&dK,bytes));
    CK(cudaMalloc(&dV,bytes)); CK(cudaMalloc(&dO,bytes));
    CK(cudaMemcpy(dQ,hQ,bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,hK,bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,hV,bytes,cudaMemcpyHostToDevice));

    flash_attn_min<<<N, D>>>(dQ,dK,dV,dO,N);  // 每行一个 block，D 个线程
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hO,dO,bytes,cudaMemcpyDeviceToHost));

    attn_ref(hQ,hK,hV,ref,N);
    float maxerr = 0.0f;
    for (int i = 0; i < N * D; ++i)
        maxerr = fmaxf(maxerr, fabsf(hO[i]-ref[i]));
    printf("max abs err vs reference = %.3e  (should be < 1e-3)\n", maxerr);

    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    free(hQ);free(hK);free(hV);free(hO);free(ref);
    return 0;
}