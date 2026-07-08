// ============================================================================
// 矩阵核 GEMM v2: __builtin_amdgcn_mmac_f32_16x16x16f16 (gfx936 v_mmac).
//   out[M,N] = x[M,K] @ W[N,K]^T (+bias), fp16 I/O, fp32 累加. 行主序.
//   C[m,n] = sum_k A[m,k]*W[n,k].  (内积维 K, A=x, B=W).
//
// ★fragment 布局(实测逆向, A=val/B=I 和 A=I/B=val 双向锚定, 256/256 bit-exact):
//   lane L: r=L%16, g=L/16 (g=0..3).
//   a_frag[i] = A[m=r][k=g*4+i]            (A 行主序 A[m][k])
//   b_frag[i] = Bm[k=r][n=g*4+i]           (Bm 行主序 Bm[k][n]); 我们 Bm[k][n]=W[n][k]
//              → b_frag[i] = W[n=g*4+i][k=r]
//   C 输出: raw acc[i] (lane L) = C[m=r][n=g*4+i]
//
// 一个 wave64 算一个 16(M)×16(N) 输出块, K 每步 16, 累加 K/16 次.
// 第一版(正确性优先): 1 wave/block, 直接 global 读 fragment.
// ============================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

typedef __fp16 half4 __attribute__((ext_vector_type(4)));
typedef float  f32x4 __attribute__((ext_vector_type(4)));

__global__ void gemm_mmac_basic(const __half* __restrict__ A, const __half* __restrict__ W,
                                const __half* __restrict__ bias, __half* __restrict__ C,
                                int M, int N, int K) {
    int lane = threadIdx.x & 63;
    int r = lane & 15;        // lane%16
    int g = lane >> 4;        // lane/16 (0..3)
    int blockM = blockIdx.y * 16;
    int blockN = blockIdx.x * 16;

    f32x4 c = {0,0,0,0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half4 a, b;
        #pragma unroll
        for (int i=0;i<4;i++){
            // ★实测 bit-exact 布局(al=0,bl=1,co=0):
            // a_frag[i] = A[m=r][k=g*4+i]
            int gm = blockM + r, gk = k0 + g*4 + i;
            a[i] = (gm<M && gk<K) ? (__fp16)__half2float(A[(size_t)gm*K + gk]) : (__fp16)0;
            // b_frag[i] = Bm[k=g*4+i][n=r] = W[n=r][k=g*4+i]
            int gn = blockN + r, wk = k0 + g*4 + i;
            b[i] = (gn<N && wk<K) ? (__fp16)__half2float(W[(size_t)gn*K + wk]) : (__fp16)0;
        }
        c = __builtin_amdgcn_mmac_f32_16x16x16f16(a, b, c);
    }
    // 写回(实测 C 布局): acc[i] = C[m=r][n=g + i*4]
    #pragma unroll
    for (int i=0;i<4;i++){
        int gm = blockM + r;
        int gn = blockN + g + i*4;
        if (gm<M && gn<N){
            float v = c[i];
            if (bias) v += __half2float(bias[gn]);
            C[(size_t)gm*N + gn] = __float2half(v);
        }
    }
}

extern "C" void gemm_mmac_run(const void* x, const void* W, const void* bias,
                              void* out, int M, int N, int K, int variant, void* stream){
    const __half* A=(const __half*)x; const __half* B=(const __half*)W;
    const __half* bs=(const __half*)bias; __half* C=(__half*)out;
    hipStream_t s=(hipStream_t)stream;
    dim3 blk(64);
    dim3 grd((N+15)/16,(M+15)/16);
    gemm_mmac_basic<<<grd,blk,0,s>>>(A,B,bs,C,M,N,K);
}
