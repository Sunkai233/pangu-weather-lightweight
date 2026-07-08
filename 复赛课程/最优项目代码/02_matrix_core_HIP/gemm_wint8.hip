// ============================================================================
// 仅权重 INT8 矩阵核 GEMM(省权重显存 ~97MiB, U 大杠杆):
//   W 存 int8 [N,K] + per-N-channel scale sW[N]. A 保 fp16. 输出 fp16.
//   kernel 内 W 载入 LDS 时 dequant int8→fp16(×sW), 再走 fp16 矩阵核 v_mmac.
//   精度: 仅权重量化(per-channel), matmul 全 fp16 → 近无损(pangu量化 +0.08%).
//   C[m,n] = sum_k A[m,k] * (W_i8[n,k] * sW[n]).
// 布局同 fp16 mmac(实测 bit-exact): a_frag[i]=A[r][g*4+i], b_frag[i]=Wdq[r][g*4+i], C[r][g+i*4].
// ============================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
typedef __fp16 half4 __attribute__((ext_vector_type(4)));
typedef float  f32x4 __attribute__((ext_vector_type(4)));
typedef struct __align__(16) { __half2 a,b,c,d; } half8;

template<int WAVES, int NB, int KB, int PAD>
__global__ __launch_bounds__(WAVES*64)
void wint8_gemm(const __half* __restrict__ A, const signed char* __restrict__ Wi8,
                const float* __restrict__ sW, const __half* __restrict__ bias,
                __half* __restrict__ C, int M, int N, int K) {
    const int BM=WAVES*16, BN=NB*16, BK=KB*16, LDK=BK+PAD;
    __shared__ __half As[BM*(BK+PAD)];
    __shared__ __half Ws[BN*(BK+PAD)];   // dequant 后的 fp16
    const int TPB=WAVES*64, tid=threadIdx.x;
    const int wave=tid>>6, lane=tid&63, r=lane&15, g=lane>>4;
    const int blockM=blockIdx.y*BM, blockN=blockIdx.x*BN;
    const int A_units=BM*(BK/8), W_units=BN*BK;   // W int8 逐元素 dequant

    f32x4 acc[NB];
    #pragma unroll
    for(int j=0;j<NB;j++) acc[j]=f32x4{0,0,0,0};

    for(int k0=0;k0<K;k0+=BK){
        // A: half8 向量化载入(同 fp16 版)
        #pragma unroll
        for(int u=tid;u<A_units;u+=TPB){
            int m=u/(BK/8), kk=(u%(BK/8))*8, gm=blockM+m, gk=k0+kk;
            half8 v;
            if(gm<M&&gk+7<K) v=*reinterpret_cast<const half8*>(&A[(size_t)gm*K+gk]);
            else{__half t[8];
                #pragma unroll
                for(int e=0;e<8;e++)t[e]=(gm<M&&gk+e<K)?A[(size_t)gm*K+gk+e]:(__half)0;
                v=*reinterpret_cast<half8*>(t);}
            *reinterpret_cast<half8*>(&As[m*LDK+kk])=v;
        }
        // W: int8 载入 + dequant→fp16 写 LDS. 每线程搬若干 (n,k).
        //   int8 用 char8(64bit) 向量化读, dequant ×sW[n] 后写 Ws.
        for(int u=tid;u<BN*(BK/8);u+=TPB){
            int n=u/(BK/8), kk=(u%(BK/8))*8, gn=blockN+n, gk=k0+kk;
            float scale = (gn<N) ? sW[gn] : 0.f;
            // 读 8 个 int8(连续 K)
            signed char t8[8];
            *reinterpret_cast<int2*>(t8) = (gn<N&&gk+7<K) ? *reinterpret_cast<const int2*>(&Wi8[(size_t)gn*K+gk]) : int2{0,0};
            __half hd[8];
            #pragma unroll
            for(int e=0;e<8;e++){
                signed char q = (gn<N&&gk+e<K) ? t8[e] : (signed char)0;
                hd[e] = __float2half((float)q * scale);
            }
            *reinterpret_cast<half8*>(&Ws[n*LDK+kk]) = *reinterpret_cast<half8*>(hd);
        }
        __syncthreads();
        #pragma unroll
        for(int ks=0;ks<KB;ks++){
            half4 a = *reinterpret_cast<const half4*>(&As[(wave*16+r)*LDK + ks*16 + g*4]);
            #pragma unroll
            for(int j=0;j<NB;j++){
                half4 b = *reinterpret_cast<const half4*>(&Ws[(j*16+r)*LDK + ks*16 + g*4]);
                acc[j]=__builtin_amdgcn_mmac_f32_16x16x16f16(a,b,acc[j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int j=0;j<NB;j++){
        #pragma unroll
        for(int i=0;i<4;i++){
            int gm=blockM+wave*16+r, gn=blockN+j*16+g+i*4;
            if(gm<M&&gn<N){ float v=acc[j][i]; if(bias)v+=__half2float(bias[gn]); C[(size_t)gm*N+gn]=__float2half(v);}
        }
    }
}

// 验证用: fp16 W → int8 + per-N scale 量化 kernel(离线 prepack, 计时区外).
__global__ void quant_w(const __half* W, signed char* Wi8, float* sW, int N, int K) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n>=N) return;
    float mx=1e-6f;
    for(int k=0;k<K;k++){ float v=fabsf(__half2float(W[(size_t)n*K+k])); if(v>mx)mx=v; }
    float s=mx/127.f; sW[n]=s; float inv=1.f/s;
    for(int k=0;k<K;k++){ int q=(int)rintf(__half2float(W[(size_t)n*K+k])*inv); q=q<-127?-127:(q>127?127:q); Wi8[(size_t)n*K+k]=(signed char)q; }
}

extern "C" void wint8_quant(const void* W, void* Wi8, void* sW, int N, int K, void* stream){
    int blk=256, grid=(N+blk-1)/blk;
    quant_w<<<grid,blk,0,(hipStream_t)stream>>>((const __half*)W,(signed char*)Wi8,(float*)sW,N,K);
}
extern "C" void wint8_gemm_run(const void* x, const void* Wi8, const void* sW, const void* bias,
                               void* out, int M, int N, int K, void* stream){
    const __half* A=(const __half*)x; const signed char* W=(const signed char*)Wi8;
    const float* s=(const float*)sW; const __half* bs=(const __half*)bias; __half* C=(__half*)out;
    hipStream_t st=(hipStream_t)stream;
    #define LW(WAVES,NB,KB,PAD) do{ const int BM=WAVES*16,BN=NB*16; \
        dim3 blk(WAVES*64); dim3 grd((N+BN-1)/BN,(M+BM-1)/BM); \
        wint8_gemm<WAVES,NB,KB,PAD><<<grd,blk,0,st>>>(A,W,s,bs,C,M,N,K);}while(0)
    if (M <= 512)        LW(2,2,2,8);
    else if (N >= 1536)  LW(8,16,1,8);
    else                 LW(4,4,2,8);
    #undef LW
}
