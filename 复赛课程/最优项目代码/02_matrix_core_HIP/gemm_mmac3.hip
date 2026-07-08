// ============================================================================
// 矩阵核 GEMM v3: LDS 双缓冲 + 大 tile, 追平 rocBLAS.
//   布局(bit-exact): lane L: r=L%16,g=L/16.
//     a_frag[i]=A[m=r][k=g*4+i]; b_frag[i]=W[n=r][k=g*4+i]; C acc[i]=C[m=r][n=g+i*4].
//   双缓冲: 预取下一 K 块到另一 LDS buffer, 与当前 mmac 重叠.
//   每 wave 算 16 行 × BN 列(NB 个 16-col 块). BM=WAVES*16.
// ============================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
typedef __fp16 half4 __attribute__((ext_vector_type(4)));
typedef float  f32x4 __attribute__((ext_vector_type(4)));
typedef struct __align__(16) { __half2 a,b,c,d; } half8;

template<int WAVES, int NB, int KB, int PAD>
__global__ __launch_bounds__(WAVES*64)
void gemm_mmac_db(const __half* __restrict__ A, const __half* __restrict__ W,
                  const __half* __restrict__ bias, __half* __restrict__ C,
                  int M, int N, int K) {
    const int BM=WAVES*16, BN=NB*16, BK=KB*16, LDK=BK+PAD;
    __shared__ __half As[2][BM*(BK+PAD)];
    __shared__ __half Ws[2][BN*(BK+PAD)];
    const int TPB=WAVES*64, tid=threadIdx.x;
    const int wave=tid>>6, lane=tid&63, r=lane&15, g=lane>>4;
    const int blockM=blockIdx.y*BM, blockN=blockIdx.x*BN;
    const int A_units=BM*(BK/8), W_units=BN*(BK/8);

    f32x4 acc[NB];
    #pragma unroll
    for(int j=0;j<NB;j++) acc[j]=f32x4{0,0,0,0};

    auto load=[&](int k0,int buf){
        #pragma unroll
        for(int u=tid;u<A_units;u+=TPB){
            int m=u/(BK/8), kk=(u%(BK/8))*8, gm=blockM+m, gk=k0+kk;
            half8 v;
            if(gm<M&&gk+7<K) v=*reinterpret_cast<const half8*>(&A[(size_t)gm*K+gk]);
            else{__half t[8];
                #pragma unroll
                for(int e=0;e<8;e++)t[e]=(gm<M&&gk+e<K)?A[(size_t)gm*K+gk+e]:(__half)0;
                v=*reinterpret_cast<half8*>(t);}
            *reinterpret_cast<half8*>(&As[buf][m*LDK+kk])=v;
        }
        #pragma unroll
        for(int u=tid;u<W_units;u+=TPB){
            int n=u/(BK/8), kk=(u%(BK/8))*8, gn=blockN+n, gk=k0+kk;
            half8 v;
            if(gn<N&&gk+7<K) v=*reinterpret_cast<const half8*>(&W[(size_t)gn*K+gk]);
            else{__half t[8];
                #pragma unroll
                for(int e=0;e<8;e++)t[e]=(gn<N&&gk+e<K)?W[(size_t)gn*K+gk+e]:(__half)0;
                v=*reinterpret_cast<half8*>(t);}
            *reinterpret_cast<half8*>(&Ws[buf][n*LDK+kk])=v;
        }
    };

    int buf=0;
    load(0,0);
    __syncthreads();
    for(int k0=0;k0<K;k0+=BK){
        int nbuf=buf^1;
        if(k0+BK<K) load(k0+BK,nbuf);
        #pragma unroll
        for(int ks=0;ks<KB;ks++){
            half4 a;
            #pragma unroll
            for(int i=0;i<4;i++) a[i]=(__fp16)__half2float(As[buf][(wave*16+r)*LDK + ks*16 + g*4 + i]);
            #pragma unroll
            for(int j=0;j<NB;j++){
                half4 b;
                #pragma unroll
                for(int i=0;i<4;i++) b[i]=(__fp16)__half2float(Ws[buf][(j*16+r)*LDK + ks*16 + g*4 + i]);
                acc[j]=__builtin_amdgcn_mmac_f32_16x16x16f16(a,b,acc[j]);
            }
        }
        __syncthreads();
        buf=nbuf;
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

extern "C" void gemm_mmac3_run(const void* x, const void* W, const void* bias,
                               void* out, int M, int N, int K, int variant, void* stream){
    const __half* A=(const __half*)x; const __half* B=(const __half*)W;
    const __half* bs=(const __half*)bias; __half* C=(__half*)out;
    hipStream_t s=(hipStream_t)stream;
    #define LD(WAVES,NB,KB,PAD) do{ const int BM=WAVES*16,BN=NB*16; \
        dim3 blk(WAVES*64); dim3 grd((N+BN-1)/BN,(M+BM-1)/BM); \
        gemm_mmac_db<WAVES,NB,KB,PAD><<<grd,blk,0,s>>>(A,B,bs,C,M,N,K);}while(0)
    switch(variant){
        case 2200: LD(8,8,1,8); break;    // 128x128 db
        case 2201: LD(8,16,1,8); break;   // 128x256 db
        case 2202: LD(16,16,1,8); break;  // 256x256 db (1024 thr) ~49KB LDS
        case 2203: LD(8,16,2,8); break;   // BK32
        case 2204: LD(4,16,1,8); break;   // 64x256 db
        case 2205: LD(16,8,1,8); break;   // 256x128
        case 2206: LD(8,8,2,8); break;
        case 2207: LD(16,8,2,8); break;   // 256x128 BK32
        case 2208: LD(4,8,2,8); break;
        case 2209: LD(8,12,1,8); break;   // 128x192
        case 2210: LD(12,16,1,8); break;  // 192x256
        case 2211: LD(12,12,1,8); break;  // 192x192
        case 2212: LD(16,12,1,8); break;
        case 2213: LD(4,4,2,8); break;
        case 2214: LD(8,4,1,8); break;
        case 2215: LD(6,16,1,8); break;
        default: break;
    }
    #undef LD
}
