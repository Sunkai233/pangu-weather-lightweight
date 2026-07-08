// ============================================================================
// 矩阵核 GEMM v4: 寄存器级 fragment 预取(隐藏 LDS→reg 延迟) + 深 K-unroll.
//   布局(bit-exact): a_frag[i]=A[r][g*4+i], b_frag[i]=W[r][g*4+i], C acc[i]=C[r][g+i*4].
//   优化: 在 mmac 计算当前 K-子块时, 预取下一 K-子块的 a/b fragment 到寄存器,
//         打破 ds_read→v_mmac 的依赖链(rocBLAS Tensile 的核心技巧).
//   每 wave 16 行 × NB×16 列. LDS 单缓冲(占用优先), fragment 寄存器双缓冲.
// ============================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
typedef __fp16 half4 __attribute__((ext_vector_type(4)));
typedef float  f32x4 __attribute__((ext_vector_type(4)));
typedef struct __align__(16) { __half2 a,b,c,d; } half8;

template<int WAVES, int NB, int KB, int PAD>
__global__ __launch_bounds__(WAVES*64)
void mmac4(const __half* __restrict__ A, const __half* __restrict__ W,
           const __half* __restrict__ bias, __half* __restrict__ C,
           int M, int N, int K) {
    const int BM=WAVES*16, BN=NB*16, BK=KB*16, LDK=BK+PAD;
    __shared__ __half As[BM*(BK+PAD)];
    __shared__ __half Ws[BN*(BK+PAD)];
    const int TPB=WAVES*64, tid=threadIdx.x;
    const int wave=tid>>6, lane=tid&63, r=lane&15, g=lane>>4;
    const int blockM=blockIdx.y*BM, blockN=blockIdx.x*BN;
    const int A_units=BM*(BK/8), W_units=BN*(BK/8);

    f32x4 acc[NB];
    #pragma unroll
    for(int j=0;j<NB;j++) acc[j]=f32x4{0,0,0,0};

    for(int k0=0;k0<K;k0+=BK){
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
        #pragma unroll
        for(int u=tid;u<W_units;u+=TPB){
            int n=u/(BK/8), kk=(u%(BK/8))*8, gn=blockN+n, gk=k0+kk;
            half8 v;
            if(gn<N&&gk+7<K) v=*reinterpret_cast<const half8*>(&W[(size_t)gn*K+gk]);
            else{__half t[8];
                #pragma unroll
                for(int e=0;e<8;e++)t[e]=(gn<N&&gk+e<K)?W[(size_t)gn*K+gk+e]:(__half)0;
                v=*reinterpret_cast<half8*>(t);}
            *reinterpret_cast<half8*>(&Ws[n*LDK+kk])=v;
        }
        __syncthreads();

        // 寄存器级预取: 先取 ks=0 的 a + 所有 b, 再循环中预取下一 ks 与 mmac 重叠.
        half4 aCur = *reinterpret_cast<const half4*>(&As[(wave*16+r)*LDK + g*4]);
        half4 bCur[NB];
        #pragma unroll
        for(int j=0;j<NB;j++) bCur[j]=*reinterpret_cast<const half4*>(&Ws[(j*16+r)*LDK + g*4]);
        #pragma unroll
        for(int ks=0;ks<KB;ks++){
            half4 aNext; half4 bNext[NB];
            if(ks+1<KB){
                aNext=*reinterpret_cast<const half4*>(&As[(wave*16+r)*LDK + (ks+1)*16 + g*4]);
                #pragma unroll
                for(int j=0;j<NB;j++) bNext[j]=*reinterpret_cast<const half4*>(&Ws[(j*16+r)*LDK + (ks+1)*16 + g*4]);
            }
            #pragma unroll
            for(int j=0;j<NB;j++) acc[j]=__builtin_amdgcn_mmac_f32_16x16x16f16(aCur,bCur[j],acc[j]);
            aCur=aNext;
            #pragma unroll
            for(int j=0;j<NB;j++) bCur[j]=bNext[j];
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

extern "C" void gemm_mmac4_run(const void* x, const void* W, const void* bias,
                               void* out, int M, int N, int K, int variant, void* stream){
    const __half* A=(const __half*)x; const __half* B=(const __half*)W;
    const __half* bs=(const __half*)bias; __half* C=(__half*)out;
    hipStream_t s=(hipStream_t)stream;
    #define L4(WAVES,NB,KB,PAD) do{ const int BM=WAVES*16,BN=NB*16; \
        dim3 blk(WAVES*64); dim3 grd((N+BN-1)/BN,(M+BM-1)/BM); \
        mmac4<WAVES,NB,KB,PAD><<<grd,blk,0,s>>>(A,B,bs,C,M,N,K);}while(0)
    switch(variant){
        case 2300: L4(4,4,2,8); break;   // 64x64 BK32 + reg prefetch
        case 2301: L4(4,4,3,8); break;   // BK48
        case 2302: L4(4,4,4,8); break;   // BK64(更深K-unroll)
        case 2303: L4(8,8,2,8); break;   // 128x128 BK32
        case 2304: L4(8,16,1,8); break;  // 128x256
        case 2305: L4(4,8,2,8); break;   // 64x128 BK32
        case 2306: L4(2,2,2,8); break;   // 32x32 BK32
        case 2307: L4(4,4,6,8); break;   // BK96(K=192 一次全载)
        case 2308: L4(8,8,1,8); break;
        case 2309: L4(4,6,2,8); break;   // 64x96
        case 2310: L4(2,4,2,8); break;   // 32x64
        case 2311: L4(4,4,1,8); break;   // BK16 对照
        default: break;
    }
    #undef L4
}
