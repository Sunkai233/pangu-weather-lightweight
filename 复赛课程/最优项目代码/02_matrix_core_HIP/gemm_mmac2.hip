// ============================================================================
// 矩阵核 GEMM LDS-tiled: v_mmac + LDS 复用, 追平 rocBLAS.
//   out[M,N]=x[M,K]@W[N,K]^T(+bias), fp16 I/O, fp32累加. C[m,n]=sum_k A[m,k]*W[n,k].
//
// 布局(实测 bit-exact): lane L: r=L%16, g=L/16.
//   a_frag[i]=A[m=r][k=g*4+i]; b_frag[i]=W[n=r][k=g*4+i]; C acc[i]=C[m=r][n=g+i*4].
//
// Block tile: BM×BN 输出, WAVES 个 wave. 每 wave 算 (16 行 × BN 列)的一条带:
//   wave w 负责输出行 [w*16, w*16+16). BM = WAVES*16.
//   每 wave 持 BN/16 个 C fragment(各 16×16). K 每步 16:
//     从 LDS 读 a_frag(本 wave 16 行 × 16 K) + 各列块 b_frag → mmac 累加.
//   LDS: As[BM][BK], Ws[BN][BK] (行主序), half8 向量化协作载入.
//   BK=16 (一个 mmac 的 K). 可多 KSTEP 一次载更多 K 减同步.
// ============================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

typedef __fp16 half4 __attribute__((ext_vector_type(4)));
typedef float  f32x4 __attribute__((ext_vector_type(4)));
typedef struct __align__(16) { __half2 a,b,c,d; } half8;

// BM=WAVES*16, BN=NB*16, BK=KB*16. WAVES waves/block.
template<int WAVES, int NB, int KB, int PAD>
__global__ __launch_bounds__(WAVES*64)
void gemm_mmac_lds(const __half* __restrict__ A, const __half* __restrict__ W,
                   const __half* __restrict__ bias, __half* __restrict__ C,
                   int M, int N, int K) {
    const int BM = WAVES*16, BN = NB*16, BK = KB*16;
    const int LDK = BK + PAD;
    __shared__ __half As[BM*(BK+PAD)];   // As[m][k]
    __shared__ __half Ws[BN*(BK+PAD)];   // Ws[n][k]
    const int TPB = WAVES*64;
    const int tid = threadIdx.x;
    const int wave = tid >> 6;       // 0..WAVES-1, 负责行块 wave
    const int lane = tid & 63;
    const int r = lane & 15, g = lane >> 4;
    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    // 每 wave 持 NB 个 C fragment(各列块)
    f32x4 acc[NB];
    #pragma unroll
    for(int j=0;j<NB;j++) acc[j]=f32x4{0,0,0,0};

    const int A_units = BM*(BK/8);   // half8 单位
    const int W_units = BN*(BK/8);

    for(int k0=0;k0<K;k0+=BK){
        // 协作载 A tile: As[m][k], m∈[0,BM), k∈[0,BK). half8 沿 K.
        #pragma unroll
        for(int u=tid;u<A_units;u+=TPB){
            int m=u/(BK/8), kk=(u%(BK/8))*8, gm=blockM+m, gk=k0+kk;
            half8 v;
            if(gm<M && gk+7<K) v=*reinterpret_cast<const half8*>(&A[(size_t)gm*K+gk]);
            else { __half t[8];
                #pragma unroll
                for(int e=0;e<8;e++) t[e]=(gm<M&&gk+e<K)?A[(size_t)gm*K+gk+e]:(__half)0;
                v=*reinterpret_cast<half8*>(t);}
            *reinterpret_cast<half8*>(&As[m*LDK+kk])=v;
        }
        #pragma unroll
        for(int u=tid;u<W_units;u+=TPB){
            int n=u/(BK/8), kk=(u%(BK/8))*8, gn=blockN+n, gk=k0+kk;
            half8 v;
            if(gn<N && gk+7<K) v=*reinterpret_cast<const half8*>(&W[(size_t)gn*K+gk]);
            else { __half t[8];
                #pragma unroll
                for(int e=0;e<8;e++) t[e]=(gn<N&&gk+e<K)?W[(size_t)gn*K+gk+e]:(__half)0;
                v=*reinterpret_cast<half8*>(t);}
            *reinterpret_cast<half8*>(&Ws[n*LDK+kk])=v;
        }
        __syncthreads();

        // 每 wave 的 16 行 = As[wave*16 + r][...]; 沿 BK 的 KB 个 16-K 子块 mmac
        #pragma unroll
        for(int ks=0;ks<KB;ks++){
            // a_frag: A[m=wave*16+r][k=ks*16+g*4 + 0..3] 连续 → half4 单 load(64bit)
            half4 a = *reinterpret_cast<const half4*>(&As[(wave*16 + r)*LDK + ks*16 + g*4]);
            #pragma unroll
            for(int j=0;j<NB;j++){
                // b_frag: W[n=j*16+r][k=ks*16+g*4 + 0..3] 连续 → half4 单 load
                half4 b = *reinterpret_cast<const half4*>(&Ws[(j*16 + r)*LDK + ks*16 + g*4]);
                acc[j]=__builtin_amdgcn_mmac_f32_16x16x16f16(a,b,acc[j]);
            }
        }
        __syncthreads();
    }
    // 写回: wave 行块 wave*16, 列块 j. C[m=blockM+wave*16+r][n=blockN+j*16 + g + i*4]
    #pragma unroll
    for(int j=0;j<NB;j++){
        #pragma unroll
        for(int i=0;i<4;i++){
            int gm = blockM + wave*16 + r;
            int gn = blockN + j*16 + g + i*4;
            if(gm<M && gn<N){
                float v=acc[j][i];
                if(bias) v+=__half2float(bias[gn]);
                C[(size_t)gm*N+gn]=__float2half(v);
            }
        }
    }
}

extern "C" void gemm_mmac2_run(const void* x, const void* W, const void* bias,
                               void* out, int M, int N, int K, int variant, void* stream){
    const __half* A=(const __half*)x; const __half* B=(const __half*)W;
    const __half* bs=(const __half*)bias; __half* C=(__half*)out;
    hipStream_t s=(hipStream_t)stream;
    #define LM(WAVES,NB,KB,PAD) do{ const int BM=WAVES*16,BN=NB*16; \
        dim3 blk(WAVES*64); dim3 grd((N+BN-1)/BN,(M+BM-1)/BM); \
        gemm_mmac_lds<WAVES,NB,KB,PAD><<<grd,blk,0,s>>>(A,B,bs,C,M,N,K);}while(0)
    switch(variant){
        case 2100: LM(4,4,1,8); break;   // 64x64 tile, BK16, 4 waves
        case 2101: LM(4,4,2,8); break;   // BK32(KB2)
        case 2102: LM(8,4,1,8); break;   // 128x64
        case 2103: LM(4,8,1,8); break;   // 64x128
        case 2104: LM(8,8,1,8); break;   // 128x128
        case 2105: LM(4,4,3,8); break;   // BK48
        case 2106: LM(2,4,2,8); break;   // 32x64
        case 2107: LM(4,6,1,8); break;   // 64x96
        case 2108: LM(8,4,2,8); break;
        case 2109: LM(4,8,2,8); break;
        case 2110: LM(2,8,1,8); break;
        case 2111: LM(8,8,2,8); break;
        case 2112: LM(4,16,1,8); break;  // 64x256
        case 2113: LM(8,16,1,8); break;  // 128x256
        case 2114: LM(2,2,2,8); break;
        case 2115: LM(4,12,1,8); break;
        default: break;
    }
    #undef LM
}
