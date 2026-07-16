// flash 窗口注意力 HIP 动态库:纯 hipcc 编译(不经 hipify),extern "C" 给 PyTorch ctypes 调用。
// 输入 q,k,v: [Bn,H,N,D] fp32 连续(Bn=窗口数,H=头数);bias:[H,N,N];scale:[H]。
// 做 QK-Norm(q,k L2归一)+ scale + 相对偏置 + 在线softmax + AV,分数矩阵不落 HBM。
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdlib>
#define M 144
#define D 16

__global__ void flash_k(const float* __restrict__ q, const float* __restrict__ k,
                        const float* __restrict__ v, const float* __restrict__ bias,
                        const float* __restrict__ scale, float* __restrict__ out, int H){
    int blk = blockIdx.x; int h = blk % H; int i = threadIdx.x;
    __shared__ float sk[M*D];
    const float* qb = q + (size_t)blk*M*D;
    const float* kb = k + (size_t)blk*M*D;
    const float* vb = v + (size_t)blk*M*D;
    const float* bb = bias + (size_t)h*M*M;
    for(int t=threadIdx.x; t<M*D; t+=blockDim.x) sk[t]=kb[t];
    __syncthreads();
    if(i<M){
        float kn=0.f; for(int c=0;c<D;c++){ float u=sk[i*D+c]; kn+=u*u; }
        float rr=rsqrtf(kn+1e-12f); for(int c=0;c<D;c++) sk[i*D+c]*=rr;
    }
    __syncthreads();
    if(i>=M) return;
    float qi[D]; float nrm=0.f;
    for(int c=0;c<D;c++){ qi[c]=qb[i*D+c]; nrm+=qi[c]*qi[c]; }
    float rq=rsqrtf(nrm+1e-12f); float sc=scale[h];
    for(int c=0;c<D;c++) qi[c]*=rq;
    float m=-1e30f, l=0.f, acc[D]; for(int c=0;c<D;c++) acc[c]=0.f;
    const float* br = bb + (size_t)i*M;
    for(int j=0;j<M;j++){
        float dot=0.f; const float* kj=sk+j*D;
        for(int c=0;c<D;c++) dot+=qi[c]*kj[c];
        float s = sc*dot + br[j];
        float mn = fmaxf(m,s); float corr=__expf(m-mn), p=__expf(s-mn);
        l=l*corr+p; const float* vj=vb+j*D;
        for(int c=0;c<D;c++) acc[c]=acc[c]*corr + p*vj[c];
        m=mn;
    }
    float inv=1.f/l; float* ob = out + (size_t)blk*M*D + (size_t)i*D;
    for(int c=0;c<D;c++) ob[c]=acc[c]*inv;
}

extern "C" void flash_attn_launch(const float* q, const float* k, const float* v,
                                  const float* bias, const float* scale, float* out,
                                  int Bn, int H, void* stream){
    int grid = Bn*H;
    flash_k<<<grid, 192, 0, (hipStream_t)stream>>>(q,k,v,bias,scale,out,H);
}

// ---- fp16 版:I/O 为 __half,累加用 fp32(匹配模型 fp16,免转换开销)----
__global__ void flash_k_h(const __half* __restrict__ q, const __half* __restrict__ k,
                          const __half* __restrict__ v, const __half* __restrict__ bias,
                          const float* __restrict__ scale, __half* __restrict__ out, int H){
    int blk = blockIdx.x; int h = blk % H; int i = threadIdx.x;
    __shared__ float sk[M*D];
    const __half* qb = q + (size_t)blk*M*D;
    const __half* kb = k + (size_t)blk*M*D;
    const __half* vb = v + (size_t)blk*M*D;
    const __half* bb = bias + (size_t)h*M*M;
    for(int t=threadIdx.x; t<M*D; t+=blockDim.x) sk[t]=__half2float(kb[t]);
    __syncthreads();
    if(i<M){
        float kn=0.f; for(int c=0;c<D;c++){ float u=sk[i*D+c]; kn+=u*u; }
        float rr=rsqrtf(kn+1e-12f); for(int c=0;c<D;c++) sk[i*D+c]*=rr;
    }
    __syncthreads();
    if(i>=M) return;
    float qi[D]; float nrm=0.f;
    for(int c=0;c<D;c++){ qi[c]=__half2float(qb[i*D+c]); nrm+=qi[c]*qi[c]; }
    float rq=rsqrtf(nrm+1e-12f); float sc=scale[h];
    for(int c=0;c<D;c++) qi[c]*=rq;
    float m=-1e30f, l=0.f, acc[D]; for(int c=0;c<D;c++) acc[c]=0.f;
    const __half* br = bb + (size_t)i*M;
    for(int j=0;j<M;j++){
        float dot=0.f; const float* kj=sk+j*D;
        for(int c=0;c<D;c++) dot+=qi[c]*kj[c];
        float s = sc*dot + __half2float(br[j]);
        float mn = fmaxf(m,s); float corr=__expf(m-mn), p=__expf(s-mn);
        l=l*corr+p; const __half* vj=vb+j*D;
        for(int c=0;c<D;c++) acc[c]=acc[c]*corr + p*__half2float(vj[c]);
        m=mn;
    }
    float inv=1.f/l; __half* ob = out + (size_t)blk*M*D + (size_t)i*D;
    for(int c=0;c<D;c++) ob[c]=__float2half(acc[c]*inv);
}

extern "C" void flash_attn_launch_h(const void* q, const void* k, const void* v,
                                    const void* bias, const float* scale, void* out,
                                    int Bn, int H, void* stream){
    int grid = Bn*H;
    flash_k_h<<<grid, 192, 0, (hipStream_t)stream>>>(
        (const __half*)q,(const __half*)k,(const __half*)v,
        (const __half*)bias, scale, (__half*)out, H);
}

// ---- 融合读版:直接吃 self.qkv(x) 的输出 qkv[Bn,N,3*H*D],免 permute/contiguous ----
// 布局: token i 的 q/k/v 在同一行,q=[0..HD), k=[HD..2HD), v=[2HD..3HD);每段内 head h 偏移 h*D。
// 输出 out[Bn,N,H*D]。partition 已在外部按窗口切好(每窗 N=M=144)。
__global__ void flash_qkv_k_h(const __half* __restrict__ qkv, const __half* __restrict__ bias,
                              const float* __restrict__ scale, __half* __restrict__ out, int H){
    int blk = blockIdx.x; int bn = blk / H, h = blk % H; int i = threadIdx.x;
    int HD = H*D, S = 3*HD;
    const __half* base = qkv + (size_t)bn*M*S;
    const __half* bb = bias + (size_t)h*M*M;
    // k 和 v 都放进 LDS(fp32 存)。原先只缓存 k,v 留在全局内存里,导致 block 内 M 个线程
    // (每个负责一个 query)在 j 循环中各自把同一份 v 重复读一遍。
    // j 循环中所有线程读同一 sk[j*D+c]/sv[j*D+c],属 LDS 广播,无 bank conflict。
    // 注:试过把 LDS 换成 half 存(18KB→9KB, 提高 occupancy),实测反而慢 12% ——
    //    该 kernel 不受 occupancy 限制,而 half→float 的转换指令加重了内层循环。故保持 fp32。
    __shared__ float sk[M*D];
    __shared__ float sv[M*D];
    for(int t=threadIdx.x; t<M*D; t+=blockDim.x){
        int j=t/D, c=t%D;
        sk[t]=__half2float(base[(size_t)j*S + HD    + h*D + c]);   // k 段
        sv[t]=__half2float(base[(size_t)j*S + 2*HD  + h*D + c]);   // v 段
    }
    __syncthreads();
    if(i<M){
        float kn=0.f; for(int c=0;c<D;c++){ float u=sk[i*D+c]; kn+=u*u; }
        float rr=rsqrtf(kn+1e-12f); for(int c=0;c<D;c++) sk[i*D+c]*=rr;
    }
    __syncthreads();
    if(i>=M) return;
    const __half* qrow = base + (size_t)i*S + h*D;           // q 段
    float qi[D]; float nrm=0.f;
    for(int c=0;c<D;c++){ qi[c]=__half2float(qrow[c]); nrm+=qi[c]*qi[c]; }
    float rq=rsqrtf(nrm+1e-12f); float sc=scale[h];
    for(int c=0;c<D;c++) qi[c]*=rq;
    float m=-1e30f, l=0.f, acc[D]; for(int c=0;c<D;c++) acc[c]=0.f;
    const __half* br = bb + (size_t)i*M;
    for(int j=0;j<M;j++){
        float dot=0.f; const float* kj=sk+j*D;
        for(int c=0;c<D;c++) dot+=qi[c]*kj[c];
        float s = sc*dot + __half2float(br[j]);
        const float* vj = sv + j*D;                            // v 段(LDS,广播读)
        // 在线 softmax:max 单调增,corr/p 必有一个=1只算另一个(省一半);exp→exp2(硬件更快,换底等价)
        if(s > m){
            float corr=exp2f((m-s)*1.442695041f); l=l*corr+1.f;
            for(int c=0;c<D;c++) acc[c]=acc[c]*corr + vj[c];
            m=s;
        } else {
            float p=exp2f((s-m)*1.442695041f); l+=p;
            for(int c=0;c<D;c++) acc[c]+=p*vj[c];
        }
    }
    float inv=1.f/l; __half* ob = out + (size_t)bn*M*HD + (size_t)i*HD + h*D;
    for(int c=0;c<D;c++) ob[c]=__float2half(acc[c]*inv);
}

// ============================================================================
// 矩阵核(MFMA)版:把 QK^T 与 AV 变成 16x16x16 的 tile 矩阵乘。
// 标量版每个线程要串行跑 M=144 轮、每轮 ~40 条标量指令(约 5760 条),只能发挥约 8% 的算力;
// 一条 __builtin_amdgcn_mmac_f32_16x16x16f16 等价 4096 次乘加,指令数降一个数量级。
//
// 布局与 gemm_lib_mmac.hip 完全一致(那里已实测 bit-exact):
//   lane L: r = L%16, g = L/16
//   a_frag[i] = A[m=r][k=g*4+i]      b_frag[i] = B[n=r][k=g*4+i]     acc[i] = C[m=r][n=g+i*4]
//   mmac(a,b,acc) 计算 C += A[M,K] @ B[N,K]^T
//
// 映射:M=144 切成 MTIL=9 个 16 行 tile;D=16 恰为 MFMA 的 K 维,一次吃完。
//   ① S = Q[144,16] @ K[144,16]^T   —— 天然就是 mmac 的 A@B^T 语义
//   ② O = P[144,144] @ V[144,16]    —— 需 B 为 [N,K]=[16,144],故 V 转置存入 LDS
// MFMA 的输出布局(C[r][g+i*4])与输入布局(A[r][g*4+i])不同,故 softmax 后的 P 过一趟 LDS 换布局。
// 一个 block = 一个(窗口,头),单 wave(64 线程),循环 9 个 m-tile。
// ============================================================================
typedef __fp16 half4v __attribute__((ext_vector_type(4)));
typedef float  f32x4v __attribute__((ext_vector_type(4)));
#define MTIL (M/16)      // 9

__global__ __launch_bounds__(64)
void flash_qkv_mfma_h(const __half* __restrict__ qkv, const __half* __restrict__ bias,
                      const float* __restrict__ scale, __half* __restrict__ out, int H){
    const int blk = blockIdx.x, bn = blk / H, h = blk % H;
    const int lane = threadIdx.x, r = lane & 15, g = lane >> 4;
    const int HD = H*D, S3 = 3*HD;
    const __half* base = qkv + (size_t)bn*M*S3;
    const __half* bb   = bias + (size_t)h*M*M;
    const float sc = scale[h];

    __shared__ __half Ks[M*D];     // K[j][d]        → b_frag: B[n=j][k=d]
    __shared__ __half Vts[D*M];    // V^T[d][j]      → b_frag: B[n=d][k=j]
    __shared__ __half Ps[16*M];    // P[m][j] (当前 m-tile) → a_frag: A[m][k=j]

    // ── 载入 K(含 QK-Norm 行归一)与 V^T ──
    for(int j=lane; j<M; j+=64){
        const __half* krow = base + (size_t)j*S3 + HD   + h*D;
        const __half* vrow = base + (size_t)j*S3 + 2*HD + h*D;
        float kv[D], kn=0.f;
        #pragma unroll
        for(int c=0;c<D;c++){ kv[c]=__half2float(krow[c]); kn+=kv[c]*kv[c]; }
        float rr = rsqrtf(kn+1e-12f);
        #pragma unroll
        for(int c=0;c<D;c++){
            Ks[j*D + c]  = __float2half(kv[c]*rr);
            Vts[c*M + j] = vrow[c];                 // 转置写入
        }
    }
    __syncthreads();

    for(int mt=0; mt<MTIL; mt++){
        // ── Q tile(含 QK-Norm)。同一行的 16 个分量分布在 g=0..3 的四个 lane 上,跨 lane 归约 ──
        const __half* qrow = base + (size_t)(mt*16 + r)*S3 + h*D;
        float q4[4]; float qn=0.f;
        #pragma unroll
        for(int i=0;i<4;i++){ q4[i]=__half2float(qrow[g*4+i]); qn += q4[i]*q4[i]; }
        qn += __shfl_xor(qn, 16, 64);
        qn += __shfl_xor(qn, 32, 64);
        float rq = rsqrtf(qn+1e-12f);
        half4v a;
        #pragma unroll
        for(int i=0;i<4;i++) a[i] = (__fp16)(q4[i]*rq);

        // ── ① S = Q @ K^T ──
        f32x4v acc[MTIL];
        #pragma unroll
        for(int nt=0; nt<MTIL; nt++){
            f32x4v z = {0.f,0.f,0.f,0.f};
            half4v b = *reinterpret_cast<const half4v*>(&Ks[(nt*16 + r)*D + g*4]);
            acc[nt] = __builtin_amdgcn_mmac_f32_16x16x16f16(a, b, z);
        }
        // acc[nt][i] = S[m=r][n = nt*16 + g + i*4]

        // ── scale + 相对位置偏置,行内 max(跨 g 的四个 lane 归约)──
        const __half* br = bb + (size_t)(mt*16 + r)*M;
        float mx = -1e30f;
        #pragma unroll
        for(int nt=0; nt<MTIL; nt++)
            #pragma unroll
            for(int i=0;i<4;i++){
                float v = sc*acc[nt][i] + __half2float(br[nt*16 + g + i*4]);
                acc[nt][i] = v;
                mx = fmaxf(mx, v);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 64));
        mx = fmaxf(mx, __shfl_xor(mx, 32, 64));

        float sum = 0.f;
        #pragma unroll
        for(int nt=0; nt<MTIL; nt++)
            #pragma unroll
            for(int i=0;i<4;i++){
                float p = exp2f((acc[nt][i]-mx)*1.442695041f);   // exp2 换底,与 exp 等价
                acc[nt][i] = p; sum += p;
            }
        sum += __shfl_xor(sum, 16, 64);
        sum += __shfl_xor(sum, 32, 64);
        const float inv = 1.f/sum;

        // ── P 写入 LDS,换成 a_frag 需要的布局 ──
        __syncthreads();                                  // 上一轮的 Ps 已被读完
        #pragma unroll
        for(int nt=0; nt<MTIL; nt++)
            #pragma unroll
            for(int i=0;i<4;i++)
                Ps[r*M + nt*16 + g + i*4] = __float2half(acc[nt][i]*inv);
        __syncthreads();

        // ── ② O = P @ V(V 已转置,故仍是 A@B^T)──
        f32x4v o = {0.f,0.f,0.f,0.f};
        #pragma unroll
        for(int kt=0; kt<MTIL; kt++){
            half4v pa = *reinterpret_cast<const half4v*>(&Ps [r*M + kt*16 + g*4]);
            half4v vb = *reinterpret_cast<const half4v*>(&Vts[r*M + kt*16 + g*4]);
            o = __builtin_amdgcn_mmac_f32_16x16x16f16(pa, vb, o);
        }
        // o[i] = O[m=r][d = g + i*4]
        __half* ob = out + (size_t)bn*M*HD + (size_t)(mt*16 + r)*HD + h*D;
        #pragma unroll
        for(int i=0;i<4;i++) ob[g + i*4] = __float2half(o[i]);
    }
}

// 入口:走矩阵核版(实测 enc 2.09x / mid 1.84x,端到端前向 26.2→21.6ms,
// 与 torch fp32 参考的相对误差 6.9e-4,与标量版 2.8e-4 同量级)。
extern "C" void flash_qkv_launch_h(const void* qkv, const void* bias, const float* scale,
                                   void* out, int Bn, int H, void* stream){
    int grid = Bn*H;
    flash_qkv_mfma_h<<<grid, 64, 0, (hipStream_t)stream>>>(
        (const __half*)qkv, (const __half*)bias, scale, (__half*)out, H);
}

// 标量参考实现(保留:矩阵核不可用时的等价路径,也是上面 MFMA 版的数值对照基准)
extern "C" void flash_qkv_scalar_launch_h(const void* qkv, const void* bias, const float* scale,
                                          void* out, int Bn, int H, void* stream){
    int grid = Bn*H;
    flash_qkv_k_h<<<grid, 192, 0, (hipStream_t)stream>>>(
        (const __half*)qkv, (const __half*)bias, scale, (__half*)out, H);
}

// ---- 整层融合:partition/reverse 折进索引 ----
// qkv:[realN,3HD](全网格上qkv GEMM结果);idx:[nW*M] int32(窗口内每位置→真实token索引,padding=-1);
// qkvb:[3HD] qkv偏置(padding位置=qkv(0)=bias);bias:[H,M,M];scale:[H];out:[realN,HD]
// 一个block=(窗口w,头h):按idx从qkv gather k/v进LDS(padding用qkvb)→在线softmax→按idx scatter out。
// 免去 grid_partition/reverse 拷贝 + pad + reshape。
__global__ void flash_grid_fused_h(const __half* __restrict__ qkv, const __half* __restrict__ qkvb,
                                   const int* __restrict__ idx, const __half* __restrict__ bias,
                                   const float* __restrict__ scale, __half* __restrict__ out, int H){
    int blk = blockIdx.x; int w = blk / H, h = blk % H; int i = threadIdx.x;
    int HD = H*D, S = 3*HD;
    const int* idxw = idx + (size_t)w*M;
    const __half* bb = bias + (size_t)h*M*M;
    __shared__ float sk[M*D];
    __shared__ float sv[M*D];
    // gather k,v (k段偏移HD, v段偏移2HD; 头h偏移h*D)
    for(int t=threadIdx.x; t<M*D; t+=blockDim.x){
        int j=t/D, c=t%D; int tok=idxw[j];
        const __half* src = (tok>=0) ? (qkv + (size_t)tok*S) : qkvb;
        sk[t]=__half2float(src[HD + h*D + c]);
        sv[t]=__half2float(src[2*HD + h*D + c]);
    }
    __syncthreads();
    if(i<M){ float kn=0.f; for(int c=0;c<D;c++){float u=sk[i*D+c];kn+=u*u;} float r=rsqrtf(kn+1e-12f); for(int c=0;c<D;c++) sk[i*D+c]*=r; }
    __syncthreads();
    if(i>=M) return;
    int toki = idxw[i];
    const __half* qsrc = (toki>=0) ? (qkv + (size_t)toki*S) : qkvb;  // q段偏移0
    float qi[D]; float nrm=0.f;
    for(int c=0;c<D;c++){ qi[c]=__half2float(qsrc[h*D+c]); nrm+=qi[c]*qi[c]; }
    float rq=rsqrtf(nrm+1e-12f); float sc=scale[h];
    for(int c=0;c<D;c++) qi[c]*=rq;
    float m=-1e30f, l=0.f, acc[D]; for(int c=0;c<D;c++) acc[c]=0.f;
    const __half* br = bb + (size_t)i*M;
    for(int j=0;j<M;j++){
        float dot=0.f; const float* kj=sk+j*D;
        for(int c=0;c<D;c++) dot+=qi[c]*kj[c];
        float s = sc*dot + __half2float(br[j]);
        const float* vj=sv+j*D;
        if(s>m){ float corr=exp2f((m-s)*1.442695041f); l=l*corr+1.f; for(int c=0;c<D;c++) acc[c]=acc[c]*corr+vj[c]; m=s; }
        else   { float p=exp2f((s-m)*1.442695041f); l+=p; for(int c=0;c<D;c++) acc[c]+=p*vj[c]; }
    }
    if(toki<0) return;                       // padding query 不写出
    float inv=1.f/l; __half* ob = out + (size_t)toki*HD + (size_t)h*D;
    for(int c=0;c<D;c++) ob[c]=__float2half(acc[c]*inv);
}
extern "C" void flash_grid_fused_launch_h(const void* qkv, const void* qkvb, const void* idx,
                                          const void* bias, const float* scale, void* out,
                                          int nW, int H, void* stream){
    flash_grid_fused_h<<<nW*H, 192, 0, (hipStream_t)stream>>>(
        (const __half*)qkv,(const __half*)qkvb,(const int*)idx,(const __half*)bias,scale,(__half*)out,H);
}

// ---- fp16 RMSNorm:wavefront/行 + 寄存器缓存 + shfl 归约(对应主机版 v2,587GB/s)----
__global__ void rmsnorm_k_h(const __half* __restrict__ x, const __half* __restrict__ w,
                            __half* __restrict__ y, int N, int C, float eps){
    int wid=threadIdx.x>>6, lane=threadIdx.x&63, nwf=blockDim.x>>6;
    int row=blockIdx.x*nwf+wid; if(row>=N) return;
    const __half* xr=x+(size_t)row*C; __half* yr=y+(size_t)row*C;
    float reg[8]; int cnt=0; float local=0.f;
    for(int i=lane;i<C;i+=64){ float v=__half2float(xr[i]); reg[cnt++]=v; local+=v*v; }
    for(int o=32;o>0;o>>=1) local+=__shfl_down(local,o,64);
    local=__shfl(local,0,64);
    float r=rsqrtf(local/C+eps); cnt=0;
    for(int i=lane;i<C;i+=64){ yr[i]=__float2half(reg[cnt++]*r*__half2float(w[i])); }
}
extern "C" void rmsnorm_launch_h(const void* x, const void* w, void* y,
                                 int N, int C, float eps, void* stream){
    int warps=4, blk=warps*64, grid=(N+warps-1)/warps;
    rmsnorm_k_h<<<grid, blk, 0, (hipStream_t)stream>>>(
        (const __half*)x, (const __half*)w, (__half*)y, N, C, eps);
}
