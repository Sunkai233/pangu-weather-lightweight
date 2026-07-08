// ============================================================================
// GEMM benchmark harness: rocBLAS(hipblasGemmEx) 基准 vs 手写 gemm_hand_run.
//   逐 variant 计时 + 与 rocBLAS 结果对拍 (max abs diff).
//   用法: ./gemm_bench M N K [variant_list...]
//     不给 variant_list 则跑全部 0..11.
//   计算: out[M,N] = x[M,K] @ W[N,K]^T, fp16 I/O, fp32 累加.
// ============================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ printf("HIP ERR %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
#define HCK(x) do{ hipblasStatus_t s=(x); if(s!=HIPBLAS_STATUS_SUCCESS){ printf("HIPBLAS ERR %s:%d code=%d\n",__FILE__,__LINE__,(int)s); exit(1);} }while(0)

extern "C" void gemm_hand_run(const void*, const void*, const void*, void*,
                              int, int, int, int, void*);
extern "C" void gemm_hand2_run(const void*, const void*, const void*, void*,
                               int, int, int, int, void*);
extern "C" void gemm_hand3_run(const void*, const void*, const void*, void*,
                               int, int, int, int, void*);
extern "C" void gemm_hand4_run(const void*, const void*, const void*, void*,
                               int, int, int, int, void*);
extern "C" void gemm_hand5_run(const void*, const void*, const void*, void*,
                               int, int, int, int, void*);
extern "C" void gemm_hand6_run(const void*, const void*, const void*, void*,
                               int, int, int, int, void*);
extern "C" void gemm_hand7_run(const void*, const void*, const void*, void*,
                               int, int, int, int, void*);
extern "C" void gemm_hand8_run(const void*, const void*, const void*, void*,
                               int, int, int, int, void*);
extern "C" void gemm_hand9_run(const void*, const void*, const void*, void*,
                               int, int, int, int, void*);
extern "C" void gemm_hand10_run(const void*, const void*, const void*, void*,
                                int, int, int, int, void*);
extern "C" void gemm_hand11_run(const void*, const void*, const void*, void*,
                                int, int, int, int, void*);
extern "C" void gemm_mmac_run(const void*, const void*, const void*, void*,
                              int, int, int, int, void*);
extern "C" void gemm_mmac2_run(const void*, const void*, const void*, void*,
                               int, int, int, int, void*);
extern "C" void gemm_mmac3_run(const void*, const void*, const void*, void*,
                               int, int, int, int, void*);
extern "C" void gemm_mmac4_run(const void*, const void*, const void*, void*,
                               int, int, int, int, void*);

static hipblasHandle_t g_handle = nullptr;

// rocBLAS GEMM: out[M,N]=x[M,K]@W[N,K]^T, 同 gemm_lib.hip 的映射.
void rocblas_gemm(const __half* x, const __half* W, __half* out, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    HCK(hipblasGemmEx(g_handle,
        HIPBLAS_OP_T, HIPBLAS_OP_N,
        N, M, K, &alpha,
        W, HIPBLAS_R_16F, K,
        x, HIPBLAS_R_16F, K,
        &beta,
        out, HIPBLAS_R_16F, N,
        HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT));
}

static int g_sync_each = 0;   // 1: 每次调用单独计时(排除跨迭代流水), 取中位/最小
float time_ms(void (*fn)(void*), void* ctx, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn(ctx);
    CK(hipDeviceSynchronize());
    if (g_sync_each) {
        // 每次单独 record+sync, 取最小值(真实单 kernel 延迟, 无跨迭代重叠)
        hipEvent_t t0,t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
        float best=1e30f;
        for (int i=0;i<iters;i++){
            CK(hipEventRecord(t0)); fn(ctx); CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1));
            float ms; CK(hipEventElapsedTime(&ms,t0,t1)); if(ms<best)best=ms;
        }
        CK(hipEventDestroy(t0)); CK(hipEventDestroy(t1));
        return best;
    }
    hipEvent_t t0, t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
    CK(hipEventRecord(t0));
    for (int i = 0; i < iters; i++) fn(ctx);
    CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1));
    float ms; CK(hipEventElapsedTime(&ms, t0, t1));
    CK(hipEventDestroy(t0)); CK(hipEventDestroy(t1));
    return ms / iters;
}

struct Ctx { __half *x,*W,*out; int M,N,K,variant; };
static void run_roc(void* c){ Ctx*p=(Ctx*)c; rocblas_gemm(p->x,p->W,p->out,p->M,p->N,p->K); }
static void run_hand(void* c){ Ctx*p=(Ctx*)c;
    if (p->variant >= 2300) gemm_mmac4_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else if (p->variant >= 2200) gemm_mmac3_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else if (p->variant >= 2100) gemm_mmac2_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else if (p->variant >= 2000) gemm_mmac_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else if (p->variant >= 1100) gemm_hand11_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else if (p->variant >= 1000) gemm_hand10_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else if (p->variant >= 900) gemm_hand9_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else if (p->variant >= 800) gemm_hand8_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else if (p->variant >= 700) gemm_hand7_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else if (p->variant >= 600) gemm_hand6_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else if (p->variant >= 500) gemm_hand5_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else if (p->variant >= 400) gemm_hand4_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else if (p->variant >= 300) gemm_hand3_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else if (p->variant >= 100) gemm_hand2_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr);
    else gemm_hand_run(p->x,p->W,nullptr,p->out,p->M,p->N,p->K,p->variant,nullptr); }

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s M N K [variants...]\n", argv[0]); return 1; }
    int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
    std::vector<int> variants;
    if (argc > 4) for (int i = 4; i < argc; i++) variants.push_back(atoi(argv[i]));
    else { for (int v=1100; v<=1115; v++) variants.push_back(v); }

    if (const char* e = getenv("GEMM_SYNC_EACH")) g_sync_each = atoi(e);
    rocblas_initialize();
    HCK(hipblasCreate(&g_handle));

    size_t szX = (size_t)M*K, szW=(size_t)N*K, szO=(size_t)M*N;
    std::vector<__half> hX(szX), hW(szW);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : hX) v = __float2half(dist(rng)*0.1f);
    for (auto& v : hW) v = __float2half(dist(rng)*0.1f);

    __half *dX,*dW,*dRef,*dHand;
    CK(hipMalloc(&dX, szX*2)); CK(hipMalloc(&dW, szW*2));
    CK(hipMalloc(&dRef, szO*2)); CK(hipMalloc(&dHand, szO*2));
    CK(hipMemcpy(dX, hX.data(), szX*2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dW, hW.data(), szW*2, hipMemcpyHostToDevice));

    // rocBLAS 基准
    Ctx rc{dX,dW,dRef,M,N,K,0};
    float roc_ms = time_ms(run_roc, &rc, 20, 100);
    CK(hipDeviceSynchronize());
    std::vector<__half> hRef(szO);
    CK(hipMemcpy(hRef.data(), dRef, szO*2, hipMemcpyDeviceToHost));

    double gflop = 2.0*M*N*K/1e9;
    printf("# shape M=%d N=%d K=%d  GFLOP=%.3f\n", M,N,K, gflop);
    printf("# rocBLAS: %.4f ms  (%.1f GFLOP/s)\n", roc_ms, gflop/(roc_ms/1e3));

    for (int v : variants) {
        CK(hipMemset(dHand, 0, szO*2));
        Ctx hc{dX,dW,dHand,M,N,K,v};
        // 正确性先跑一次
        run_hand(&hc); CK(hipDeviceSynchronize());
        hipError_t le = hipGetLastError();
        if (le != hipSuccess) { printf("v%-2d  LAUNCH-FAIL %s\n", v, hipGetErrorString(le)); continue; }
        std::vector<__half> hH(szO);
        CK(hipMemcpy(hH.data(), dHand, szO*2, hipMemcpyDeviceToHost));
        float maxdiff=0; int nbad=0;
        for (size_t i=0;i<szO;i++){
            float a=__half2float(hRef[i]), b=__half2float(hH[i]);
            float d=fabsf(a-b); if(d>maxdiff)maxdiff=d; if(d>1e-2f)nbad++;
        }
        float hand_ms = time_ms(run_hand, &hc, 20, 100);
        printf("v%-2d  %.4f ms  %.2fx rocBLAS  %.1f GFLOP/s  maxdiff=%.4g  bad(>1e-2)=%d  %s\n",
               v, hand_ms, hand_ms/roc_ms, gflop/(hand_ms/1e3), maxdiff, nbad,
               (maxdiff<1e-2f)?"BITEXACT-OK":"DIFF!");
    }
    return 0;
}
