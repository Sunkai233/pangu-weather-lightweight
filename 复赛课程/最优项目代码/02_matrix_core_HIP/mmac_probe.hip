// 验证 gfx936 矩阵核 v_mmac_f32_16x16x16_f16: ①峰值算力 ②数值正确性(学操作数布局)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do{hipError_t e=(x);if(e!=hipSuccess){printf("ERR %s\n",hipGetErrorString(e));exit(1);}}while(0)

// ---- 峰值: 每 wave 反复 v_mmac, 无内存 ----
// C[0:3](4 VGPR fp32) += A[0:1](2 VGPR=4 f16) * B[0:1]; 16x16x16.
template<int ITER>
__global__ void peak_mmac(float* out, int iter){
    // 用 4 组累加器并行隐藏延迟
    float4 c0=make_float4(0,0,0,0);
    // A/B 放寄存器(用 half2 装 4 个 fp16/2VGPR)
    __half2 a0=__float2half2_rn(1.001f), a1=__float2half2_rn(1.002f);
    __half2 b0=__float2half2_rn(1.003f), b1=__float2half2_rn(1.004f);
    for(int t=0;t<iter;t++){
        asm volatile(
          "v_mmac_f32_16x16x16_f16 %0, %4, %5, %0\n"
          : "+v"(c0.x),"+v"(c0.y),"+v"(c0.z),"+v"(c0.w)
          : "v"(a0),"v"(a1),"v"(b0),"v"(b1));
    }
    out[blockIdx.x*blockDim.x+threadIdx.x]=c0.x+c0.y+c0.z+c0.w;
}

int main(int argc,char**argv){
    int blocks=80*8, threads=256;
    float* o; CK(hipMalloc(&o,(size_t)blocks*threads*4));
    const int ITER=4096;
    for(int i=0;i<5;i++) peak_mmac<ITER><<<blocks,threads>>>(o,ITER);
    CK(hipDeviceSynchronize());
    hipError_t le=hipGetLastError();
    if(le!=hipSuccess){ printf("LAUNCH FAIL: %s\n",hipGetErrorString(le)); return 1; }
    hipEvent_t a,b;CK(hipEventCreate(&a));CK(hipEventCreate(&b));
    int it=50; CK(hipEventRecord(a));
    for(int i=0;i<it;i++) peak_mmac<ITER><<<blocks,threads>>>(o,ITER);
    CK(hipEventRecord(b));CK(hipEventSynchronize(b));
    float ms;CK(hipEventElapsedTime(&ms,a,b));ms/=it;
    // 每 v_mmac = 16x16x16 MAC = 16*16*16*2 = 8192 flop
    double cnt=(double)blocks*threads/64.0*ITER;  // 每 wave 一条 mmac/iter(64线程协作)
    // 注意: v_mmac 是 wave 级指令, 64 线程一起算一个 16x16x16. 所以指令数 = blocks*threads/64*ITER
    double flop=cnt*8192.0;
    printf("v_mmac peak: %.4f ms  %.1f TFLOP/s  (每wave每iter一条mmac)\n", ms, flop/(ms/1e3)/1e12);
    printf("  对比: v_pk_fma 峰值 30.4 TFLOP/s; rocBLAS recovery 95-101 TFLOP/s\n");
    return 0;
}
