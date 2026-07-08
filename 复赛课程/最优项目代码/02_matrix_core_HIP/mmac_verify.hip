// v_mmac_f32_16x16x16_f16 数值正确性 + 布局解码.
// 测1: A=B=全1 → C 每元素 = sum_{k=0..15} 1*1 = 16.0 (确认是 16-deep dot).
// 测2: 用结构化输入推 lane→元素映射.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do{hipError_t e=(x);if(e!=hipSuccess){printf("ERR %s\n",hipGetErrorString(e));exit(1);}}while(0)

// 每 lane 持 A 的 4 个 fp16(2 VGPR=v4,v5) 和 B 的 4 个(v6,v7). C 输出 4 fp32(v0..v3).
// kernel 入参: 每 lane 的 a 打包(2×u32), b 打包(2×u32). 输出 4 fp32/lane.
__global__ void mmac_test(const unsigned* aPacked, const unsigned* bPacked, float* cOut){
    int lane = threadIdx.x & 63;
    unsigned a0=aPacked[lane*2+0], a1=aPacked[lane*2+1];
    unsigned b0=bPacked[lane*2+0], b1=bPacked[lane*2+1];
    float c0,c1,c2,c3;
    asm volatile(
      "v_mov_b32 v4, %4\n v_mov_b32 v5, %5\n"
      "v_mov_b32 v6, %6\n v_mov_b32 v7, %7\n"
      "v_mov_b32 v0, 0\n v_mov_b32 v1, 0\n v_mov_b32 v2, 0\n v_mov_b32 v3, 0\n"
      "v_mmac_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]\n"
      "s_nop 8\n"
      "v_mov_b32 %0, v0\n v_mov_b32 %1, v1\n v_mov_b32 %2, v2\n v_mov_b32 %3, v3\n"
      : "=v"(c0),"=v"(c1),"=v"(c2),"=v"(c3)
      : "v"(a0),"v"(a1),"v"(b0),"v"(b1)
      : "v0","v1","v2","v3","v4","v5","v6","v7");
    cOut[lane*4+0]=c0; cOut[lane*4+1]=c1; cOut[lane*4+2]=c2; cOut[lane*4+3]=c3;
}

int main(int argc,char**argv){
    int mode = argc>1?atoi(argv[1]):0;
    unsigned *dA,*dB; float* dC;
    CK(hipMalloc(&dA,64*2*4)); CK(hipMalloc(&dB,64*2*4)); CK(hipMalloc(&dC,64*4*4));
    unsigned hA[128], hB[128]; float hC[256];
    // 主机端 fp32→fp16 bits(IEEE half), 避免 device-only intrinsic
    auto f2h=[](float f)->unsigned short{
        unsigned x; memcpy(&x,&f,4);
        unsigned sign=(x>>16)&0x8000; int exp=((x>>23)&0xff)-127+15; unsigned man=(x>>13)&0x3ff;
        if(exp<=0) return (unsigned short)sign;
        if(exp>=31) return (unsigned short)(sign|0x7c00);
        return (unsigned short)(sign|(exp<<10)|man);
    };
    auto pk=[&](float lo,float hi){ return (unsigned)f2h(lo) | ((unsigned)f2h(hi)<<16); };
    if(mode==0){
        // 全1
        for(int i=0;i<128;i++){ hA[i]=pk(1.0f,1.0f); hB[i]=pk(1.0f,1.0f);}
    } else if(mode==1){
        // A 全1, B: lane0 的第一个=2 其余1 (探 C 哪些元素受影响)
        for(int i=0;i<128;i++){ hA[i]=pk(1.0f,1.0f); hB[i]=pk(1.0f,1.0f);}
        hB[0]=pk(2.0f,1.0f);
    }
    CK(hipMemcpy(dA,hA,128*4,hipMemcpyHostToDevice));
    CK(hipMemcpy(dB,hB,128*4,hipMemcpyHostToDevice));
    mmac_test<<<1,64>>>(dA,dB,dC);
    CK(hipDeviceSynchronize());
    hipError_t le=hipGetLastError();
    if(le!=hipSuccess){ printf("LAUNCH FAIL %s\n",hipGetErrorString(le)); return 1;}
    CK(hipMemcpy(hC,dC,256*4,hipMemcpyDeviceToHost));
    printf("mode=%d  C[lane][0..3] (前8 lane):\n",mode);
    for(int l=0;l<8;l++) printf("  lane%2d: %.1f %.1f %.1f %.1f\n",l,hC[l*4],hC[l*4+1],hC[l*4+2],hC[l*4+3]);
    // 统计非全16的
    if(mode==0){ int bad=0; for(int i=0;i<256;i++) if(hC[i]!=16.0f)bad++;
        printf("全1测试: 期望全 16.0, 不符=%d/256\n",bad);}
    return 0;
}
