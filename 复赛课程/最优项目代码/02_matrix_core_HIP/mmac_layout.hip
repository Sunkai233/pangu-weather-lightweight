// 完整解码 v_mmac_f32_16x16x16_f16 的 lane→元素布局(干净版, s_nop 充足).
// C[m][n] = sum_k A[m][k]*B[k][n], 16x16x16. wave64 协作.
// 喂"单热点"输入: 只设 A 或 B 的某一(lane,fp16槽)=特殊值, 看 C 哪些(lane,acc)变 → 反推映射.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do{hipError_t e=(x);if(e!=hipSuccess){printf("ERR %s\n",hipGetErrorString(e));exit(1);}}while(0)

__global__ void mmac_k(const unsigned* aP, const unsigned* bP, float* cO){
    int lane=threadIdx.x&63;
    unsigned a0=aP[lane*2+0],a1=aP[lane*2+1],b0=bP[lane*2+0],b1=bP[lane*2+1];
    float c0,c1,c2,c3;
    asm volatile(
      "v_mov_b32 v8, %4\n v_mov_b32 v9, %5\n v_mov_b32 v10, %6\n v_mov_b32 v11, %7\n"
      "v_mov_b32 v12, 0\n v_mov_b32 v13, 0\n v_mov_b32 v14, 0\n v_mov_b32 v15, 0\n"
      "v_mmac_f32_16x16x16_f16 v[12:15], v[8:9], v[10:11], v[12:15]\n"
      "s_nop 15\n s_nop 15\n s_nop 15\n s_nop 15\n"
      "v_mov_b32 %0, v12\n v_mov_b32 %1, v13\n v_mov_b32 %2, v14\n v_mov_b32 %3, v15\n"
      : "=v"(c0),"=v"(c1),"=v"(c2),"=v"(c3)
      : "v"(a0),"v"(a1),"v"(b0),"v"(b1)
      : "v8","v9","v10","v11","v12","v13","v14","v15");
    cO[lane*4+0]=c0; cO[lane*4+1]=c1; cO[lane*4+2]=c2; cO[lane*4+3]=c3;
}

static unsigned short f2h(float f){ unsigned x; memcpy(&x,&f,4);
  unsigned sign=(x>>16)&0x8000; int exp=((x>>23)&0xff)-127+15; unsigned man=(x>>13)&0x3ff;
  if(exp<=0)return sign; if(exp>=31)return sign|0x7c00; return sign|(exp<<10)|man; }
static unsigned pk(float lo,float hi){ return (unsigned)f2h(lo)|((unsigned)f2h(hi)<<16); }

int main(int argc,char**argv){
    // argv: which(A/B) lane slot  → 设该热点=2, 其余=1, 打印所有受影响 C(lane,acc)
    char which = argc>1?argv[1][0]:'A';
    int hlane = argc>2?atoi(argv[2]):0;
    int hslot = argc>3?atoi(argv[3]):0;   // 0..3 (lane 的 4 个 fp16: a0.lo,a0.hi,a1.lo,a1.hi)
    unsigned *dA,*dB; float* dC;
    CK(hipMalloc(&dA,64*2*4));CK(hipMalloc(&dB,64*2*4));CK(hipMalloc(&dC,64*4*4));
    unsigned hA[128],hB[128]; float hC[256];
    for(int i=0;i<128;i++){hA[i]=pk(1,1);hB[i]=pk(1,1);}
    // 设热点: lane hlane 的 slot hslot = 2.0
    unsigned* tgt = (which=='A')?hA:hB;
    float lo=1,hi=1; if(hslot%2==0)lo=2; else hi=2;
    int vidx=hlane*2 + hslot/2;
    tgt[vidx]=pk(lo,hi);
    CK(hipMemcpy(dA,hA,512,hipMemcpyHostToDevice));
    CK(hipMemcpy(dB,hB,512,hipMemcpyHostToDevice));
    mmac_k<<<1,64>>>(dA,dB,dC); CK(hipDeviceSynchronize());
    if(hipGetLastError()!=hipSuccess){printf("FAIL\n");return 1;}
    CK(hipMemcpy(hC,dC,1024,hipMemcpyDeviceToHost));
    printf("热点 %c lane%d slot%d =2 → 受影响的 C(基线16.0):\n",which,hlane,hslot);
    int n=0;
    for(int l=0;l<64;l++) for(int a=0;a<4;a++){
        if(hC[l*4+a]>16.5f||hC[l*4+a]<15.5f){ if(n<20)printf("  C[lane%d][acc%d]=%.1f\n",l,a,hC[l*4+a]); n++; }
    }
    printf("  共 %d 个受影响\n",n);
    return 0;
}
