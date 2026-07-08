// 暴力搜 v_mmac 的 (a_frag, b_frag, c_out) 布局组合, 对 CPU 参考找 bit-exact 组合.
// 单 16x16x16 tile. A,B 随机 fp16. C_ref[m][n]=sum_k A[m][k]*B[k][n].
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
typedef __fp16 half4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

// A,B 行主序 16x16 (A[m][k], B[k][n]). 各 fill 模式由 al,bl 选; 输出 raw dump.
__global__ void k(const __half* A,const __half* B,float* Craw,int al,int bl){
  int lane=threadIdx.x&63, r=lane&15, g=lane>>4;
  half4 a,bb;
  for(int i=0;i<4;i++){
    int am,ak,bk,bn;
    // a fill 模式
    if(al==0){am=r; ak=g*4+i;}      // A[r][g4+i]
    else     {am=g*4+i; ak=r;}      // A[g4+i][r]
    // b fill 模式
    if(bl==0){bk=r; bn=g*4+i;}      // B[r][g4+i]
    else if(bl==1){bk=g*4+i; bn=r;} // B[g4+i][r]
    else if(bl==2){bk=g*4+i; bn=r;} // (同1, 占位)
    a[i]=(__fp16)__half2float(A[am*16+ak]);
    bb[i]=(__fp16)__half2float(B[bk*16+bn]);
  }
  f32x4 c={0,0,0,0}; c=__builtin_amdgcn_mmac_f32_16x16x16f16(a,bb,c);
  for(int i=0;i<4;i++) Craw[lane*4+i]=c[i];
}
int main(){
  __half *dA,*dB; float* dC; hipMalloc(&dA,512);hipMalloc(&dB,512);hipMalloc(&dC,1024);
  __half hA[256],hB[256]; float hC[256],ref[256];
  srand(7);
  for(int i=0;i<256;i++){ hA[i]=__float2half((rand()%100-50)*0.01f); hB[i]=__float2half((rand()%100-50)*0.01f);}
  for(int m=0;m<16;m++)for(int n=0;n<16;n++){float s=0;for(int kk=0;kk<16;kk++)s+=__half2float(hA[m*16+kk])*__half2float(hB[kk*16+n]);ref[m*16+n]=s;}
  hipMemcpy(dA,hA,512,hipMemcpyHostToDevice);hipMemcpy(dB,hB,512,hipMemcpyHostToDevice);
  // 对每个 (al,bl) 组合 + 每个 c-out 映射, 检查是否能匹配 ref
  for(int al=0;al<2;al++)for(int bl=0;bl<2;bl++){
    k<<<1,64>>>(dA,dB,dC,al,bl);hipDeviceSynchronize();
    hipMemcpy(hC,dC,1024,hipMemcpyDeviceToHost);
    // 试 4 种 c-out 映射: (lane,acc)->(m,n)
    // co0: m=r, n=g+i*4 ; co1: m=r,n=g*4+i ; co2: m=g+i*4,n=r ; co3: m=g*4+i,n=r
    for(int co=0;co<4;co++){
      float md=0;int bad=0;
      for(int lane=0;lane<64;lane++){int r=lane&15,g=lane>>4;
        for(int i=0;i<4;i++){int m,n;
          if(co==0){m=r;n=g+i*4;} else if(co==1){m=r;n=g*4+i;}
          else if(co==2){m=g+i*4;n=r;} else {m=g*4+i;n=r;}
          float v=hC[lane*4+i]; float d=fabsf(v-ref[m*16+n]); if(d>md)md=d; if(d>1e-2f)bad++;
        }}
      if(bad<20) printf("al=%d bl=%d co=%d: maxdiff=%.4g bad=%d %s\n",al,bl,co,md,bad,bad==0?"<<< BIT-EXACT!":"");
    }
  }
  return 0;
}
