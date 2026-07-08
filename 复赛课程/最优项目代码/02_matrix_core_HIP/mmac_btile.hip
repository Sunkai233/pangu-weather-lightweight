// 解 B fragment 布局: B=单位阵, A=已知 → C 应=A. 试不同 b_frag 填法.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#define CK(x) do{hipError_t e=(x);if(e!=hipSuccess){printf("ERR\n");return 1;}}while(0)
typedef __fp16 half4 __attribute__((ext_vector_type(4)));
typedef float  f32x4 __attribute__((ext_vector_type(4)));

// A 行主序 A[m][k], B 行主序 B[k][n]. a_frag/C 已知正确(A[r][g*4+i], C[g*4+i][r]).
// 试 b_frag[i] 的多种映射 layout.
__global__ void tile(const __half* A,const __half* B,float* C,int layout){
    int lane=threadIdx.x&63, r=lane%16, g=lane/16;
    half4 a,b;
    for(int i=0;i<4;i++){
        int k=g*4+i;
        a[i]=(__fp16)__half2float(A[r*16+k]);
        int bk,bn;
        switch(layout){
            case 0: bk=g*4+i; bn=r; b[i]=(__fp16)__half2float(B[bk*16+bn]); break; // B[k=g4+i][n=r]
            case 1: bk=r; bn=g*4+i; b[i]=(__fp16)__half2float(B[bk*16+bn]); break; // B[r][g4+i]
            case 2: bn=g*4+i; bk=r; b[i]=(__fp16)__half2float(B[bk*16+bn]); break; // B[n=g4+i? ]同1
            case 3: // b_frag = B[k][n], k=lane%16? n=...? 试 B 转置读 W[n][k]: 我们传入的是W行主序
                    b[i]=(__fp16)__half2float(B[r*16 + (g*4+i)]); break;
        }
    }
    f32x4 c={0,0,0,0};
    c=__builtin_amdgcn_mmac_f32_16x16x16f16(a,b,c);
    __syncthreads();
    for(int i=0;i<4;i++) C[(g*4+i)*16 + r]=c[i];
}
int main(int argc,char**argv){
    int layout=argc>1?atoi(argv[1]):0;
    __half *dA,*dB; float* dC;
    hipMalloc(&dA,512);hipMalloc(&dB,512);hipMalloc(&dC,1024);
    __half hA[256],hB[256]; float hC[256],ref[256];
    for(int m=0;m<16;m++)for(int k=0;k<16;k++) hA[m*16+k]=__float2half(0.05f*(m+1)+0.003f*(k+1));
    for(int k=0;k<16;k++)for(int n=0;n<16;n++) hB[k*16+n]=__float2half(k==n?1.0f:0.0f); // B=I
    for(int m=0;m<16;m++)for(int n=0;n<16;n++){ float s=0;
        for(int k=0;k<16;k++) s+=__half2float(hA[m*16+k])*__half2float(hB[k*16+n]);
        ref[m*16+n]=s; }
    hipMemcpy(dA,hA,512,hipMemcpyHostToDevice); hipMemcpy(dB,hB,512,hipMemcpyHostToDevice);
    tile<<<1,64>>>(dA,dB,dC,layout); hipDeviceSynchronize();
    if(hipGetLastError()!=hipSuccess){printf("FAIL\n");return 1;}
    hipMemcpy(hC,dC,1024,hipMemcpyDeviceToHost);
    float md=0;int bad=0; for(int i=0;i<256;i++){float d=fabsf(hC[i]-ref[i]);if(d>md)md=d;if(d>1e-2f)bad++;}
    printf("B=I layout=%d maxdiff=%.4g bad=%d/256 %s\n",layout,md,bad,bad==0?"BIT-EXACT(=A)!":"diff");
    return 0;
}
