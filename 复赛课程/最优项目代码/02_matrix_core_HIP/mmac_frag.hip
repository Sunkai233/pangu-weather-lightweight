// 解出 v_mmac builtin 的精确 fragment 布局: 用受控 A/B 跑 builtin, 对比 CPU 参考,
// 自动推断 a_frag/b_frag/c_frag 的 lane→(行,K)/(K,列)/(行,列) 映射.
// 测: A[m][k], B[k][n] 已知(行主序 16x16), C[m][n]=sum_k A*B (CPU真值).
// 用不同假设的 fragment 填法跑, 看哪个 bit-exact.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstring>
#define CK(x) do{hipError_t e=(x);if(e!=hipSuccess){printf("ERR %s\n",hipGetErrorString(e));return 1;}}while(0)
typedef __fp16 half4 __attribute__((ext_vector_type(4)));
typedef float  f32x4 __attribute__((ext_vector_type(4)));

// A,B 行主序 16x16 在 global. kernel 按"假设布局"填 fragment.
// 假设(标准CDNA): a_frag[i]=A[lane%16][ (lane/16)*4+i ]; b_frag[i]=B[ (lane/16)*4+i ][lane%16];
//   c 输出: C[(lane/16)*4+i][lane%16].
__global__ void tile(const __half* A, const __half* B, float* C, int layout){
    int lane=threadIdx.x&63;
    int r=lane%16, g=lane/16;
    half4 a,b;
    for(int i=0;i<4;i++){
        int k=g*4+i;
        // A fragment 取 A[r][k]
        a[i]=(__fp16)__half2float(A[r*16+k]);
        // B fragment: 假设 b_frag[i]=B[k][n=r]  (B 行主序 B[k*16+n])
        if(layout==0) b[i]=(__fp16)__half2float(B[k*16+r]);      // B[k][r]
        else if(layout==1) b[i]=(__fp16)__half2float(B[r*16+k]); // B[r][k] (=B^T)
    }
    f32x4 c={0,0,0,0};
    c=__builtin_amdgcn_mmac_f32_16x16x16f16(a,b,c);
    __syncthreads();
    // 写回假设: C[g*4+i][r]
    for(int i=0;i<4;i++){
        int m=g*4+i, n=r;
        C[m*16+n]=c[i];
    }
}

int main(int argc,char**argv){
    int layout=argc>1?atoi(argv[1]):0;
    __half *dA,*dB; float* dC;
    CK(hipMalloc(&dA,16*16*2));CK(hipMalloc(&dB,16*16*2));CK(hipMalloc(&dC,16*16*4));
    __half hA[256],hB[256]; float hC[256], ref[256];
    // A[m][k]=(m*16+k 的小值), B[k][n]=...
    for(int m=0;m<16;m++)for(int k=0;k<16;k++) hA[m*16+k]=__float2half(0.01f*(m+1)+0.001f*(k+1));
    for(int k=0;k<16;k++)for(int n=0;n<16;n++) hB[k*16+n]=__float2half(0.02f*(n+1)-0.001f*(k+1));
    // CPU 真值: C[m][n]=sum_k A[m][k]*B[k][n]
    for(int m=0;m<16;m++)for(int n=0;n<16;n++){ float s=0;
        for(int k=0;k<16;k++) s+=__half2float(hA[m*16+k])*__half2float(hB[k*16+n]);
        ref[m*16+n]=s; }
    CK(hipMemcpy(dA,hA,512,hipMemcpyHostToDevice));
    CK(hipMemcpy(dB,hB,512,hipMemcpyHostToDevice));
    tile<<<1,64>>>(dA,dB,dC,layout); CK(hipDeviceSynchronize());
    if(hipGetLastError()!=hipSuccess){printf("FAIL launch\n");return 1;}
    CK(hipMemcpy(hC,dC,1024,hipMemcpyDeviceToHost));
    float md=0; int bad=0;
    for(int i=0;i<256;i++){ float d=fabsf(hC[i]-ref[i]); if(d>md)md=d; if(d>1e-2f)bad++; }
    printf("layout=%d maxdiff=%.4g bad=%d/256 %s\n",layout,md,bad,bad==0?"BIT-EXACT!":"diff");
    if(bad){ printf("  C[0][0..3]=%.3f %.3f %.3f %.3f  ref=%.3f %.3f %.3f %.3f\n",
        hC[0],hC[1],hC[2],hC[3],ref[0],ref[1],ref[2],ref[3]); }
    return 0;
}
