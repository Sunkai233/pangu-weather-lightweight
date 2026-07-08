// 一次性解出 v_mmac fragment 完整布局: 用"独热"探针锚定 a_frag/b_frag/c_frag 各槽对应元素.
// 方法: 每次只让 A 的一个 fragment 槽(lane L, slot i)=1 其余0, B=全1 → C 的非零位置揭示
//   "a_frag[L][i] 参与了哪个输出行(对所有列)" → 反推 a_frag[L][i]=A[?][?].
// 但更直接: 我们要的是"给定 A[m][k] 行主序, 每 lane 该取哪个 (m,k) 填 a_frag[i]".
// 用受控: A[m][k]= m*16+k (唯一编码), B=单位阵 → C[m][n]=A[m][n]. 读 C(用已知C布局) → 得 A 的逆映射.
// 先用 C 布局表(对角{0,17,34,51}步长17)解出 (lane,acc)->(m,n), 再用 A=I/B=val 解 B, A=val/B=I 解 A.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
typedef __fp16 half4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

// 通用: a_frag/b_frag 按 (lane,i) 直接从 global 16x16 行主序取 — 由 amode/bmode 选索引公式.
//   amode/bmode: 0=A[lane%16][(lane/16)*4+i], 1=A[(lane/16)*4+i][lane%16], 等
__global__ void k(const __half* A,const __half* B,float* Craw,int amode,int bmode){
    int lane=threadIdx.x&63,r=lane%16,g=lane/16;
    half4 a,b;
    for(int i=0;i<4;i++){
        int am,ak,bm,bk;
        if(amode==0){am=r; ak=g*4+i;} else {am=g*4+i; ak=r;}
        if(bmode==0){bk=r; bm=g*4+i;} else {bk=g*4+i; bm=r;}
        a[i]=(__fp16)__half2float(A[am*16+ak]);
        b[i]=(__fp16)__half2float(B[bk*16+bm]);  // B 行主序 B[bk][bm]
    }
    f32x4 c={0,0,0,0}; c=__builtin_amdgcn_mmac_f32_16x16x16f16(a,b,c);
    for(int i=0;i<4;i++) Craw[lane*4+i]=c[i];  // raw dump (lane,acc)
}

int main(int argc,char**argv){
    int amode=argc>1?atoi(argv[1]):0, bmode=argc>2?atoi(argv[2]):0;
    __half *dA,*dB; float* dC; hipMalloc(&dA,512);hipMalloc(&dB,512);hipMalloc(&dC,1024);
    __half hA[256],hB[256]; float hC[256],ref[256];
    // A[m][k]=m*16+k+1 (唯一), B=I
    for(int m=0;m<16;m++)for(int kk=0;kk<16;kk++) hA[m*16+kk]=__float2half((float)(m*16+kk+1)*0.01f);
    for(int kk=0;kk<16;kk++)for(int n=0;n<16;n++) hB[kk*16+n]=__float2half(kk==n?1.f:0.f);
    for(int m=0;m<16;m++)for(int n=0;n<16;n++){float s=0;for(int kk=0;kk<16;kk++)s+=__half2float(hA[m*16+kk])*__half2float(hB[kk*16+n]);ref[m*16+n]=s;}
    hipMemcpy(dA,hA,512,hipMemcpyHostToDevice);hipMemcpy(dB,hB,512,hipMemcpyHostToDevice);
    k<<<1,64>>>(dA,dB,dC,amode,bmode);hipDeviceSynchronize();
    if(hipGetLastError()!=hipSuccess){printf("FAIL\n");return 1;}
    hipMemcpy(hC,dC,1024,hipMemcpyDeviceToHost);
    // raw C[(lane,acc)] = ref[m][n] (A=I... 这里B=I → C=A). 解每 (lane,acc)= A[m][n] 的哪个 → m*16+n
    printf("amode=%d bmode=%d  A=val,B=I → C应=A. 解(lane,acc)->A[m][n]:\n",amode,bmode);
    int nz=0;
    for(int l=0;l<64;l++)for(int a=0;a<4;a++){
        float v=hC[l*4+a]/0.01f; int code=(int)roundf(v); // = m*16+n+1
        if(code>=1 && code<=256){ int idx=code-1,m=idx/16,n=idx%16;
            if(l<6) printf("  lane%d acc%d -> A[%d][%d] (val=%.2f)\n",l,a,m,n,hC[l*4+a]); nz++; }
    }
    printf("  有效解码 %d/256\n",nz);
    return 0;
}
