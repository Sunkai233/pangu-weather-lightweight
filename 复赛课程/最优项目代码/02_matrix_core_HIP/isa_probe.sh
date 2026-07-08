#!/bin/bash
# 探测 gfx936 支持哪些矩阵/点积指令(决定 rocBLAS 95 TFLOP/s 是否来自矩阵核)
cd /tmp
probe(){
  printf '#include <hip/hip_runtime.h>\n__global__ void k(float* o){ float acc=0; asm volatile("%s\\n" : "+v"(acc)); o[0]=acc; }\nint main(){return 0;}\n' "$1" > /tmp/pi.hip
  R=$(/opt/dtk/bin/hipcc -O3 --offload-arch=gfx936 -std=c++17 -c /tmp/pi.hip -o /tmp/pi.o 2>&1 | grep -iE "error:|invalid operand|not a recognized|unexpected" | head -1)
  if [ -z "$R" ]; then echo "  OK    : $1"; else echo "  FAIL  : $1"; fi
}
echo "=== gfx936 指令支持(hipcc inline asm) ==="
probe 'v_pk_fma_f16 v0, v1, v2, v3'
probe 'v_dot2_f32_f16 v0, v1, v2, v3'
probe 'v_dot4_i32_i8 v0, v1, v2, v3'
probe 'v_dot8_i32_i4 v0, v1, v2, v3'
probe 'v_mfma_f32_16x16x16f16 v[0:3], v[4:5], v[6:7], v[0:3]'
probe 'v_mfma_f32_32x32x8f16 v[0:15], v[16:17], v[18:19], v[0:15]'
probe 'v_dot2c_f32_f16 v0, v1, v2'
