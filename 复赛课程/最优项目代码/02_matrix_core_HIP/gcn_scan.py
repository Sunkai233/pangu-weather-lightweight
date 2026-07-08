# -*- coding: utf-8 -*-
"""位置无关扫描 kernel .text: 在每个 dword 边界检测特征指令编码,
统计 v_pk_fma_f16/v_dot*/v_mfma*/v_fma_f32/v_mac_f32/v_fmac_f32 等关键 FMA 指令出现频次。
GCN5 指令都 4 字节对齐, 所以扫每个 dword 位置即可覆盖(会有少量误命中, 但频次趋势可靠)。"""
import sys, struct
blob=open(sys.argv[1],'rb').read()
W=[struct.unpack('<I',blob[i:i+4])[0] for i in range(0,len(blob)-3,4)]

def is_vop3p(w): return (w & 0xFF800000)==0xCC000000
def vop3p_op(w): return (w>>16)&0x7F
def is_vop3(w):  return (w & 0xFC000000)==0xD0000000  # VOP3 1101 00
def vop3_op(w):  return (w>>16)&0x3FF

# VOP3 关键 opcode(GCN5): v_fma_f32=0x1ab? 实际 VOP3 op 编码; v_mad_f32, v_fma_f32
# VOP2(bit31=0, [30:25]=opcode for VOP2): v_mac_f32=0x16, v_fmac_f32=0x1c (gfx9)
def is_vop2(w): return (w>>31)==0
def vop2_op(w): return (w>>25)&0x3F

from collections import Counter
v3p=Counter(); v3=Counter(); v2=Counter()
n_vop3p_fma=0; n_dot=0; n_mfma=0
for w in W:
    if is_vop3p(w):
        op=vop3p_op(w); v3p[op]+=1
        if op==0x0e: n_vop3p_fma+=1
        if 0x20<=op<=0x26: n_dot+=1
        if op>=0x40: n_mfma+=1
    elif is_vop3(w):
        v3[vop3_op(w)]+=1
    elif is_vop2(w):
        v2[vop2_op(w)]+=1

print("=== VOP3P 子opcode 频次(packed/dot/mfma) ===")
NAMES={0x0e:'v_pk_fma_f16',0x0f:'v_pk_add_f16',0x10:'v_pk_mul_f16',
       0x20:'v_dot2_f32_f16',0x23:'v_dot4_i32_i8',0x25:'v_dot8_i32_i4'}
for op,c in v3p.most_common(15):
    print(f"  op={op:#04x} {NAMES.get(op,''):16s}: {c}")
print(f"  → v_pk_fma_f16={n_vop3p_fma}  v_dot*={n_dot}  v_mfma*={n_mfma}")
print("=== VOP3 top opcodes ===")
V3N={0x1ab:'v_fma_f32',0x1a3:'v_mad_f32',0x140:'v_add_f32',0x141:'v_sub_f32',0x143:'v_mul_f32',0x1ff:'v_fma_f16',0x206:'v_fma_mix?'}
for op,c in v3.most_common(15):
    print(f"  op={op:#05x} {V3N.get(op,''):14s}: {c}")
print("=== VOP2 top opcodes ===")
V2N={0x16:'v_mac_f32',0x1c:'v_fmac_f32',0x03:'v_add_f32',0x08:'v_mul_f32',0x01:'v_add(legacy)'}
for op,c in v2.most_common(15):
    print(f"  op={op:#04x} {V2N.get(op,''):14s}: {c}")
