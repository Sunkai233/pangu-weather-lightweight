# -*- coding: utf-8 -*-
"""GCN5/Vega (gfx9xx) 指令分类解码器 — 统计 kernel .text 用了哪些指令类,
重点识别 VOP3P(packed: v_pk_fma_f16/v_dot/v_mfma)、VALU、LDS、访存, 找 rocBLAS 95 TFLOP/s 秘密。
不做完整反汇编, 只识别指令类别和 VOP3P 子opcode(够回答'热循环发什么指令')。"""
import sys, struct

blob = open(sys.argv[1], 'rb').read()
words = [struct.unpack('<I', blob[i:i+4])[0] for i in range(0, len(blob)-3, 4)]

# GCN5 VOP3P opcode 表(bits [22:16] of first word)
VOP3P = {
    0x00:'v_pk_mad_i16', 0x01:'v_pk_mul_lo_u16', 0x02:'v_pk_add_i16', 0x03:'v_pk_sub_i16',
    0x04:'v_pk_lshlrev_b16',0x05:'v_pk_lshrrev_b16',0x06:'v_pk_ashrrev_i16',
    0x07:'v_pk_max_i16',0x08:'v_pk_min_i16',0x09:'v_pk_mad_u16',0x0a:'v_pk_add_u16',
    0x0b:'v_pk_sub_u16',0x0c:'v_pk_max_u16',0x0d:'v_pk_min_u16',
    0x0e:'v_pk_fma_f16',   # ★ packed fp16 FMA
    0x0f:'v_pk_add_f16',0x10:'v_pk_mul_f16',0x11:'v_pk_min_f16',0x12:'v_pk_max_f16',
    0x20:'v_dot2_f32_f16',0x21:'v_dot2_i32_i16',0x22:'v_dot2_u32_u16',
    0x23:'v_dot4_i32_i8',0x24:'v_dot4_u32_u8',0x25:'v_dot8_i32_i4',0x26:'v_dot8_u32_u4',
    # MFMA(MAI) 子码(若有)
    0x40:'v_mfma_f32_32x32x1f32',0x42:'v_mfma_f32_16x16x1f32',
    0x68:'v_mfma_f32_32x32x4f16',0x6a:'v_mfma_f32_16x16x4f16',
    0x6c:'v_mfma_f32_4x4x4f16',0x6e:'v_mfma_f32_32x32x8f16',0x6f:'v_mfma_f32_16x16x16f16',
}

def cls(w):
    # 判断指令编码类别(GCN5)
    if (w & 0x80000000)==0:
        # 0xxxxxxx: VOP2 / VOPC / VOP1 / SOP? 细分
        top2 = (w>>30)&3
        if top2==0:  # SOP2 (10) etc handled below; 0b0xxx -> VOP? actually:
            pass
        # GCN: bit31=0 -> VOP1/VOP2/VOPC/VINTRP; need finer
    # 用高位模式粗分类
    if (w & 0xFF800000)==0xCC000000:  # VOP3P (110011001)
        op=(w>>16)&0x7F
        return ('VOP3P', VOP3P.get(op, f'vop3p_op{op:#x}'))
    if (w & 0xFC000000)==0xD0000000:  # VOP3 / VOP3B (1101 00)
        return ('VOP3','vop3')
    if (w & 0xFC000000)==0xD8000000:  # DS (LDS) 110110
        return ('LDS','ds')
    if (w & 0xFC000000)==0xDC000000:  # FLAT/global 110111
        return ('MEM','flat/global')
    if (w & 0xFC000000)==0xE0000000:  # MUBUF 111000
        return ('MEM','mubuf')
    if (w & 0xFC000000)==0xC0000000:  # SMEM 110000
        return ('SMEM','smem')
    if (w & 0x80000000)==0 and (w>>25)==0b0111110:  # VOP1 0111111? approx
        return ('VOP1','vop1')
    if (w & 0xC0000000)==0x00000000:  # VOP2 (0xxxxxxx, bit31=0) broad
        return ('VOP2','vop2')
    if (w & 0xFF800000)==0xBF800000:  # SOPP (s_waitcnt/s_branch etc) 10111111 1
        return ('SOPP','sopp')
    if (w & 0xFF800000)==0xBE800000 or (w&0xFF800000)==0xBE000000:
        return ('SOP1','sop1')
    if (w & 0xC0000000)==0x80000000:  # SOP2 10xxxxx
        return ('SOP2','sop2')
    return ('?', f'{w:08x}')

from collections import Counter
clsc=Counter(); opc=Counter()
i=0
# VOP3/VOP3P/SMEM/DS/MEM 是 64bit(2 dword); VOP1/2/SOP* 多为 32bit。简化:都按可能含第二字判断
while i < len(words):
    w=words[i]
    c,name=cls(w)
    clsc[c]+=1
    if c=='VOP3P': opc[name]+=1
    # 64-bit 指令跳 2 字(VOP3P/VOP3/SMEM/DS/MEM 都有 literal/第二字)
    if c in ('VOP3P','VOP3','SMEM','LDS','MEM'):
        i+=2
    else:
        i+=1

print("=== 指令类别统计 ===")
for c,n in clsc.most_common():
    print(f"  {c:8s}: {n}")
print("=== VOP3P(packed/matrix/dot) 子opcode ===")
for o,n in opc.most_common():
    print(f"  {o:24s}: {n}")
print("\n★ 关键: 若 v_mfma_* 出现 → 有矩阵核; 若只 v_pk_fma_f16 → 纯packed; v_dot* → 点积加速")
