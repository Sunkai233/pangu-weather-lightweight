# onescience 网络库参考(★决赛闭卷唯一能读的"网络写法"来源)

> 决赛闭卷,别的看不到,但 **onescience 装在容器里,源码能读**:
> `ls /usr/local/lib/python3.11/dist-packages/onescience/modules/` → `less xxx.py`。
> 本目录 `baseline/onescience源码/modules/`(257 个 .py)已把它整包存下来对照。

## 一、工厂模式:`OneXxx(style="名字", ...)`

onescience 用**工厂 + 注册表**:每类组件有个 `OneXxx` 工厂,`style="具体名"` 映射到实现类。
所以 Pangu 里写的 `OneEmbedding(style="PanguEmbedding", ...)` 实际调的是 `embedding/panguembedding.py:PanguEmbedding`。

**读法**:先看 `xxx/onexxx.py` 的注册字典(style→模块:类),再去对应文件读实现。例:
```python
# attention/oneattention.py 的注册表(节选)
"EarthAttention3D": ("...attention.earthattention3d", "EarthAttention3D"),
"WindowAttention":  ("...attention.windowattention", "WindowAttention"),
"MultiHeadAttention": (...), "LinearAttention": (...), "FlashAttention": (...),
```

## 二、★ Pangu 五大积木(我们学生就是借这些,只自己换了注意力)

| 积木 | 文件 | 核心写法(可默写参考) |
|---|---|---|
| **PatchEmbedding** | `embedding/panguembedding.py` | `nn.ZeroPad3d`补齐到patch整数倍 + `nn.Conv3d(V,embed,kernel=patch,stride=patch)`(stride=patch 就是切patch+投影);2D输入自动 unsqueeze 成3D |
| **PatchRecovery** | `recovery/pangupatchrecovery.py` | `nn.ConvTranspose3d(in,out,kernel=patch,stride=patch)` 上采样回原分辨率 + **center-crop** 去掉多余padding。是embedding的逆 |
| **EarthAttention3D** | `attention/earthattention3d.py` | 窗口内多头:`qkv=Linear(dim,3dim)` →reshape/permute→ `q*scale; attn=q@k.T; attn+=地球位置偏置; softmax; x=attn@v; proj`。位置偏置用 `earth_position_index` 查表 |
| **Fuser(主干块)** | `fuser/pangufuser.py` | 堆 `depth` 个 `EarthTransformer3DBlock`,`shift_size`偶数层(0,0,0)/奇数层None(**Swin 移窗交替**),在 `transformer/onetransformer.py` 注册 |
| **Down/UpSample** | `sample/pangudownsample.py`/`panguupsample.py` | 三维 patch merge/expand,改分辨率(1/2)与通道(×2) |

> 教师 `models/pangu/pangu.py` 就是把这五样按 U 形(embed→layer1→down→layer2/3→up→layer4→recovery + skip)串起来。**读它 = 读一个完整气象网络怎么搭。**

## 三、可用 style 全清单(闭卷想换组件直接抄)

**注意力(`OneAttention(style=)`,11 种)** —— 想手写/换注意力抄这里:
`EarthAttention2D/3D`、`WindowAttention`、`MultiHeadAttention`、`SelfAttention`、`FlashAttention`、`LinearAttention`(线性注意力省显存)、`FactAttention2D/3D`、`NystromAttention`、`FeatureGrouping/UngroupingAttention`。

**Transformer 块(`OneTransformer(style=)`)**:`EarthTransformer2D/3DBlock`、`SwinTransformerBlock`、`PreLNTransformerBlock`、`NeuralSpectralBlock1/2/3D`、`FuxiTransformer`、`XiHeTransformer3D` 等。

**Embedding**:`PanguEmbedding`、`FuxiEmbedding`、`FourCastNetEmbedding`、`XiheEmbedding`、`FourierPosEmbedding`、`GraphCastEncoder/DecoderEmbedder`。
**Recovery**:`PanguPatchRecovery`、`XihePatchRecovery`。
**Sample(下/上采样)**:`PanguDownSample/UpSample`、`FuxiDownSample/UpSample`、`XiheUpSample`、`SpatialGraphDownsample/Upsample`。
**Fuser**:`PanguFuser`、`FengWuFuser`、`FourCastNetFuser`、`XiheLocal/GlobalSIEFuser`。
**Encoder/Decoder**:`UNetEncoder1/2/3D`、`UNetHead1/2/3D`、`FengWuEncoder`、`GraphViTEncoder`、`MeshGraphEncoder`。
**谱方法(`OneFourier(style=)`)**:`FNOSpectralConv1/2/3d`、`FFNOSpectralConv*`、`GeoSpectralConv*`、`WaveletFourierKernel*`、`MultiWaveletTransform*`(想做 FNO/AFNO 频域全局混合抄这里)。
**MLP**:`StandardMLP/SimpleMLP/DeepResMLP/LightweightMLP`、`XiheMlp`。
**其它整模型参考**:`afno/`(FourCastNet AFNO)、`evolution/nowcastnet.py`、`fc/`、各家 `fengwu/fourcastnet/fuxi/xihe/graphcast` 的 encoder/decoder/fuser/embedding。

## 四、我们的学生 vs Pangu(借了什么/改了什么)

`baseline/官方源码/maxvit3d_student.py`:
- **借用(直接 import onescience)**:`OneEmbedding`(=PanguEmbedding)、`OneRecovery`(=PanguPatchRecovery)、`OneSample`(=PanguDown/UpSample)——I/O 与教师严格一致,可蒸馏、可用官方 result.py 评。
- **自己写替换**:把 Pangu 的 `PanguFuser`(EarthAttention3D 窗口注意力)换成本文件的 **MaxViT 局部block+全局grid 注意力**(带 QK-Norm/RMSNorm/SwiGLU),更省参数、可缩 embed。

所以决赛现场:**embed/recovery/采样照抄 onescience**,**注意力/主干块可以参考 earthattention3d.py 或 windowattention.py 自己写**(或直接 `OneAttention(style="WindowAttention")`)。

## 五、闭卷现场怎么用(实操)
```bash
# 1) 看某类组件有哪些 style
sed -n '1,60p' /usr/local/lib/python3.11/dist-packages/onescience/modules/attention/oneattention.py
# 2) 读某个实现(比如想抄窗口注意力)
less /usr/local/lib/python3.11/dist-packages/onescience/modules/attention/earthattention3d.py
# 3) 直接用工厂(不用自己写)
python -c "from onescience.modules import OneAttention; a=OneAttention(style='WindowAttention', dim=96, num_heads=6); print(a)"
```
> 记忆点:**能 import 就别默写**(embed/recovery/sample/attention 都能 `OneXxx(style=)` 调);真要自己写,`earthattention3d.py`/`panguembedding.py` 是最好的抄写模板。
