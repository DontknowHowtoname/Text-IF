# softlight_blend_probe

柔光混合（W3C Soft Light）用于红外/可见光融合的初步验证实验。作为零参数基线，与 Text-IF text_ratio=T5 的融合结果做同图对比，回答一个问题：不学习的 PS 混合模式能走多远？

## 1. 动机

Photoshop 中"柔光"的经典用法是：**保底图基调，用 blend 层做明暗修饰**——base 层的结构与色调不被替换，blend 层亮处轻微提亮 base、暗处轻微压暗，且修饰强度随 blend 偏离中灰的程度平滑变化（中灰 = 不动）。

这与红外/可见光融合的直觉高度契合：

- **VI 保色彩/结构**：以 vi 为 base，自然保住可见光的彩色纹理基调；
- **IR 提亮热目标**：以 ir 为 blend，红外亮目标（行人等）恰好落在"亮处提亮"的一侧，且 IR 越亮修饰越强。

它全程零参数、逐像素、可微（分段函数在定义域内连续），如果性能尚可，就有资格作为融合网络中的一种**先验混合算子**。本探针用小规模抽样实验检验这一点。

## 2. 方法

### 2.1 W3C 柔光公式

`softlight_blend_probe/probe_softlight.py` 中的 `softlight(base, blend)` 实现 W3C compositing 规范的柔光：

```text
out = b + (2a − 1) · (D(b) − b)

D(b) = ((16b − 12)·b + 4)·b    若 b ≤ 0.25
D(b) = √b                      若 b > 0.25
```

其中 b = base、a = blend，均取值 [0,1]，逐像素计算。两条性质：

- a = 0.5（中灰）时 2a−1 = 0，输出 ≡ base（blend 亮度决定"增/减"方向）；
- a = 1 时 out = D(b)（全亮 blend 下的提亮曲线）；a = 0 时镜像压暗。

### 2.2 不透明度 α

`softlight_blend(base, blend, alpha)` 在混合后做不透明度插值：

```text
out = (1−α)·base + α·softlight(base, blend)
```

α 控制柔光强度，α=0 退化为 base 本身。实验扫 α ∈ {0.6, 0.8, 1.0}。

### 2.3 两种层序语义

`fuse(ir, vi, order, alpha)`：

| order | base | blend | 语义 |
| --- | --- | --- | --- |
| `ir_on_vi` | vi（彩色） | ir（灰度广播到 3 通道） | VI 保基调，IR 做明暗修饰（契合直觉的用法） |
| `vi_on_ir` | ir（灰度复制 3 通道） | vi | IR 保基调，VI 做修饰（对照组） |

### 2.4 与 PS 实际实现的差异

Photoshop 实际采用的是 **Pegtop 近似**（D(b) = b²，全域单一多项式），并非 W3C 分段式。两者在中灰 blend 附近行为一致，主要差别在 base 极暗区（b < 0.25）：W3C 分段曲线在 b = 0.25 处一阶连续（有单测保证），Pegtop 则近似。本实验统一用 W3C 闭式解，复现时注意与 PS 截图的细微差别来源即在此。

## 3. 验证设计

- **抽样**：每数据集从 T5 生成目录（`sweeps/out/text_ratio_T5/gen/{ds}/fused`，按文件名排序）取前 3 张：MSRS 00055D/00091D/00095D + TNO TNO_0023/TNO_0024/TNO_0051，共 6 张。
- **配置网格**：2 层序 × α ∈ {0.6, 0.8, 1.0}，共 6 个柔光配置 + T5 基线。
- **图件**：每图输出 `{name}_grid.png`（6 配置网格，挑参数用）与 `{name}_compare.png`（IR | VI | 柔光最优配置 | T5 四联对比）。
- **指标口径**：复用 `metric/Metric_torch.py` 的 10 个指标（EN/MI/SF/AG/SD/VIF/SSIM/Qabf/Nabf/SCD），灰度转换与 `evaluate_experiment_metrics.py` 一致（PIL convert('L')），且**与 T5 输出同图同尺寸**计算——T5 推理管线会把输入 resize 到 16 的倍数（TNO 381×461 → 368×448），因此 `load_pair_t5` 在算指标前把 ir/vi 双线性对齐到 T5 输出尺寸，否则形状不一致。
- **选优规则**（`select_best_config`）：每个 higher-better 指标（除 Nabf 外全部）在配置间做逐图平均、按名次打分（rank 1 得 N−1 分 … rank N 得 0 分），Nabf 按 lower-better 排序，总分最高的配置胜出。

## 4. 结果与结论

### 4.1 指标均值（6 张抽样图）

```text
method                        EN     MI     SF     AG     SD     VIF    SSIM   Qabf   Nabf   SCD
T5                          6.951  4.611  10.591  3.629  55.396  1.133  0.991  0.674  0.013  1.579
softlight/ir_on_vi/0.6      6.374  4.593  11.123  3.470  64.106  1.026  0.856  0.570  0.017  1.530
softlight/ir_on_vi/0.8      6.326  4.372  11.504  3.536  65.404  0.993  0.785  0.556  0.018  1.534
softlight/ir_on_vi/1.0      5.976  3.935  11.909  3.604  66.773  0.965  0.723  0.552  0.019  1.537
softlight/vi_on_ir/0.6      5.727  3.887   5.186  1.736  22.441  0.929  0.805  0.356  0.006  1.187
softlight/vi_on_ir/0.8      5.605  3.575   5.677  1.851  24.882  0.889  0.712  0.354  0.007  1.213
softlight/vi_on_ir/1.0      5.314  3.222   6.230  1.973  27.628  0.867  0.650  0.368  0.009  1.243
```

rank 选优：**best config = ir_on_vi / α=0.6**。

### 4.2 解读

- **柔光最优配置 vs T5**：对比度/空间频率类指标 SF（11.12 vs 10.59）、SD（64.1 vs 55.4）超过 T5——柔光的"亮处提亮、暗处压暗"天然拉伸动态范围。但保真/信息传递类指标全面落后：MI 4.59 vs 4.61（勉强持平）、SSIM **0.856 vs 0.991**、Qabf **0.570 vs 0.674**、VIF 1.03 vs 1.13、SCD 1.53 vs 1.58，Nabf（伪影）也略差。学习方法在"从两模态传递多少信息、与源图结构多一致"上的优势是本质性的。
- **层序主导，α 次之**：vi_on_ir 全面垫底（SF 仅 5~6，SD 仅 22~28）——把彩色 vi 打到灰度 ir 上等于用低对比度底图压制了 vi 的全部纹理与色彩，压暗、压低对比度。层序间差距远大于 α 间差距；α 增大只是在既定层序上加强修饰幅度。
- **定性观察**：ir_on_vi 顶行偏 IR 暗调、vi_on_ir 底行偏 VI 亮调（网格图行间差异一目了然）；柔光结果亮度自然、接近"VI 略增强"的外观。

### 4.3 总体判断

柔光作为**零参数**混合能获得不错的外观与高对比度（SF/SD 胜出），但保真类/信息传递类指标（SSIM/Qabf/VIF/MI/SCD）与学习方法差距明显——它不是端到端融合方法的替代品。更有价值的用法是作为**融合网络中的一种先验混合算子**：在网络内用柔光替代/补充逐像素加权（例如把 IR 特征以柔光语义调制到 VI 特征上），让网络保住柔光的对比度增益同时学回保真度。

## 5. 已知局限

- **无空间自适应**：同一 (base, blend) 组合全图同一修饰规则，没有"哪里该融合多少"的判断。
- **W3C ≠ PS**：与 Photoshop 的 Pegtop 实现在暗区行为有差异，主观经验不能完全平移。
- **灰度 VI 下优势不适用**：TNO 的 vi 本身是灰度，"VI 保色彩"的动机不成立，柔光退化成两个灰度图的单调混合。
- **抽样仅 6 张、指标为均值**：样本量小，未做显著性检验，结论方向性参考。
- **尺寸对齐**：任何与 T5 输出同图算指标的脚本必须先对齐 16 倍数 resize（`load_pair_t5` 的存在原因），直接用原图尺寸会形状不匹配。

## 6. 运行

```bash
D:/software/anaconda3/envs/xpu/python.exe softlight_blend_probe/probe_softlight.py
```

（从仓库根目录运行。可选参数：`--datasets`（默认 MSRS TNO）、`--num`（每数据集抽样张数，默认 3）、`--orders`（默认 ir_on_vi vi_on_ir）、`--alphas`（默认 0.6 0.8 1.0）、`--t5-root`（默认 `sweeps/out/text_ratio_T5/gen`）、`--out`（默认 `softlight_blend_probe/out`）。）

输出到 `out/`：每图 `{name}_compare.png`（IR|VI|柔光|T5 四联）、`{name}_grid.png`（配置网格）、以及逐图 10 指标 `metrics.csv`。

## 7. 测试

```bash
D:/software/anaconda3/envs/xpu/python.exe -m pytest softlight_blend_probe/test_probe_softlight.py -v
```

15 个单元测试覆盖：中灰 blend 返回 base、全白提亮/全黑压暗方向、α=0 返回 base / α=1 等于裸柔光、输出落在 [0,1]、低/高 base 区闭式解逐点一致、b=0.25 处分段连续、两种层序的形状与语义（含非法 order 抛错、α=0 时 ir_on_vi 退化为 vi）、`load_pair_t5` 尺寸一致时恒等 / 不一致时对齐到 T5 尺寸。
