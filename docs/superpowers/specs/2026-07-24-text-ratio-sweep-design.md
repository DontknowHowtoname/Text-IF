# Text Ratio 扫描实验设计文档

**Date**: 2026-07-24
**Status**: Design
**Parent spec**: `2026-07-03-flir-text-fusion-design.md`
**Target script**: `train_fusion_full_recon_v2_ft.py` (dual-path reconstruction variant)

## 1. 目标与动机

### 1.1 当前问题

现有实验 `all_experiments_ranking_fair.csv` 已经显示出清晰的**保真度 vs 细节量**反相关轴：

- `text_recon2_ft_myweight`（rank 8）: SF=1.0, AG=0.88（细节最大），但 PSNR/SSIM/VIF 全 0
- `text_rcon`（rank 5）: VIF=1.0, PSNR=0.95（保真最大），但 MI/SF/AG 全 0
- `textif-pre`（rank 1）: 中间位置

这是一个 Pareto 前沿，但**样本点是离散且非受控的**——每个实验同时改了多个变量（loss 配方、权重、模型结构），无法支撑论文中"连续权衡曲线"的叙述。

### 1.2 设计目标

- **G1（受控单变量扫描）**: 固定模型结构、固定 `recon_weight`/`max_ratio`/`ssim_ratio`/`upper_weight`，只扫描 `text_ratio`（L_Grad_position 的权重），取值 `{0, 1, 2, 3, 5, 8, 10, 15}`（8 个点）。
- **G2（完整训练 + 完整评估）**: 每个点都跑一次完整训练（`--epochs 50` 默认），再用 `evaluate_experiment_metrics.py` 跑全部 21 个指标。
- **G3（HPC 批量执行）**: 用 SLURM array job 并行跑 8 个点，互不干扰，失败任务可单独重跑。
- **G4（结果汇总）**: 聚合成一个 CSV，schema 与 `all_experiments_fusion_metrics.csv` 对齐，方便复用现有排名脚本。

### 1.3 非目标（YAGNI）

- ❌ 不做 Pareto 画图脚本（画图是后续任务 D 的工作，本任务只产出数据）
- ❌ 不改 `_TASK_DEFAULTS` 的默认值（只加 override 通道）
- ❌ 不增加新的 loss 项或新的指标
- ❌ 不在本地训练，所有训练在超算上通过 `sbatch` 提交

## 2. 现状分析

### 2.1 CLI 已经暴露的权重参数

见 [train_fusion_full_recon_v2_ft.py:260-270](../../../train_fusion_full_recon_v2_ft.py):

```
--upper_weight   (default 1.3)
--recon_weight   (default 0.3)
--max_ratio      (default None = per-task defaults)
--ssim_ratio     (default None = per-task defaults)
```

### 2.2 缺失的 CLI 参数

**`text_ratio` 没有暴露**。当前由 `fusion_dual_recon_prompt_loss._TASK_DEFAULTS` 写死为每任务 `{3, 2, 3, 2}`（见 [scripts/losses.py:476-481](../../../scripts/losses.py#L476-L481)）：

```python
_TASK_DEFAULTS = {
    "low_light":        {"max_ratio": 4, "ssim_ratio": 1,  "text_ratio": 3},
    "over_exposure":    {"max_ratio": 3, "ssim_ratio": 1,  "text_ratio": 2},
    "ir_low_contrast":  {"max_ratio": 4, "ssim_ratio": 1,  "text_ratio": 3},
    "ir_noise":         {"max_ratio": 3, "ssim_ratio": 1,  "text_ratio": 2},
}
```

`max_ratio`/`ssim_ratio` 已有完整的 override 通道（`self.max_ratio` 存下来，`forward` 里判断 None 走默认），照抄这个模式即可。

## 3. 设计

### 3.1 代码改动（最小侵入，3 个文件）

实例化链路（**三层 plumbing**，每层都要加 `text_ratio`）：

```
train_fusion_full_recon_v2_ft.py
  └── calls train_one_epoch_recon_dual(..., max_ratio, ssim_ratio)  [scripts/utils.py:515]
        └── constructs fusion_dual_recon_prompt_loss(...)            [scripts/losses.py:460]
```

`max_ratio`/`ssim_ratio` 已经走完整条链路（CLI → 训练函数参数 → loss 构造），`text_ratio` 照抄这个模式。

#### 改动 1：`scripts/losses.py` — `fusion_dual_recon_prompt_loss.__init__` 增加 `text_ratio`

对照 [losses.py:460-501](../../../scripts/losses.py#L460-L501)：

```python
class fusion_dual_recon_prompt_loss(nn.Module):
    def __init__(self, upper_weight=1.3, recon_weight=1.0,
                 max_ratio=None, ssim_ratio=None, text_ratio=None):  # 新增 text_ratio
        super(fusion_dual_recon_prompt_loss, self).__init__()
        self.fusion_loss = fusion_loss()
        self.dual_recon_loss = DualReconLoss(upper_weight=upper_weight)
        self.recon_weight = recon_weight
        self.max_ratio = max_ratio
        self.ssim_ratio = ssim_ratio
        self.text_ratio = text_ratio  # 新增

    def forward(self, ...):
        ...
        for idx, task_type in enumerate(task):
            ...
            defaults = self._TASK_DEFAULTS[task_type]
            mr = self.max_ratio if self.max_ratio is not None else defaults["max_ratio"]
            sr = self.ssim_ratio if self.ssim_ratio is not None else defaults["ssim_ratio"]
            tr = self.text_ratio if self.text_ratio is not None else defaults["text_ratio"]  # 新增
            loss, ... = self.fusion_loss(
                img_A, img_B, img_f,
                max_ratio=mr, ssim_ratio=sr, text_ratio=tr)  # 修改：tr 替换 defaults["text_ratio"]
```

#### 改动 2：`scripts/utils.py` — `train_one_epoch_recon_dual` 和 `evaluate_recon_dual` 增加 `text_ratio` 参数

对照 [utils.py:515-520](../../../scripts/utils.py#L515-L520)：

```python
def train_one_epoch_recon_dual(model, model_clip, optimizer, lr_scheduler, data_loader, device, epoch,
                                recon_weight=1.0, max_ratio=None, ssim_ratio=None,
                                text_ratio=None):  # 新增
    ...
    loss_function = fusion_dual_recon_prompt_loss(recon_weight=recon_weight,
                                                  max_ratio=max_ratio, ssim_ratio=ssim_ratio,
                                                  text_ratio=text_ratio)  # 新增
```

同样修改 [evaluate_recon_dual](../../../scripts/utils.py#L596-L598)（line 596-598），保持 eval 和 train 的 loss 一致。

#### 改动 3：`train_fusion_full_recon_v2_ft.py` — 增加 `--text_ratio` CLI 并贯通

CLI 入口在 [line 269 后](../../../train_fusion_full_recon_v2_ft.py#L269)：

```python
parser.add_argument('--text_ratio', type=float, default=None,
                    help='Override text_ratio for all tasks (default: None = per-task defaults)')
```

调用 `train_one_epoch_recon_dual` 处（[line 171-181](../../../train_fusion_full_recon_v2_ft.py#L171-L181)）：

```python
(train_loss, ...) = train_one_epoch_recon_dual(
    ...,
    recon_weight=args.recon_weight,
    max_ratio=args.max_ratio,
    ssim_ratio=args.ssim_ratio,
    text_ratio=args.text_ratio,  # 新增
)
```

调用 `evaluate_recon_dual` 处（[line 192 附近](../../../train_fusion_full_recon_v2_ft.py#L192)）同样加 `text_ratio=args.text_ratio`。

### 3.2 新增文件（`sweeps/` 目录，4 个文件）

#### 文件 1：`sweeps/sweep_text_ratio.sbatch`

SLURM array job 脚本，单任务 = 单个 `text_ratio` 值。

**关键设计**：
- `#SBATCH --array=0-7`，对应 `text_ratios=({0,1,2,3,5,8,10,15})` 8 个值
- 通过 `${text_ratios[$SLURM_ARRAY_TASK_ID]}` 取当前值
- 输出目录：`sweeps/out/text_ratio_T{T}/train/`，`sweeps/out/text_ratio_T{T}/metrics/`
- 调用 `sweeps/run_single.sh` 完成训练 + 评估
- 失败任务可用 `sbatch --array=3-3 sweep_text_ratio.sbatch` 单独重跑

**模板变量（需要用户填）**：
- `REPO_DIR`：仓库在超算上的路径
- `DATASET_*_PATH`：4 个任务的数据路径
- `PRETRAINED_WEIGHTS`：`textif-me` 预训练权重路径
- `CONDA_ENV` / `MODULE_LOAD`：环境激活方式
- `GPU_TYPE` / `PARTITION`：SLURM 队列

#### 文件 2：`sweeps/run_single.sh`

单次运行的包装脚本，参数化 `text_ratio`：

```bash
run_single.sh <text_ratio_value>
```

执行流程：
1. `cd $REPO_DIR`
2. 激活环境
3. 训练：
   ```bash
   python train_fusion_full_recon_v2_ft.py \
     --text_ratio $T \
     --weights $PRETRAINED_WEIGHTS \
     --low_light_path $DATASET_LL \
     --over_exposure_path $DATASET_OE \
     --ir_low_contrast_path $DATASET_IC \
     --ir_noise_path $DATASET_IN \
     --output_dir sweeps/out/text_ratio_T${T}/train
   ```
4. 评估（用 [evaluate_textif_full_recon_v2.py](../../../evaluate_textif_full_recon_v2.py)，与现有 `text_recon2_ft*` 实验一致）：
   ```bash
   python evaluate_textif_full_recon_v2.py \
     --weights_path sweeps/out/text_ratio_T${T}/train/weights/checkpoint.pth \
     --data_path $EVAL_DATA_PATH \
     --output_dir sweeps/out/text_ratio_T${T}/metrics
   ```
   输出 `evaluation_summary.csv`（long-format: 两列 `metric,average`）和 `fused/`（融合图）。

#### 文件 3：`sweeps/aggregate_sweep.py`

收集所有 `sweeps/out/text_ratio_T*/metrics/evaluation_summary.csv`，pivot 成一个 wide-format 汇总 CSV：

```
text_ratio, EN, MI, NMI, SF, AG, SD, CC, SCD, PSNR, MSE, VIF, SSIM, MS_SSIM, Qabf, Nabf, CE, QNCIE, TE, EI, Qy, Qcb, experiment_dir
0,    ...
1,    ...
2,    ...
...
```

- 输入：long-format（每行一个 metric，两列 `metric,average`）—— 对应 [evaluate_textif_full_recon_v2.py:368](../../../evaluate_textif_full_recon_v2.py#L368) 的输出格式
- 输出：wide-format（每行一个 `text_ratio`，21 个 metric 列）—— 与 [results/all_experiments_fusion_metrics.csv](../../../results/all_experiments_fusion_metrics.csv) schema 对齐（多一个 `text_ratio` 列）
- 支持部分扫描（缺失的 `T` 值跳过并在 stderr 报告）
- 输出：`sweeps/text_ratio_sweep_summary.csv`

#### 文件 4：`sweeps/README.md`

使用说明：
1. 如何填模板变量
2. 如何提交 `sbatch sweeps/sweep_text_ratio.sbatch`
3. 如何监控 `squeue -u $USER`
4. 如何重跑失败任务
5. 如何聚合 `python sweeps/aggregate_sweep.py`

### 3.3 前置依赖：训练脚本的 `--output_dir` 入口

**需要确认**：`train_fusion_full_recon_v2_ft.py` 当前输出目录是写死的 `./experiments/TextIF_full_recon_v2_ft_{timestamp}`（见 [line 33-34](../../../train_fusion_full_recon_v2_ft.py#L33-L34)）。需要支持自定义输出目录。

**选项**（实施时决定，倾向 A）：
- **A. 加 `--output_dir` CLI 参数**：默认行为不变，传了就写到指定目录（无 timestamp 后缀），方便 sweep 管理脚本预测路径。
- **B. 在 `run_single.sh` 里训练后 `mv` 目录**：不动训练脚本，但脆弱（要解析 timestamp）。

→ 选项 A 更干净。**这算改动 2 的附加小改**，在 [line 33](../../../train_fusion_full_recon_v2_ft.py#L33) 改为：

```python
filefold_path = args.output_dir or "./experiments/TextIF_full_recon_v2_ft_{}".format(file_name)
```

## 4. 数据流

```
[SLURM array 0-7]
        │
        ├── task 0 (T=0)  ── train ──> evaluate ──> metrics.csv
        ├── task 1 (T=1)  ── train ──> evaluate ──> metrics.csv
        ├── ...
        └── task 7 (T=15) ── train ──> evaluate ──> metrics.csv
                                                      │
                                                      ▼
                                  aggregate_sweep.py
                                                      │
                                                      ▼
                          sweeps/text_ratio_sweep_summary.csv
                                                      │
                                                      ▼
                            复用 rank_experiments.py / 后续画图
```

## 5. 验收标准

1. ✅ `python -c "from scripts.losses import fusion_dual_recon_prompt_loss; fusion_dual_recon_prompt_loss(text_ratio=5)"` 不报错（构造函数兼容）
2. ✅ `python train_fusion_full_recon_v2_ft.py --help` 显示 `--text_ratio`
3. ✅ `python train_fusion_full_recon_v2_ft.py --text_ratio 0 --epochs 1`（dry run 1 epoch）能正常启动训练，且训练日志里能看到 `text_ratio=0` 被使用
4. ✅ `sbatch --array=0-0 sweep_text_ratio.sbatch` 能成功提交并产出 `sweeps/out/text_ratio_T0/metrics/evaluation_summary.csv`
5. ✅ `python sweeps/aggregate_sweep.py` 能从现有 out 目录聚合出 wide-format CSV，且 schema 与 `all_experiments_fusion_metrics.csv` 一致（多 `text_ratio` 列）

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| HPC 数据集路径与本地不同 | `sbatch` 模板把路径作为变量，让用户填 |
| 8 个并行任务占卡多 | 用户可自行调整 `--array=0-1` 分批提交 |
| `text_ratio=0` 训练发散（梯度损失项消失） | 已有 `grad_clip=1.0` 和 NaN skip 逻辑；如仍发散，记录在文档里 |
| 评估脚本需要模型权重作为输入 | `evaluate_textif_full_recon_v2.py --weights_path` 直接读取，无需改动；run_single.sh 已传递正确路径 |

## 7. 后续（不在本任务范围内）

- 任务 B：loss 组件消融（去一项跑一次）
- 任务 D：基于 `text_ratio_sweep_summary.csv` 画 Pareto 曲线
- 在其他数据集（MSRS/RoadScene）上重复 `text_ratio` 扫描，验证权衡轴的 dataset-independence
