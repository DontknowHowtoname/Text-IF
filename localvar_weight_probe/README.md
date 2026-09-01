# localvar_weight_probe

验证“局部方差 + 温度 Softmax”动态权重方案的两个前提：

1. **纹理/边缘提取**：局部方差 `V = E[x²] − E[x]²`（box filter, win=7, reflect 边界）等价于局部高频能量，是否与 Sobel/Canny 高亮区域一致。
2. **温度权重极限行为**：`W_ir = exp(V_ir/τ) / (exp(V_ir/τ) + exp(V_vi/τ))`
   - τ→0 退化为 argmax（传统最大值包络）
   - τ→∞ 权重→0.5（简单平均）
   - 中间 τ 为连续混合，方差接近处平滑过渡

**量纲约定**：方差图按最大值归一化到 [0,1] 后再进入 Softmax，τ 定义在归一化尺度上。

## 运行

```bash
D:/software/anaconda3/envs/xpu/python.exe localvar_weight_probe/probe_localvar.py \
  --images 00099D.png 00016N.png --win 7 --taus 0.05 0.1 0.5 1.0 10 1000
```

输出四组图到 `out/`：`*_fig1_texture.png`（四联提取验证）、`*_fig2_weights.png`（各 τ 的 W_ir/W_vi 互补权重网格）、`*_fig3_fusion.png`（梯度域硬/软融合对比）、`*_fig4_fusion_img.png`（图像域融合 F = W_ir·IR + W_vi·VI，硬 max vs 软 τ=0.5，附原模态参考）。

## 测试

```bash
cd localvar_weight_probe && D:/software/anaconda3/envs/xpu/python.exe -m pytest test_probe_localvar.py -v
```
