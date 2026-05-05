# 实验2：同系列内跨模型 E-E / U-U 仿射关系检验

## 目标

对**同系列**模型，检验：
- **E 与 E**：模型 A 的 embedding 矩阵与模型 B 的 embedding 矩阵是否满足仿射变换（E_B ≈ A @ E_A + b，在公共 token 上）
- **U 与 U**：模型 A 的 unembedding 矩阵与模型 B 的 unembedding 矩阵是否满足仿射变换（U_B ≈ A @ U_A + b）

即跨模型的 **E 和 E**、**U 和 U** 的仿射关系，而非同一模型内的 E 与 U。

## 方法

1. 按系列（Qwen3、Qwen2.5、Llama、Gemma2）枚举模型对 (A, B)。
2. 用 token 字符串对齐得到公共 token，得到 E_A_c (n×d_A)、E_B_c (n×d_B)，以及 U_A_c、U_B_c。
3. 拟合仿射：E_B_c = E_A_c @ A^T + b（A: d_B×d_A, b: d_B），用最小二乘求解。
4. 同样对 U 拟合 U_B_c = U_A_c @ A_U^T + b_U。
5. 报告 R²、相对误差等。

## 运行

```bash
cd /root/shared-nvme/ijcai/exp2_affine_cross_model
python run_affine_cross_model.py
```

结果输出在 `results/` 目录。

## 指标与结果说明

- 各指标含义、E/U 分开的解读、微调对 vs 跨规模对的结论：**`results/指标与结果说明.md`**
- **结果与各指标详细分析**（逐指标解释 + 如何读数字 + 结论）：**`results/结果与指标详细分析.md`**
- 实验设计检查、复现性修正、相关文献与结果对比：**`results/实验检查与相关文献.md`**
