

---

# 一、 理论基石：互信息神经网络估计 (MINE)

代码中 `mine_fa` 类实现了 **Mutual Information Neural Estimation**，基于 Belghazi 等人 (2018) 提出的 Donsker-Varadhan 表示：

$$I(X; Y) = \sup_{T_\theta} \mathbb{E}_{P_{XY}}[T_\theta(X, Y)] - \log \mathbb{E}_{P_X \otimes P_Y}[e^{T_\theta(X, Y')}]$$

**其中：**
*   $T_\theta$ 是统计量网络（此处采用 MLP 架构：10 → 10 → log_softmax）。
*   $P_{XY}$ 是真实联合分布。
*   $P_X \otimes P_Y$ 是边缘分布的乘积（通过打乱标签构造负样本）。
*   `fit_mlp()` 通过对比有特征 vs 零特征的损失差来估计 MI 下界。

**工程亮点：**
*   采用 **NLLLoss + LogSoftmax** 组合，等价于 CrossEntropyLoss。
*   SGD 优化器配置 `momentum=0.9`，加速收敛。
*   3 次重复实验取统计显著性边界：$\bar{\mu} - 2\sigma/\sqrt{n}$。

---

# 二、 特征选择引擎：FSNID (Feature Selection via Null Information Difference)

`fsnid_selection` 类实现了**零模型引导的迭代特征删除策略**。

### 算法流程：
1.  **零模型校准**：`null_model()` 生成随机特征计算 MI 上界 $I_{null}^{upper} = \bar{\mu}_{null} + 2\sigma_{null}/\sqrt{3}$。
2.  **互信息排序**：`mi_ordering()` 对每个特征计算 MI，并按升序遍历（从最不重要的开始）。
3.  **贝叶斯决策**：对特征 $f_i$ 计算：
    *   $I(X_{feats}; Y)$：包含 $f_i$ 的 MI。
    *   $I(X_{feats \setminus f_i}; Y)$：删除 $f_i$ 的 MI。
    *   下界判定：$LB = \bar{\mu}_{diff} - 2\sigma_{diff}/\sqrt{3}$。
4.  **若 $LB < I_{null}^{upper}$，则删除 $f_i$**。

**数学意义**：这等价于在 95% 置信水平下检验特征是否显著优于随机噪声。

---

# 三、 对比学习损失：BACLoss (Bidirectional Anchor Contrastive Loss)

这是代码中最具创新性的部分，实现了**双向锚点对比学习**：

```python
def forward(self, logits, labels):
    # 正样本视角 (X维)
    logits_normalx = logits[labels == 0]  # 锚点
    logits_normaly = logits_normalx[:, labels == 0] # 正样本
    logits_normaly_abnormaly = logits_normalx[:, labels == 1] # 负样本
    loss_x = -log( exp(pos) / (exp(pos) + Σexp(neg)) )
    
    # 标签视角 (Y维)
    # 对称计算...
```

**损失函数形式：**
$$\mathcal{L}_{BAC} = -\frac{1}{N_n} \sum_{i=1}^{N_n} \log \frac{\exp(s_{ii}^x/\tau)}{\exp(s_{ii}^x/\tau) + \sum_{j=1}^{N_a} \exp(s_{ij}^x/\tau)} - \frac{1}{N_n} \sum_{i=1}^{N_n} \log \frac{\exp(s_{ii}^y/\tau)}{\exp(s_{ii}^y/\tau) + \sum_{j=1}^{N_a} \exp(s_{ji}^y/\tau)}$$

**其中：**
*   $\tau = 0.1$：温度系数，控制 $max$ 的锐度。
*   双向对比确保**特征空间与标签空间的语义对齐**。

---

# 四、 双塔编码器架构

`ContrastiveModel` 采用了 **Siamese 双端编码器** 设计：

```plaintext
特征编码器 (Encoder_X): Input_dim -> 10 -> 10 -> 5 -> L2归一化
标签编码器 (Encoder_Y): Label_dim -> 10 -> 10 -> 5 -> L2归一化
相似度矩阵: S = Z_X @ Z_Y^T
```

**关键技巧：**
*   **L2 归一化**：将嵌入投影到单位超球面，相似度等价于余弦相似度。
*   **参数冻结**：`label_encoder` 在特征评估时冻结权重，仅更新 `feature_encoder`。
*   **缓存机制**：`cached_z_y` 避免重复计算标签嵌入。

---

# 五、 特征-特征互信息估计

`features_mi_estimation()` 计算特征间的冗余度：

**构造正负样本对：**
*   正样本：$(f_i, f_j)$ 真实配对。
*   负样本：$(f_i, f'_j)$ 打乱 $f_j$ 构造独立分布。

**优化目标：**
$$\max_{T} I(f_i; f_j) = \mathbb{E}[T(f_i, f_j)] - \log \mathbb{E}[e^{T(f_i, f'_j)}]$$

**应用场景**：对 Top3 特征计算平均 MI，评估特征集的信息冗余度。

---

# 六、 训练策略与正则化

### 早停机制 (Early Stopping)
```python
if avg_loss < best_loss:
    best_loss = avg_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        break
```
**原理**：监控验证损失，防止过拟合，在损失不再下降时提前终止训练。

### 随机特征计算
```python
for run_i in range(3):
    random_mi_list.append(compute_bacloss(..., use_random=True))
cached_random_mi = np.mean(random_mi_list)
```
**目的**：建立基线，任何特征若无法显著优于随机噪声（通过 $L(X) < L(X|x, x^*)$），则被删除。

### 冗余性过滤
```python
def is_redundant(feature_idx, selected_features, corr_threshold=0.75):
    for sf in selected_features:
        if |corr(feat_i, sf)| > 0.75:
            return True
```
**Pearson 相关性阈值**：0.75，防止引入高度冗余的特征。

---

# 七、 实验评估协议

### 分类器：KNN (k=5)
*   **理由**：非参数方法，适合评估特征本身而非分类器性能。

### 评估指标：
1.  **Accuracy**：整体分类准确率。
2.  **F1-Score (weighted)**：处理类别不平衡。
3.  **False Positive Rate (FPR)**：安全检测场景关键指标。

### 统计推断：
```python
def calculate_mean_ci(values, confidence=0.95):
    ci_lower, ci_upper = stats.t.interval(confidence, n-1, loc=mean, scale=std_err)
```
**t-分布置信区间**：小样本场景下更精确的区间估计（假设结果服从正态分布）。

---

# 八、 代码实现中的技术细节

### GPU 加速
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
*   自动检测 CUDA 可用性。
*   所有张量操作通过 `.to(device)` 显式设备迁移。

### 批处理策略
```python
for i in range(0, X.size(0), self.batch_size):
    indices = permutation[i : i + self.batch_size]
    if len(indices) == self.batch_size: # 仅处理完整batch
```
**动态裁剪**：确保 batch 大小一致，避免最后一个不完整 batch 的梯度噪声。

### 梯度优化
```python
optimizer = torch.optim.SGD(mlp_regressor.parameters(), lr=0.0001, momentum=0.9)
```
*   **SGD + Momentum**：经典优化组合，适合稀疏梯度。
*   **学习率 1e-4**：保守设置确保稳定收敛。

---

# 理论创新点

1.  **双重信息流**：
    *   **特征-标签互信息 (MINE)**：评估判别能力。
    *   **特征-特征互信息**：评估冗余度。
2.  **对比学习引导**：
    *   BACLoss 通过双向对比确保语义一致性。
    *   损失值直接反映特征与标签的对齐程度。
3.  **零模型假设检验**：
    *   贝叶斯框架下的特征显著性检验。
    *   95% 置信度的统计决策。
4.  **多目标优化**：
    *   最小化分类损失 → 最大判别性。
    *   最小化特征冗余 → 最大化互信息增益。

---

# 潜在改进方向

1.  **MINE 估计稳定性**：
    *   当前采用单点估计，可以引入**梯度惩罚**或**谱归一化**稳定训练。
    *   替代方案：InfoNCE 或 Jensen-Shannon 估计。
2.  **BACLoss 扩展**：
    *   当前仅支持二分类（正常/异常）。
    *   可扩展为多类对比：SupCon (Supervised Contrastive)。
3.  **特征选择策略**：
    *   当前贪心删除可能陷入局部最优。
    *   可引入**遗传算法**或**强化学习**全局优化。
4.  **计算效率**：
    *   当前 42 个特征需要 42 次完整模型训练。
    *   可采用特征重要性排序 + 阈值截断或**并行评估**。

---

# 总结

这是一个融合了 **信息论 + 深度表征学习 + 假设检验** 的交叉领域实现，核心贡献在于：
*   MINE 框架的无监督互信息估计。
*   BACLoss 的双向对比学习设计。
*   零模型引导的特征显著性检验。
*   冗余性过滤的 Pearson 相关性阈值。

