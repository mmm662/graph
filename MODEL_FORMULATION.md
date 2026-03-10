# GraphMM 公式说明（修正版）

## 0. 当前模型定位（统一结论）

当前配置 `use_crf=false, crf_train_loss=ce, inference_use_input_context=true` 下，实际运行的模型是：

> **图增强的 token-level 轨迹纠错模型**：
> 路网图编码 -> 轨迹转移图增强 -> BiGRU 编码 -> GRU + attention 解码 -> 对全图节点分类 -> anchor bias + correction gate。
>
> 当前默认**训练目标是加权 token-level Cross Entropy**，默认**推理不是 CRF 结构化解码**。

---

## 6.2 GCN 归一化权重（修正版）

定义加权入度：

$$
d_i=\sum_{j:(j,i)\in E_T}\tilde w_{ji}.
$$

对边 $(j\to i)$，采用对称归一化权重：

$$
\alpha_{ji}=\frac{\tilde w_{ji}}{\sqrt{d_i d_j}}.
$$

若工程实现中实际采用的是其他归一化形式，则应以实现为准；本文以下统一使用对称归一化写法，因为它更符合常见加权 GCN 的稳定形式。

## 6.3 Weighted GCN 单层（修正版）

设输入为：

$$
z_i^{(0)}=\hat h_i,
$$

则第 $l$ 层更新为：

$$
\bar z_i^{(l+1)}=\sum_{j:(j,i)\in E_T}\alpha_{ji} W_g^{(l)} z_j^{(l)},
$$

$$
z_i^{(l+1)}=\mathrm{ReLU}(\bar z_i^{(l+1)}).
$$

经过 $L_t$ 层后，再做 L2 归一化：

$$
\tilde h_i=\frac{z_i^{(L_t)}}{\|z_i^{(L_t)}\|_2}.
$$

最终增强后的路网表示记为：

$$
\tilde H_R=[\tilde h_1,\ldots,\tilde h_N]^\top.
$$

---

## 8.3 初始解码隐状态（修正版）

设有效位置掩码为：

$$
m_t\in\{0,1\},
$$

其中 $m_t=1$ 表示第 $t$ 个位置有效，$m_t=0$ 表示 padding。
则编码器输出的 masked mean pooling 为：

$$
\bar k=\frac{\sum_{t=1}^L m_t \hat k_t}{\sum_{t=1}^L m_t}.
$$

然后将其作为 decoder 的初始隐状态：

$$
h_0^{\mathrm{dec}}=\bar k.
$$

---

## 9.1 解码输入的构造（修正版）

设 decoder 第 $t$ 步输入为 $d_t$。

### 训练阶段

训练时使用 teacher forcing 或 scheduled sampling 生成的前一时刻 token 作为 decoder 输入。记训练输入序列为：

$$
y^{\mathrm{in}}=(y^{\mathrm{in}}_1,\ldots,y^{\mathrm{in}}_L),
$$

则：

$$
d_1=H_R[x_1],
$$

$$
d_t=H_R[y^{\mathrm{in}}_{t-1}],\qquad t\ge 2.
$$

其中：
- 在纯 teacher forcing 下，$y^{\mathrm{in}}_{t-1}=y^*_{t-1}$。
- 在 scheduled sampling 下，$y^{\mathrm{in}}_{t-1}$ 由真值 token 与模型预测 token 按给定概率混合得到。

### 推理阶段

当前默认配置 `inference_use_input_context=true`，推理时不使用自回归预测前缀，而是使用原始观测轨迹提供上下文，即：

$$
d_1=H_R[x_1],
$$

$$
d_t=H_R[x_{t-1}],\qquad t\ge 2.
$$

因此，当前模型在推理时更接近**参考原始观测序列进行逐点纠错**，而不是完全自回归地重新生成整条轨迹。

### 备注

若希望训练-推理条件完全一致，则可在训练阶段也采用 input-context 形式构造 decoder 输入；否则应将当前做法明确解释为一种“训练时使用前缀监督、推理时使用观测上下文”的工程折中设计。

---

## 13. Graph-CRF 的完整推导（修正版）

源码中实现了可选的 Graph-CRF，用于在候选节点空间上做结构化路径建模。
但需强调：**当前配置默认 `use_crf=false`，因此这一节描述的是可选分支，而非当前线上默认推理路径。**

### 13.1 候选剪枝

对每个时刻 $t$，从 unary logits 中取 top-$r$ 个候选节点：

$$
C_t=\mathrm{TopR}(\ell'_t).
$$

训练时为保证真值路径可见，若真值 $y_t^*\notin C_t$，则补入真值：

$$
C_t\leftarrow C_t\cup \{y_t^*\}.
$$

### 13.2 Unary 分数

定义第 $t$ 个时刻候选节点 $j$ 的 unary 分数为：

$$
u_t(j)=\ell'_{t,j}.
$$

这里 $\ell'_t$ 是加入 input anchor bias 后的 logits。

### 13.3 Pairwise 分数

CRF 的转移分数由节点表示双线性计算：

$$
\psi_t(i,j)=H_R[i]^\top W H_R[j],
$$

其中 $W\in\mathbb R^{d\times d}$ 为可学习参数矩阵。

### 13.4 可达性约束

定义 $A(j)$ 为在路网图上能于给定 hop 限制内到达节点 $j$ 的前驱集合。
若 $i\notin A(j)$，则对该转移加入一个极大负惩罚：

$$
\psi_t^{(\mathrm{final})}(i,j)=\psi_t(i,j)+\lambda_{\mathrm{unreach}}\cdot \mathbf 1(i\notin A(j)),
$$

其中：

$$
\lambda_{\mathrm{unreach}}<0.
$$

### 13.5 序列总分

对一条候选路径

$$
y=(y_1,\ldots,y_L),\qquad y_t\in C_t,
$$

其总分定义为：

$$
S(y)=u_1(y_1)+\sum_{t=2}^L \Big(u_t(y_t)+\psi_t^{(\mathrm{final})}(y_{t-1},y_t)\Big).
$$

### 13.6 条件概率

在候选集剪枝后的路径空间 $\mathcal Y$ 上定义条件概率：

$$
P(y\mid x)=\frac{\exp(S(y))}{\sum_{y'\in\mathcal Y}\exp(S(y'))}.
$$

### 13.7 CRF 训练损失

真值路径记为 $y^*$，则负对数似然为：

$$
\mathcal L_{\mathrm{CRF}}=-\log P(y^*\mid x)=\log Z-S(y^*),
$$

其中配分函数为：

$$
Z=\sum_{y'\in\mathcal Y}\exp(S(y')).
$$

### 13.8 前向递推计算 $\log Z$

定义前向变量 $\alpha_t(j)$ 为以状态 $j$ 结尾的部分路径 log-sum-exp 分数。

初始条件：

$$
\alpha_1(j)=u_1(j),\qquad j\in C_1.
$$

递推：

$$
\alpha_t(j)=u_t(j)+\log\sum_{i\in C_{t-1}}\exp\Big(\alpha_{t-1}(i)+\psi_t^{(\mathrm{final})}(i,j)\Big),\qquad j\in C_t.
$$

终止：

$$
\log Z=\log\sum_{j\in C_L}\exp(\alpha_L(j)).
$$

### 13.9 Viterbi 解码

结构化解码目标为求最大分数路径：

$$
\hat y=\arg\max_{y\in\mathcal Y} S(y).
$$

定义动态规划量：

$$
\delta_1(j)=u_1(j),\qquad j\in C_1,
$$

$$
\delta_t(j)=u_t(j)+\max_{i\in C_{t-1}}\Big(\delta_{t-1}(i)+\psi_t^{(\mathrm{final})}(i,j)\Big).
$$

同时记录回溯指针：

$$
\mathrm{bp}_t(j)=\arg\max_{i\in C_{t-1}}\Big(\delta_{t-1}(i)+\psi_t^{(\mathrm{final})}(i,j)\Big).
$$

最后从

$$
\hat y_L=\arg\max_{j\in C_L}\delta_L(j)
$$

开始回溯，得到最优路径 $\hat y$。

---

## 14. 当前工程默认使用的训练损失（修正版）

当前配置为：

$$
\texttt{use\_crf=false},\qquad \texttt{crf\_train\_loss=ce},
$$

因此实际优化目标不是 CRF 的序列级负对数似然，而是**加权 token-level Cross Entropy**。

### 14.1 token 级分类概率

对每个时刻 $t$，加入 input anchor bias 后的 logits 记为 $\ell'_t\in\mathbb R^N$。对应 softmax 概率为：

$$
p_{t,n}=\frac{\exp(\ell'_{t,n})}{\sum_{m=1}^N \exp(\ell'_{t,m})}.
$$

### 14.2 token 级交叉熵

真值为 $y_t^*$ 时，第 $t$ 个位置的交叉熵为：

$$
\mathcal L_t^{\mathrm{CE}}=-\log p_{t,y_t^*}.
$$

### 14.3 错误位置加权

定义有效位置掩码：

$$
m_t=\mathbf 1(y_t^*\neq \mathrm{PAD}),
$$

原输入是否错误：

$$
e_t=\mathbf 1(x_t\neq y_t^*)\cdot m_t.
$$

则位置权重定义为：

$$
w_t=m_t+e_t(\lambda_{\mathrm{err}}-1),
$$

其中 $\lambda_{\mathrm{err}}=\texttt{error\_token\_weight}$。

### 14.4 最终损失

最终加权交叉熵损失为：

$$
\mathcal L_{\mathrm{CE}}=\frac{\sum_{t=1}^L w_t \mathcal L_t^{\mathrm{CE}}}{\sum_{t=1}^L w_t}.
$$

### 14.5 说明

因此，在当前默认配置下，模型学到的是**逐时刻节点分类**，而不是显式地在整条路径空间上做全局联合最优。

---

## 15.2 混合 teacher input（修正版补充）

先由当前模型产生无梯度预测：

$$
\tilde y_t=\arg\max_n \ell'_{t,n}.
$$

然后按 teacher forcing 比例 $\rho_e$ 构造训练输入 token：

$$
y_t^{\mathrm{in}}=
\begin{cases}
y_t^*, & \text{with prob. }\rho_e,\\
\tilde y_t, & \text{with prob. }1-\rho_e.
\end{cases}
$$

由此构成训练阶段 decoder 输入前缀。
需要注意的是：scheduled sampling 解决的是 teacher forcing 与预测前缀之间的偏差；而当前推理默认使用的是 input-context 模式，因此它与训练输入条件仍不是完全同分布。

---

## 16. 推理阶段纠错门控（修正版）

当前推理先得到未门控预测：

$$
\hat y_t=\arg\max_n \ell'_{t,n}.
$$

若 $\hat y_t=x_t$，则该位置本来就未发生纠错，直接保留。
若 $\hat y_t\neq x_t$，则只有在同时满足以下两个条件时，才允许真正修改原 token。

### 16.1 置信度条件

定义 softmax 概率：

$$
p_{t,n}=\mathrm{softmax}(\ell'_t)_n,
$$

最大概率作为模型置信度：

$$
c_t=\max_n p_{t,n}.
$$

要求：

$$
c_t\ge \gamma_{\mathrm{conf}}.
$$

### 16.2 logit 增益条件

定义候选修正相对原输入 token 的 logit 增益为：

$$
g_t=\ell'_{t,\hat y_t}-\ell'_{t,x_t}.
$$

要求：

$$
g_t\ge \gamma_{\mathrm{gain}}.
$$

### 16.3 最终门控输出

最终输出 $y_t^{\mathrm{final}}$ 为：

$$
y_t^{\mathrm{final}}=
\begin{cases}
\hat y_t, & \hat y_t=x_t,\\
\hat y_t, & \hat y_t\neq x_t,\ c_t\ge \gamma_{\mathrm{conf}},\ g_t\ge \gamma_{\mathrm{gain}},\\
x_t, & \text{otherwise}.
\end{cases}
$$

### 16.4 说明

由于 $\ell'_t$ 已经包含 input anchor bias，因此当 $\hat y_t\neq x_t$ 时，上述增益条件实际上要求候选修正不仅要超过原 token 的原始打分，还必须克服 anchor bias 带来的保守先验。

---

## 17. 当前默认配置下的实际完整公式（修正版）

当前配置为：

$$
\texttt{use\_crf=false},\quad
\texttt{crf\_train\_loss=ce},\quad
\texttt{apply\_input\_anchor\_bias\_training=true},\quad
\texttt{apply\_input\_anchor\_bias\_inference=true},\quad
\texttt{inference\_use\_input\_context=true}.
$$

因此当前实际运行链路如下。

### 17.1 路网节点编码

节点初始表示：

$$
h_i^{(0)}=
\phi\left(
W_o
\begin{bmatrix}
\phi(W_u u_i+b_u)\\
\mathrm{Emb}(f_i)
\end{bmatrix}
+b_o
\right).
$$

边嵌入：

$$
a_{ij}=W_{e2}\phi(W_{e1}e_{ij}+b_{e1})+b_{e2}.
$$

GINE 第 $l$ 层：

$$
m_{j\to i}^{(l)}=\mathrm{MLP}_{\mathrm{msg}}^{(l)}\!\left(
\begin{bmatrix}
h_j^{(l)}\\
a_{ji}
\end{bmatrix}
\right),
$$

$$
m_i^{(l)}=\sum_{j:(j,i)\in E_R} m_{j\to i}^{(l)},
$$

$$
h_i^{(l+1)}=\mathrm{MLP}_{\mathrm{self}}^{(l)}\Big((1+\epsilon^{(l)})h_i^{(l)}+m_i^{(l)}\Big).
$$

最终路网表示：

$$
\hat h_i=\frac{h_i^{(L_g)}}{\|h_i^{(L_g)}\|_2}.
$$

### 17.2 轨迹转移图增强

初始化：

$$
z_i^{(0)}=\hat h_i.
$$

加权 GCN：

$$
\alpha_{ji}=\frac{\tilde w_{ji}}{\sqrt{d_i d_j}},\qquad
 d_i=\sum_{j:(j,i)\in E_T}\tilde w_{ji},
$$

$$
\bar z_i^{(l+1)}=\sum_{j:(j,i)\in E_T}\alpha_{ji}W_g^{(l)}z_j^{(l)},
$$

$$
z_i^{(l+1)}=\mathrm{ReLU}(\bar z_i^{(l+1)}).
$$

增强后的节点表示：

$$
H_R[i]=\frac{z_i^{(L_t)}}{\|z_i^{(L_t)}\|_2}.
$$

### 17.3 编码器

输入轨迹节点嵌入：

$$
e_t=H_R[x_t].
$$

BiGRU 编码：

$$
s_t=
\begin{bmatrix}
\overrightarrow{\mathrm{GRU}}_{\mathrm{enc}}(e_t)\\
\overleftarrow{\mathrm{GRU}}_{\mathrm{enc}}(e_t)
\end{bmatrix}.
$$

线性投影并归一化：

$$
k_t=W_p s_t+b_p,\qquad
\hat k_t=\frac{k_t}{\|k_t\|_2}.
$$

masked mean pooling 初始化 decoder：

$$
h_0^{\mathrm{dec}}=\frac{\sum_{t=1}^L m_t \hat k_t}{\sum_{t=1}^L m_t}.
$$

### 17.4 当前默认推理方式：input-context 解码

$$
d_1=H_R[x_1],\qquad d_t=H_R[x_{t-1}],\ t\ge 2.
$$

decoder GRU：

$$
h_t^{\mathrm{dec}}=\mathrm{GRU}_{\mathrm{dec}}(d_t,h_{t-1}^{\mathrm{dec}}).
$$

dot attention：

$$
s_{t,\tau}=(h_t^{\mathrm{dec}})^\top \hat k_\tau,
$$

$$
\alpha_{t,\tau}=\frac{\exp(s_{t,\tau})}{\sum_{\tau'=1}^L \exp(s_{t,\tau'})},
$$

$$
c_t=\sum_{\tau=1}^L \alpha_{t,\tau}\hat k_\tau.
$$

融合输出：

$$
z_t=W_z
\begin{bmatrix}
h_t^{\mathrm{dec}}\\
c_t
\end{bmatrix}+b_z,\qquad
\hat z_t=\frac{z_t}{\|z_t\|_2}.
$$

### 17.5 对全图节点分类

对任意节点 $n\in\{1,\ldots,N\}$，分类 logit 为：

$$
\ell_{t,n}=\tau \cdot \hat z_t^\top H_R[n].
$$

### 17.6 输入锚点偏置

$$
\ell'_{t,n}=\ell_{t,n}+b_{\mathrm{anchor}}\cdot \mathbf 1(n=x_t).
$$

### 17.7 训练目标

softmax 概率：

$$
p_{t,n}=\frac{\exp(\ell'_{t,n})}{\sum_{m=1}^N \exp(\ell'_{t,m})}.
$$

token 级交叉熵：

$$
\mathcal L_t^{\mathrm{CE}}=-\log p_{t,y_t^*}.
$$

位置权重：

$$
w_t=m_t+\mathbf 1(x_t\neq y_t^*)m_t(\lambda_{\mathrm{err}}-1).
$$

最终损失：

$$
\mathcal L=\frac{\sum_{t=1}^L w_t \mathcal L_t^{\mathrm{CE}}}{\sum_{t=1}^L w_t}.
$$

### 17.8 推理输出

未门控预测：

$$
\hat y_t=\arg\max_n \ell'_{t,n}.
$$

置信度：

$$
c_t=\max_n \mathrm{softmax}(\ell'_t)_n.
$$

logit 增益：

$$
g_t=\ell'_{t,\hat y_t}-\ell'_{t,x_t}.
$$

最终输出：

$$
y_t^{\mathrm{final}}=
\begin{cases}
\hat y_t, & \hat y_t=x_t,\\
\hat y_t, & \hat y_t\neq x_t,\ c_t\ge \gamma_{\mathrm{conf}},\ g_t\ge \gamma_{\mathrm{gain}},\\
x_t, & \text{otherwise}.
\end{cases}
$$

---

## 18. 方法说明（建议新增）

1. 当前默认配置下，模型优化的是**逐位置纠错准确性**，而不是显式的全路径联合最优。
2. Graph-CRF 分支提供了可选的结构化路径约束，但默认配置中未启用。
3. `input_anchor_bias` 与 `confidence/gain gate` 共同引入了“保守纠错”机制，使模型只有在证据充分时才偏离原始观测。
4. 当前推理默认使用 `input-context` 而非纯自回归，因此模型更适合“基于原轨迹进行纠错”，而不是“从头生成轨迹”。
