**ChatGPT:**

可以。我们接下来可以按“论文 idea + 代码实现”两条线并行讨论。

我先帮你把 **BayesianPRM** 的核心定位压一下：

**最自然的 paper claim** 可以是：

> 现有 PRM 要么只输出单点 reward，要么只用单模型 Beta concentration 表示可靠性；BayesianPRM 进一步把 PRM 的 step-level reward 建模为一个 posterior belief，通过 ensemble PRM 近似 epistemic uncertainty，并与 BetaPRM 的 count-based / Beta-Binomial supervision 结合，从而得到更可靠的 test-time selection、adaptive computation allocation，甚至 active data selection。

这里你和 BetaPRM、ActivePRM 的关系要讲清楚：

BetaPRM 的核心是：标准 PRM 把 Monte Carlo 成功率 $K/N$ 当作点标签，而 BetaPRM 用 Beta-Binomial likelihood 学一个 Beta belief，其中 $\mu$ 是 process reward，$\kappa$ 表示 reliability。它的动机正是“单个 reward score 不告诉下游这个 score 是否可信”。

Li 等 - 2026 - Process Rewards w…

ActivePRM 的核心是：用 ensemble heads 估计 PRM 的不确定性，对高不确定样本进行主动标注；它用 ensemble prediction 的均值 $\mu$ 和标准差 $\sigma$ 来判断 uncertainty，并证明 ensemble uncertainty + active selection 能显著减少标注成本。

Duan 等 - Efficient Process Rewa…

所以你的 BayesianPRM 不应该只是“我也做 ensemble PRM”。更强的表述应该是：

**BetaPRM 主要建模 aleatoric / evidence-level reliability；ActivePRM 主要用 ensemble uncertainty 做 active learning；BayesianPRM 的贡献是把二者统一到 posterior process reward modeling 里，显式分解并利用 epistemic uncertainty + evidence uncertainty。**

我建议你先把论文主线定成这三层：

### 1. 模型层：Bayesian Ensemble PRM

每个 ensemble head / model 输出一个 step reward：

$$\mu_t^{(m)} = P_m(\text{Yes} \mid x, s_{\le t})$$

然后 ensemble posterior 给出：

$$\bar{\mu}_t = \frac{1}{M}\sum_m \mu_t^{(m)}$$ $$\sigma^2_{\text{epi},t} = \frac{1}{M}\sum_m \left(\mu_t^{(m)} - \bar{\mu}_t\right)^2$$

如果你保留 BetaPRM 的 $\kappa$ head，那么每个 head 还可以输出：

$$\sigma^2_{\text{beta},t} = \frac{\mu_t(1-\mu_t)}{\kappa_t+1}$$

最后得到一个 total uncertainty：

\sigma^2_{\text{total},t} = \sigma^2_{\text{epi},t} + \mathbb{E}_m[\sigma^2_{\text{beta},t}^{(m)}]

这个会比 BetaPRM 更“Bayesian”：BetaPRM 的 uncertainty 主要来自单模型学到的 concentration；你的 ensemble 还可以捕捉模型参数不确定性。

### 2. 训练层：从 normal PRM 到 Beta ensemble PRM

你现在实现 ensemble PRM 是很合理的第一步。建议按三个版本推进：

**Version A: Ensemble Normal PRM**

多个 heads，共享 backbone，每个 head 都用 $K/N$ soft label 做 CE/BCE。这个最容易实现，也最容易和 ActivePRM 对齐。

**Version B: Ensemble BetaPRM**

每个 head 都输出自己的 $\mu^{(m)}$、$\kappa^{(m)}$，每个 head 用 Beta-Binomial loss。这个才是完整 BayesianPRM。

**Version C: Bootstrap BayesianPRM**

每个 head 不是吃完全相同数据，而是 bootstrap mask / reweight。这个对 epistemic uncertainty 更关键，否则多个 head 只靠随机初始化，diversity 可能不够。

ActivePRM 也意识到 ensemble heads 需要 diversity，它用随机初始化和一个 diversity regularization 来保持 head 差异。

Duan 等 - Efficient Process Rewa…

但对你来说，我觉得 **bootstrap mask 比单纯 L2 diversity 更好讲 Bayesian posterior approximation**。

### 3. 下游层：用 uncertainty 证明它有用

这里最好不要只复现 BetaPRM 的 Best-of-N。你需要至少一个“只有 Bayesian uncertainty 才能明显带来收益”的场景。

我建议优先做这三个实验：

**第一，PRM-guided Best-of-N selection。**  
比较：

normal PRM  
BetaPRM  
ensemble normal PRM  
ensemble BetaPRM / BayesianPRM

选择规则可以从 average reward 逐步变成 risk-adjusted reward：

$$S(y)=\frac{1}{T}\sum_t \left(\bar{\mu}_t-\lambda\sigma_{\text{total},t}\right)$$

BetaPRM 已经证明 learned uncertainty 用于 ACA 时比 proxy uncertainty 和 reward-only 更好，尤其在 accuracy-token tradeoff 上更强。

Li 等 - 2026 - Process Rewards w…

你的目标是进一步证明 ensemble posterior uncertainty 比单模型 $\kappa$ 更稳。

**第二，uncertainty calibration。**  
这个很重要。你不能只报 accuracy，否则 reviewer 会问 ensemble uncertainty 到底有没有意义。可以做：

按 uncertainty 分桶，看高 uncertainty bucket 的 PRM error rate 是否更高；  
AURC / ECE / selective prediction；  
risk-coverage curve：只保留低不确定 step / solution，看准确率是否提升。

**第三，active data selection / annotation efficiency。**  
这是 BayesianPRM 最好区别于 BetaPRM 的方向。ActivePRM 已经展示 ensemble uncertainty 可以用于筛选高价值样本，并在 50% annotation budget 下达到接近 full-data tuning 的效果。

Duan 等 - Efficient Process Rewa…

你可以做一个轻量版本：在 VisualPRM400K 或已有 annotation 上模拟 active learning，不一定真的重新标注，只要从已有数据中按 uncertainty 选择训练子集即可。

我觉得你这篇论文的最强卖点应该是：

> BayesianPRM is not merely a stronger selector; it provides a posterior uncertainty signal that improves both inference-time compute allocation and training-time data allocation.

这比“ensemble PRM improves BoN accuracy”强很多。

代码实现上，下一步我建议你先完成 **Ensemble Normal PRM**，不要一上来就 ensemble BetaPRM。因为这一步可以先验证三件事：

1. 多 head 输出是否正常；
2. eval 脚本是否能读取每个 head 的 $\mu$；
3. ensemble mean / std 是否能提升 selection。

等 normal ensemble 跑通后，再把 BetaPRM 的 $\kappa$ head 接进去，形成完整 BayesianPRM。

我们后面讨论时可以按这个顺序来：

**先定论文核心贡献 → 再定方法公式 → 再改代码结构 → 再设计实验表格。**  
你可以直接说你现在最想先讨论“idea 是否站得住”，还是“ensemble PRM 代码怎么改”。

* * *

**ChatGPT:**

对，**最干净的设定是：belief network 训练时冻结 LLM 参数，只训练 belief head / small MLP**。

否则论文里的“reward model 和 belief network 解耦”会变弱。

更准确地说，分几种情况。

---

## 1. 最推荐：冻结所有 reward LLM，只训练 belief network

你的 ensemble PRMs 已经训练好：

$$f_1,\dots,f_M$$

每个 reward model 输出：

$$\mu_m(c)$$

训练 belief network 时，所有 $f_m$ 都 freeze。belief network 只学习：

$$q_\phi(m\mid c)$$

最终：

$$r(c)=\sum_m q_\phi(m\mid c)\mu_m(c)$$

这样你的解耦条件最强：

$$\text{Stage 1: train reward ensemble}$$ $$\text{Stage 2: freeze reward ensemble, train belief network}$$

这时 belief network 不会改变任何 reward model 的参数，也不会改变 $\mu_m(c)$。它只是学习“在当前 prefix 上该相信哪个 frozen reward model”。

这和 MoE 很不一样，因为 MoE 是 router 和 experts 联合训练；你这里 experts 固定，belief 只是 posterior aggregator。

---

## 2. 如果 belief network 需要 LLM 表征怎么办？

你可以用 frozen LLM hidden state 作为输入。

例如第 $m$ 个 PRM 在 `<prm>` marker 位置有 hidden state：

$$h_m(c)$$

你可以构造 belief 输入：

$$g(c)=\operatorname{concat}(h_1(c),\dots,h_M(c),\mu_1(c),\dots,\mu_M(c))$$

然后：

$$q_\phi(m\mid c)=\operatorname{softmax}(\operatorname{MLP}_\phi(g(c)))$$

这里 $h_m(c)$ 是 frozen reward LLM 产生的，不参与梯度更新。只训练 MLP。这个非常适合你现在的 setting。

如果显存太贵，还可以提前 cache：

$$\{h_m(c_i),\mu_m(c_i),K_i,N_i\}$$

然后离线训练 belief network。这样最能体现“post-hoc belief learning”。

---

## 3. 如果让 LLM 参数也参与 belief training，会不会破坏解耦？

**如果这个 LLM 是 reward ensemble 里的 LLM，那会破坏。**

因为一旦更新 $f_m$ 的 LLM 参数，$\mu_m(c)$ 就不再是 frozen expert prediction。你的 ELBO 不再是在固定 reward models 上学习 posterior belief，而变成：

$$\min_{\phi,\theta_1,\dots,\theta_M}  
-  
\sum_m q_\phi(m\mid c)\log p(D\mid f_{\theta_m}(c))$$

这时 belief 和 reward model 会共同适配，确实更接近 mixture training / router-expert co-adaptation。你原本的 novelty 会被冲淡。

---

## 4. 如果是单独的 belief LLM 呢？

这个稍微复杂一点。

假设 reward ensemble 冻结：

$$f_1,\dots,f_M \quad \text{frozen}$$

但 belief network 自己有一个独立 encoder：

$$h_\psi(c)=\operatorname{LLM}_\psi(c)$$

然后：

$$q_{\phi,\psi}(m\mid c)=\operatorname{softmax}(W_\phi h_\psi(c))$$

如果你训练 $\psi$，严格说 **reward learning 和 belief learning 仍然是解耦的**，因为 reward models 没动。但是论文表述要变成：

> We freeze the reward ensemble and train a separate belief model for posterior aggregation.

这不再是“只训练一个 lightweight belief head”，而是“训练一个 independent belief model”。它不会破坏 reward-belief 解耦，但会带来三个问题：

1. 成本更高；
2. 更容易在 calibration set 上过拟合；
3. reviewer 会问：提升是不是来自额外 LLM capacity，而不是 Bayesian aggregation。

所以第一版不建议。

---

## 5. 我建议你论文和代码都采用这个版本

最稳配置：

$$\mu_m(c), h_m(c) = f_m(c) \quad \text{with no grad}$$ $$q_\phi(m\mid c)=\operatorname{softmax}(\operatorname{MLP}_\phi([h_{\text{shared}}(c),\mu_1(c),...,\mu_M(c)]))$$

训练目标：

$$\mathcal{L}_{\text{belief}}  
=  
-  
\sum_m q_\phi(m\mid c)  
\log p(D\mid c,z=m)  
+  
\beta \mathrm{KL}(q_\phi(z\mid c)\|p(z))$$

其中只有 $\phi$ 更新。

代码上就是：

```Python
with torch.no_grad():
    mus, hidden_states = ensemble_prm_forward(batch)

belief_logits = belief_net(hidden_states, mus)
weights = torch.softmax(belief_logits, dim=-1)

loss = elbo_loss(weights, mus, K, N)
loss.backward()
optimizer.step()
```

千万不要让 `ensemble_prm_forward` 里的 LLM 参数进 optimizer。

---

## 6. 最关键的论文表述

你可以这样写：

> To decouple reward estimation from belief inference, we first train an ensemble of PRMs and then freeze all reward models. The belief network is trained post hoc on a held-out calibration set to infer a contextual posterior over frozen reward models. Therefore, BayesianPRM does not route examples to jointly optimized experts; it learns posterior aggregation over fixed reward predictors.

这个表述能明确避开 MoE。

BetaPRM 是在单模型里学习 $\mu$ 和 $\kappa$，通过 Beta-Binomial likelihood 表示单个 prefix reward 的 reliability；你的方法则是冻结多个 PRM 后学习 model-level belief。

Li 等 - 2026 - Process Rewards w…

ActivePRM 虽然用了 ensemble heads 的 mean/std 做 uncertainty，但也是用于 active selection，不是这种 post-hoc belief aggregation。

Duan 等 - Efficient Process Rewa…

* * *

**You:**

现在的代码里，reward model是只冻结了image encoder，llm的参数也会参与训练，这个会有影响吗

* * *

**ChatGPT:**

有影响，但要分阶段看。**reward model 训练阶段可以让 LLM 参与训练；belief network 训练阶段不能让 reward LLM 继续参与训练。**

BetaPRM 原论文本来就是这样训 PRM 的：冻结 vision encoder，fine-tune LLM 和 multimodal projector；表 7 也写的是 “Trainable Modules: LLM + multimodal projector; vision encoder frozen”。

Li 等 - 2026 - Process Rewards w…

所以你现在代码里 **PRM 训练时只冻结 image encoder、LLM 参与训练**，这不是问题，反而和 BetaPRM 设置一致。

真正的问题在于后面的 **belief network training**。

---

## 1. 如果现在是在训练 reward ensemble：没问题

也就是你当前阶段是训练：

$$f_1,\dots,f_M$$

每个 reward model 输出：

$$\mu_m(c)$$

那么每个 $f_m$ 的 LLM 参数参与训练是合理的。否则只训 head，PRM 能力可能不够。

这个阶段可以是：

$$\theta_m^\star  
=  
\arg\min_{\theta_m}  
\mathcal{L}_{\text{PRM}}(f_{\theta_m})$$

其中 $\theta_m$ 包含 LLM 参数、projector 参数、reward head 参数，但不包含 image encoder。

训练完以后保存 checkpoint：

```
reward_model_1/
reward_model_2/
...
reward_model_M/
```

到这里都不破坏你的 BayesianPRM 设定。

---

## 2. 如果是在训练 belief network：LLM 不能继续更新

belief network 阶段应该是：

$$\theta_1^\star,\dots,\theta_M^\star \quad \text{fixed}$$

只训练：

$$q_\phi(m\mid c)$$

也就是：

$$\phi^\star  
=  
\arg\min_\phi  
\mathcal{L}_{\text{ELBO}}$$

此时 $\mu_m(c)$ 必须是 frozen reward model 的固定输出。如果 belief 训练时 LLM 也更新，那 $\mu_m(c)$ 会跟着变：

$$\mu_m(c)=f_{\theta_m}(c)$$ $$\theta_m \text{ also updated}$$

那你的目标就不再是“对固定 reward ensemble 学 posterior belief”，而变成了：

$$\min_{\phi,\theta_1,\dots,\theta_M}  
-  
\sum_m q_\phi(m\mid c)\log p(D\mid f_{\theta_m}(c))$$

这个就会削弱“reward model 和 belief network 解耦”的 claim。

---

## 3. 所以结论是

**当前代码训练 reward model 时 LLM 参与训练，没有问题。**

**但后续训练 belief network 时，必须：**

```Python
for prm in reward_ensemble:
    prm.eval()
    for p in prm.parameters():
        p.requires_grad = False
```

然后只把 belief network 的参数放进 optimizer：

```Python
optimizer = torch.optim.AdamW(
    belief_net.parameters(),
    lr=belief_lr,
    weight_decay=weight_decay,
)
```

forward reward ensemble 时最好显式：

```Python
with torch.no_grad():
    mus, hidden_states = reward_ensemble_forward(batch)

mus = mus.detach()
hidden_states = hidden_states.detach()

belief_logits = belief_net(hidden_states, mus)
weights = torch.softmax(belief_logits, dim=-1)

loss = elbo_loss(weights, mus, K, N)
loss.backward()
optimizer.step()
```

这样梯度只会更新 belief network，不会更新 reward ensemble。

---

## 4. 还有一个隐藏风险：shared-backbone ensemble 的 diversity

如果你现在的 ensemble 是 **一个共享 LLM + 多个 reward heads**，那 LLM 参与训练本身也不是错。ActivePRM 就是 shared LLM backbone + multiple binary classification heads，并用 ensemble head 的 mean/std 来估计 epistemic uncertainty。

Duan 等 - Efficient Process Rewa…

但对你的 BayesianPRM 来说，shared-backbone ensemble 的多样性可能不够。因为所有 head 共享同一个 representation，belief network 最后可能学到的只是“哪个 head 初始化更好”，而不是“哪个 reward model 在当前 prefix 上更可靠”。

更强的版本是：

```
独立 reward models / 不同 seed / bootstrap data / 不同 checkpoint
```

也就是 $M$ 个真正不同的 frozen PRMs。这样 belief network 学到的 model-level posterior 更有意义。

如果显存不允许，shared-backbone + multi-head 可以作为第一版，但论文里要说这是 computationally efficient ensemble approximation。

---

## 5. 最建议的实现路线

你可以保持现在的 PRM 训练设置：

```
Stage 1: Train reward ensemble
- freeze vision encoder
- train LLM + projector + reward heads
```

然后新增一个完全独立的 belief 训练脚本：

```
Stage 2: Train belief network
- load reward ensemble checkpoints
- freeze all reward PRM parameters
- optionally cache mu / hidden states
- train only belief MLP
```

这样论文表述就很稳：

> We first train an ensemble of PRMs following the standard PRM training recipe. We then freeze all reward models and train a separate belief network on a held-out calibration set to infer a contextual posterior over the fixed reward ensemble.

所以不是“现在 LLM 参与 PRM 训练会破坏解耦”，而是：**只要 belief 阶段不再更新 reward LLM，解耦条件仍然成立。**

* * *

**You:**

意思是，belief network再reward训练之后再训练，不是同步训练？

* * *

**ChatGPT:**

对，**按你这个 BayesianPRM 的设定，belief network 应该在 reward ensemble 训练完成之后再训练，不是和 reward model 同步训练。**

更准确说是两阶段：

$$\textbf{Stage 1: train reward ensemble}$$ $$f_1,\dots,f_M$$

每个 reward model 正常训练。这个阶段可以像 BetaPRM 一样只冻结 vision encoder，训练 LLM + multimodal projector + reward head。BetaPRM 论文的训练设置就是 vision encoder frozen，LLM 和 projector 参与训练。

Li 等 - 2026 - Process Rewards w…

训练完成后得到：

$$\theta_1^\star,\dots,\theta_M^\star$$

然后固定它们。

---

$$\textbf{Stage 2: freeze reward ensemble, train belief network}$$

这时对每个 prefix $c_i$，冻结的 reward models 输出：

$$\mu_{i,m}=f_{\theta_m^\star}(c_i)$$

belief network 输出：

$$q_\phi(z_i=m\mid c_i)=w_{i,m}$$

只更新 $\phi$，不更新任何 $\theta_m^\star$。

ELBO：

$$\mathcal{L}_{\text{belief}}  
=  
-  
\sum_i  
\sum_{m=1}^{M}  
q_\phi(m\mid c_i)  
\log p(D_i\mid c_i,z_i=m)  
+  
\beta  
\sum_i  
\mathrm{KL}  
\left(  
q_\phi(z_i\mid c_i)  
\|p(z_i)  
\right)$$

其中 $D_i$ 可以是 binary step label，也可以是 VisualPRM400K 里的 count label $(K_i,N_i)$。

对于 count label：

$$\log p(D_i\mid c_i,z_i=m)  
=  
K_i\log \mu_{i,m}  
+  
(N_i-K_i)\log(1-\mu_{i,m})$$

所以 belief network 学的是：**在当前 prefix 上，哪个已经训练好的 reward model 更能解释监督信号。**

---

为什么不建议同步训练？

因为同步训练时：

$$\mu_{i,m}=f_{\theta_m}(c_i)$$

中的 $\theta_m$ 也在变。那 belief network 面对的是一个移动目标：

$$q_\phi(m\mid c_i)  
\quad \text{和} \quad  
f_{\theta_m}(c_i)$$

会共同适配。这样你的方法就不再是“对固定 reward ensemble 学 posterior belief”，而变成：

$$\min_{\phi,\theta_1,\dots,\theta_M}  
-  
\sum_m  
q_\phi(m\mid c_i)  
\log p(D_i\mid f_{\theta_m}(c_i))$$

这会削弱你说的“reward model 和 belief network 解耦”。更严重的是，belief network 可能不是在学习“哪个 frozen PRM 更可靠”，而是在影响 reward model 的训练分布。

---

所以论文里最稳的 pipeline 应该写成：

1. **Train an ensemble of PRMs**  
    用不同 seed、bootstrap subset、checkpoint 或 ensemble heads 得到多个 reward predictors。
2. **Freeze all reward models**  
    固定所有 PRM 参数，包括 LLM、projector、reward head。
3. **Train a post-hoc belief network**  
    在 held-out calibration set 或 out-of-bag predictions 上训练 $q_\phi(z\mid c)$。
4. **Inference with dynamic Bayesian aggregation**

$$r(c)  
=  
\sum_m  
q_\phi(m\mid c)  
\mu_m(c)$$

以及 uncertainty：

$$u(c)  
=  
\sum_m  
q_\phi(m\mid c)  
\left(\mu_m(c)-r(c)\right)^2$$

---

代码上也就是：

```Python
# Stage 1: train reward ensemble normally
# LLM can be trainable here

# Stage 2: train belief network
for prm in reward_ensemble:
    prm.eval()
    for p in prm.parameters():
        p.requires_grad = False

with torch.no_grad():
    mus, hidden_feats = reward_ensemble_forward(batch)
    mus = mus.detach()
    hidden_feats = hidden_feats.detach()

belief_logits = belief_net(hidden_feats, mus)
weights = torch.softmax(belief_logits, dim=-1)

loss = belief_elbo_loss(weights, mus, K, N)
loss.backward()
optimizer.step()
```

一句话：**reward model 训练时可以更新 LLM；belief network 训练时必须 freeze reward LLM。你的方法核心就是 post-hoc contextual Bayesian aggregation，而不是同步训练 router 和 experts。**

* * *

**You:**

ok,那就分阶段训练吧。你再写一下更新belief network的公式，我检查一下

* * *

**ChatGPT:**

对，下面给一版**严格对应“两阶段训练、reward ensemble 冻结、只更新 belief network”**的公式。

---

## 1. 基本设定

对第 $i$ 个 supervised PRM marker，prefix 记为：

$$c_i=(x_i,s_{\le t})$$

已经训练好的 frozen ensemble PRM 有 $M$ 个：

$$f_1,\dots,f_M$$

第 $m$ 个 reward model 输出：

$$\mu_{i,m}=f_m(c_i)$$

其中：

$$\mu_{i,m}\in(0,1)$$

表示第 $m$ 个 PRM 对 prefix success probability / step correctness 的预测。

引入离散 latent variable：

$$z_i\in\{1,\dots,M\}$$

表示当前 prefix 应该相信哪个 frozen reward model。

belief network 输出：

$$q_\phi(z_i=m\mid c_i)=w_{i,m}$$

并且：

$$\sum_{m=1}^{M}w_{i,m}=1$$

推理时用的就是这个 $w_{i,m}$。

---

## 2. Generative model

先验设为：

$$p(z_i=m)=\omega_m$$

通常可以取 uniform prior：

$$\omega_m=\frac{1}{M}$$

如果监督是 VisualPRM400K / BetaPRM 风格的 count label，即 $K_i$ 个 successful continuations、总数 $N_i$，那么给定 expert $m$ 后：

$$K_i\mid c_i,z_i=m \sim \operatorname{Binomial}(N_i,\mu_{i,m})$$

所以第 $m$ 个 frozen PRM 的 log-likelihood 是：

$$\ell_{i,m}  
=  
\log p(K_i\mid c_i,z_i=m)$$ $$=  
\log {N_i\choose K_i}  
+  
K_i\log \mu_{i,m}  
+  
(N_i-K_i)\log(1-\mu_{i,m})$$

因为 $\log {N_i\choose K_i}$ 对所有 $m$ 一样，并且不影响 belief network 的相对权重训练，所以实际实现里可以省略：

$$\ell_{i,m}  
=  
K_i\log \mu_{i,m}  
+  
(N_i-K_i)\log(1-\mu_{i,m})$$

BetaPRM 也是强调这种 $K,N$ count supervision，因为 $K/N$ 只是 Monte Carlo success probability 的有限样本估计；它用 Beta-Binomial likelihood 避免把 $K/N$ 当作精确点标签。

Li 等 - 2026 - Process Rewards w…

---

## 3. ELBO

真实 evidence 是：

$$\log p(K_i\mid c_i)  
=  
\log  
\sum_{m=1}^{M}  
p(z_i=m)p(K_i\mid c_i,z_i=m)$$ $$=  
\log  
\sum_{m=1}^{M}  
\omega_m\exp(\ell_{i,m})$$

引入 belief network $q_\phi(z_i\mid c_i)$，ELBO 为：

$$\log p(K_i\mid c_i)  
\ge  
\mathcal{E}_i(\phi)$$

其中：

$$\mathcal{E}_i(\phi)  
=  
\mathbb{E}_{q_\phi(z_i\mid c_i)}  
\left[  
\log p(K_i,z_i\mid c_i)  
-  
\log q_\phi(z_i\mid c_i)  
\right]$$

展开：

$$\mathcal{E}_i(\phi)  
=  
\sum_{m=1}^{M}  
w_{i,m}  
\left[  
\ell_{i,m}  
+  
\log \omega_m  
-  
\log w_{i,m}  
\right]$$

也可以写成：

$$\mathcal{E}_i(\phi)  
=  
\sum_{m=1}^{M}  
w_{i,m}\ell_{i,m}  
-  
\mathrm{KL}  
\left(  
q_\phi(z_i\mid c_i)  
\|p(z_i)  
\right)$$

---

## 4. Belief network 的优化 loss

最大化 ELBO 等价于最小化 negative ELBO：

$$\mathcal{L}_{\text{belief}}  
=  
-  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\mathcal{E}_i(\phi)$$

即：

$$\mathcal{L}_{\text{belief}}  
=  
-  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\sum_{m=1}^{M}  
w_{i,m}\ell_{i,m}  
+  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\mathrm{KL}  
\left(  
q_\phi(z_i\mid c_i)  
\|p(z_i)  
\right)$$

如果使用 uniform prior：

$$p(z_i=m)=\frac{1}{M}$$

则：

$$\mathrm{KL}  
\left(  
q_\phi(z_i\mid c_i)  
\|p(z_i)  
\right)  
=  
\sum_{m=1}^{M}  
w_{i,m}\log(Mw_{i,m})$$

所以完整 loss 是：

$$\boxed{  
\mathcal{L}_{\text{belief}}  
=  
-  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\sum_{m=1}^{M}  
w_{i,m}  
\left[  
K_i\log \mu_{i,m}  
+  
(N_i-K_i)\log(1-\mu_{i,m})  
\right]  
+  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\sum_{m=1}^{M}  
w_{i,m}\log(Mw_{i,m})  
}$$

这里：

$$w_{i,m}=q_\phi(z_i=m\mid c_i)$$

是 belief network 输出，只有 $\phi$ 更新；$\mu_{i,m}$ 来自 frozen reward model，不更新。

---

## 5. 带系数版本

实际训练里可以加 KL 系数：

$$\boxed{  
\mathcal{L}_{\text{belief}}  
=  
-  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\sum_{m=1}^{M}  
w_{i,m}\ell_{i,m}  
+  
\beta_{\mathrm{KL}}  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\mathrm{KL}  
\left(  
q_\phi(z_i\mid c_i)  
\|p(z_i)  
\right)  
}$$

其中：

$$\ell_{i,m}  
=  
K_i\log \mu_{i,m}  
+  
(N_i-K_i)\log(1-\mu_{i,m})$$

严格的 ELBO 对应：

$$\beta_{\mathrm{KL}}=1$$

如果你把 $\beta_{\mathrm{KL}}$ 调成别的值，那就是 **$\beta$-ELBO / regularized ELBO**，论文里要这么叫。

---

## 6. 等价 posterior matching 形式

这个 ELBO 还有一个很重要的等价形式。

由 Bayes rule，给定 frozen ensemble 后，每个样本的真实 posterior responsibility 是：

$$\rho_{i,m}  
=  
p(z_i=m\mid K_i,c_i)$$ $$=  
\frac{  
\omega_m\exp(\ell_{i,m})  
}{  
\sum_{j=1}^{M}  
\omega_j\exp(\ell_{i,j})  
}$$

也就是：

$$\rho_{i,m}  
=  
\operatorname{softmax}_m  
\left(  
\ell_{i,m}+\log\omega_m  
\right)$$

标准 ELBO 满足：

$$\log p(K_i\mid c_i)  
-  
\mathcal{E}_i(\phi)  
=  
\mathrm{KL}  
\left(  
q_\phi(z_i\mid c_i)  
\|  
p(z_i\mid K_i,c_i)  
\right)$$

也就是：

$$\log p(K_i\mid c_i)  
-  
\mathcal{E}_i(\phi)  
=  
\mathrm{KL}  
\left(  
w_i  
\|  
\rho_i  
\right)$$

所以训练 belief network，本质上是在让：

$$q_\phi(z_i\mid c_i)  
\approx  
\rho_i$$

但注意：$\rho_i$ 训练时可以用 $K_i$ 算，推理时不能用；推理时只能用已经训练好的 $q_\phi(z\mid c)$。

---

## 7. 参数更新公式

belief network 参数更新就是：

$$\phi  
\leftarrow  
\phi  
-  
\eta  
\nabla_\phi  
\mathcal{L}_{\text{belief}}$$

因为 reward models 冻结，所以：

$$\nabla_{\theta_m}  
\mathcal{L}_{\text{belief}}  
=  
0$$

只有：

$$\nabla_{\phi}  
\mathcal{L}_{\text{belief}}  
\neq 0$$

如果写得更展开一点：

$$\nabla_\phi \mathcal{L}_i  
=  
\sum_{m=1}^{M}  
\left[  
-\ell_{i,m}  
+  
\beta_{\mathrm{KL}}  
\left(  
\log\frac{w_{i,m}}{\omega_m}  
+  
1  
\right)  
\right]  
\nabla_\phi w_{i,m}$$

其中：

$$w_{i,m}  
=  
\operatorname{softmax}(g_\phi(c_i))_m$$

所以梯度会推动 belief network 给 log-likelihood 更高的 frozen PRM 更大权重，同时 KL 项限制它不要过度 collapse。

---

## 8. Binary label 是特例

如果不是 count label，而是 binary step label：

$$y_i\in\{0,1\}$$

则：

$$K_i=y_i,\quad N_i=1$$

于是：

$$\ell_{i,m}  
=  
y_i\log \mu_{i,m}  
+  
(1-y_i)\log(1-\mu_{i,m})$$

loss 保持不变：

$$\mathcal{L}_{\text{belief}}  
=  
-  
\sum_m w_{i,m}\ell_{i,m}  
+  
\beta_{\mathrm{KL}}  
\mathrm{KL}  
\left(  
q_\phi(z_i\mid c_i)  
\|p(z_i)  
\right)$$

ActivePRM 的 ensemble 主要用 mean/std 来估计 uncertainty，而不是训练这种 $q_\phi(z\mid c)$ 的 posterior aggregator；它明确用 ensemble prediction 的均值和标准差判断 uncertain steps。

Duan 等 - Efficient Process Rewa…

---

最终你这版方法的核心公式可以压缩成一句：

$$\boxed{  
\max_\phi  
\sum_{i\in\mathcal{P}}  
\left[  
\sum_{m=1}^{M}  
q_\phi(m\mid c_i)  
\log p(D_i\mid c_i,z_i=m)  
-  
\mathrm{KL}  
\left(  
q_\phi(z_i\mid c_i)  
\|p(z_i)  
\right)  
\right]  
}$$

其中 $D_i=(K_i,N_i)$ 或 $D_i=y_i$，且所有 reward model $f_m$ 冻结，只更新 belief network $\phi$。

* * *

**You:**

写成带beta系数版本的吧

* * *

**ChatGPT:**

可以，带 $\beta_{\mathrm{KL}}$ 系数的版本建议这样写。

## 1. Frozen reward ensemble

第 $i$ 个 PRM marker 的 prefix：

$$c_i=(x_i,s_{\le t})$$

冻结的 ensemble reward models：

$$f_1,\dots,f_M$$

第 $m$ 个 reward model 输出：

$$\mu_{i,m}=f_m(c_i),\quad \mu_{i,m}\in(0,1)$$

注意：这里 $\mu_{i,m}$ 不参与梯度更新。

belief network 输出：

$$w_{i,m}  
=  
q_\phi(z_i=m\mid c_i)  
=  
\operatorname{softmax}(g_\phi(c_i))_m$$

其中：

$$\sum_{m=1}^{M}w_{i,m}=1$$

---

## 2. Count label likelihood

如果监督信号是 count label：

$$D_i=(K_i,N_i)$$

其中 $K_i$ 是成功 rollout 数，$N_i$ 是总 rollout 数，则第 $m$ 个 frozen reward model 的 log-likelihood 为：

$$\ell_{i,m}  
=  
\log p(K_i\mid c_i,z_i=m)$$ $$=  
K_i\log \mu_{i,m}  
+  
(N_i-K_i)\log(1-\mu_{i,m})$$

这里省略了组合数项：

$$\log {N_i\choose K_i}$$

因为它对所有 $m$ 相同，不影响 belief network 的相对权重学习。

---

## 3. 带 $\beta_{\mathrm{KL}}$ 的 ELBO objective

设 prior 为：

$$p(z_i=m)=\omega_m$$

通常取 uniform prior：

$$\omega_m=\frac{1}{M}$$

带 $\beta_{\mathrm{KL}}$ 的 objective 是：

$$\mathcal{J}_{\beta}(\phi)  
=  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\left[  
\sum_{m=1}^{M}  
w_{i,m}\ell_{i,m}  
-  
\beta_{\mathrm{KL}}  
\mathrm{KL}  
\left(  
q_\phi(z_i\mid c_i)  
\|p(z_i)  
\right)  
\right]$$

其中：

$$\mathrm{KL}  
\left(  
q_\phi(z_i\mid c_i)  
\|p(z_i)  
\right)  
=  
\sum_{m=1}^{M}  
w_{i,m}  
\log  
\frac{w_{i,m}}{\omega_m}$$

如果 prior 是 uniform：

$$\mathrm{KL}  
=  
\sum_{m=1}^{M}  
w_{i,m}\log(Mw_{i,m})$$

所以最大化目标可以写成：

$$\boxed{  
\max_{\phi}  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\left[  
\sum_{m=1}^{M}  
w_{i,m}  
\left(  
K_i\log \mu_{i,m}  
+  
(N_i-K_i)\log(1-\mu_{i,m})  
\right)  
-  
\beta_{\mathrm{KL}}  
\sum_{m=1}^{M}  
w_{i,m}\log(Mw_{i,m})  
\right]  
}$$

---

## 4. 对应 minimization loss

实际代码里一般最小化 negative objective：

$$\boxed{  
\mathcal{L}_{\text{belief}}  
=  
-  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\sum_{m=1}^{M}  
w_{i,m}  
\left[  
K_i\log \mu_{i,m}  
+  
(N_i-K_i)\log(1-\mu_{i,m})  
\right]  
+  
\beta_{\mathrm{KL}}  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\sum_{m=1}^{M}  
w_{i,m}\log(Mw_{i,m})  
}$$

这个就是最核心的 belief network 更新公式。

---

## 5. 参数更新

只更新 belief network：

$$\phi  
\leftarrow  
\phi  
-  
\eta  
\nabla_\phi  
\mathcal{L}_{\text{belief}}$$

冻结 reward ensemble：

$$\nabla_{\theta_m}  
\mathcal{L}_{\text{belief}}=0,  
\quad m=1,\dots,M$$

也就是说：

$$\mu_{i,m}=f_{\theta_m^\star}(c_i)$$

是固定 likelihood source，belief network 只学习：

$$c_i\mapsto q_\phi(z_i\mid c_i)$$

---

## 6. Binary label 特例

如果监督是 binary step label：

$$y_i\in\{0,1\}$$

则：

$$K_i=y_i,\quad N_i=1$$

所以：

$$\ell_{i,m}  
=  
y_i\log \mu_{i,m}  
+  
(1-y_i)\log(1-\mu_{i,m})$$

loss 变成：

$$\mathcal{L}_{\text{belief}}  
=  
-  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\sum_{m=1}^{M}  
w_{i,m}  
\left[  
y_i\log \mu_{i,m}  
+  
(1-y_i)\log(1-\mu_{i,m})  
\right]  
+  
\beta_{\mathrm{KL}}  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\sum_{m=1}^{M}  
w_{i,m}\log(Mw_{i,m})$$

---

## 7. $\beta_{\mathrm{KL}}$ 的作用

严格 ELBO 对应：

$$\beta_{\mathrm{KL}}=1$$

如果：

$$\beta_{\mathrm{KL}}<1$$

belief network 会更偏向选择 likelihood 高的 expert，权重更尖锐。

如果：

$$\beta_{\mathrm{KL}}>1$$

belief network 会更接近 prior，权重更平滑，避免过度 collapse。

所以论文里最好写成：

$$\beta_{\mathrm{KL}}$$

而不是只写 $\beta$，避免和 BetaPRM 里的 Beta distribution 混淆。

* * *

**You:**

ok，那接下来就具体实现一下。现在更新的代码库在BaPRM的地址，请你先拉去最新的代码看是否有问题。然后我想达到的效果是：训练beyesianPRM的时候有一个开关，决定是否加载ensemble checkpoint，不加载就从头先训练ensemble，训练流程和checkpoint保存和之前一样，然后再训练belief network。否则直接加载最新的checkpoint训练belief network。你先进行详细的步骤规划，并double check，列出你的规划，我之后会根据你的规划让codex进行实现

* * *

**ChatGPT:**

我先说明一下代码检查范围：我这里本地 `git clone` 因为容器 DNS 解析不到 GitHub 失败了，所以没有办法本地跑 `py_compile` 或 smoke test；但我已经通过 GitHub raw/main 查看了当前公开仓库的关键文件。仓库主页显示当前是 `zq2r/BaPRM` main 分支，包含 `shell/`、`src/internvl/`、`eval/` 等目录以及 15 次提交。[GitHub](https://github.com/zq2r/BaPRM)

从当前代码看，**ensemble PRM 的基础链路已经有了**：训练脚本 `visualprm400k_train_ensemble_prm.sh` 已经支持 `--prm_loss_type ensemble_prm`、`--ensemble_prm_num_heads`、`--ensemble_prm_hidden_dim`、`--ensemble_prm_dropout`，并且保留了原来的 checkpoint / resume / WandB 逻辑。[GitHub](https://github.com/zq2r/BaPRM/raw/refs/heads/main/shell/scripts/visualprm400k_train_ensemble_prm.sh) 模型里也已经有 `EnsembleScalarRewardHead`，它用共享 hidden state 经过多个独立 scalar heads 输出 ensemble logits，并提供 sigmoid 后的 mean/std。[GitHub](https://raw.githubusercontent.com/zq2r/BaPRM/main/src/internvl/model/internvl_chat/ensemble_prm_head.py) 当前 forward 里 `ensemble_prm` 分支会取 `<prm>` 位置 hidden states，用 ensemble head 做 BCE loss，并记录 `ensemble_prm_loss`、`ensemble_reward_mean/std` 等统计。[GitHub](https://raw.githubusercontent.com/zq2r/BaPRM/main/src/internvl/model/internvl_chat/modeling_internvl_chat_beta_binom.py)

但目前还没有 BayesianPRM 需要的几个关键组件：没有 `bayesian_prm` loss mode；没有 belief network；没有“加载 ensemble checkpoint 后冻结 reward ensemble，只训练 belief network”的阶段；也没有你要的总控开关。

下面是我建议给 Codex 的详细实现规划。

---

## 总体目标

新增一个两阶段训练流程：

$$\text{Stage 1: train ensemble PRM}$$ $$\text{Stage 2: freeze ensemble PRM, train belief network}$$

用户通过一个开关控制是否跳过 Stage 1：

```Bash
LOAD_ENSEMBLE_CHECKPOINT=0
```

表示不加载已有 ensemble checkpoint，从头训练 ensemble PRM，训练和 checkpoint 保存逻辑沿用现在的 `visualprm400k_train_ensemble_prm.sh`。

```Bash
LOAD_ENSEMBLE_CHECKPOINT=1
```

表示直接寻找或指定已有 ensemble checkpoint，然后只训练 belief network。

---

## 需要新增的入口脚本

建议新增：

```
shell/scripts/visualprm400k_train_bayesian_prm.sh
```

这个脚本只负责 orchestration，不要把所有逻辑硬塞进现有 ensemble 脚本。

建议核心变量：

```Bash
LOAD_ENSEMBLE_CHECKPOINT=${LOAD_ENSEMBLE_CHECKPOINT:-0}

ENSEMBLE_OUTPUT_DIR=${ENSEMBLE_OUTPUT_DIR:-"${REPO_ROOT}/log/ensemble-${model_name}-visualprm400k"}
ENSEMBLE_CHECKPOINT=${ENSEMBLE_CHECKPOINT:-""}

BAYESIAN_OUTPUT_DIR=${BAYESIAN_OUTPUT_DIR:-"${REPO_ROOT}/log/bayesian-${model_name}-visualprm400k"}

RESUME_ENSEMBLE_TRAINING=${RESUME_ENSEMBLE_TRAINING:-0}
RESUME_BELIEF_TRAINING=${RESUME_BELIEF_TRAINING:-0}
```

逻辑：

```Bash
if [ "${LOAD_ENSEMBLE_CHECKPOINT}" = "0" ]; then
    # 从头或按 RESUME_ENSEMBLE_TRAINING 训练 ensemble
    OUTPUT_DIR="${ENSEMBLE_OUTPUT_DIR}" \
    RESUME_TRAINING="${RESUME_ENSEMBLE_TRAINING}" \
    bash shell/scripts/visualprm400k_train_ensemble_prm.sh
fi

# 找 ensemble checkpoint
if [ -z "${ENSEMBLE_CHECKPOINT}" ]; then
    ENSEMBLE_CHECKPOINT=$(find "${ENSEMBLE_OUTPUT_DIR}" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
fi

if [ -z "${ENSEMBLE_CHECKPOINT}" ]; then
    echo "ERROR: no ensemble checkpoint found"
    exit 1
fi

# 用 ENSEMBLE_CHECKPOINT 训练 belief network
torchrun ... internvl_chat_finetune_beta_binom.py \
    --model_name_or_path "${ENSEMBLE_CHECKPOINT}" \
    --output_dir "${BAYESIAN_OUTPUT_DIR}" \
    --prm_loss_type bayesian_prm \
    ...
```

这里 `model_name_or_path` 在 Stage 2 必须指向 ensemble checkpoint，而不是 base model。因为 belief 阶段需要加载已经训练好的 LLM、projector 和 `ensemble_prm_head`。

---

## 修改 ModelArguments / config 参数

当前训练脚本已经有 `prm_loss_type`，帮助文本是 `beta_binom, normal_prm, or ensemble_prm`，并且已经有 ensemble 相关参数。[GitHub](https://raw.githubusercontent.com/zq2r/BaPRM/main/src/internvl/train/internvl_chat_finetune_beta_binom.py) 需要扩展为：

```Python
prm_loss_type: str = field(
    default="beta_binom",
    metadata={"help": "PRM loss type: beta_binom, normal_prm, ensemble_prm, or bayesian_prm."},
)
```

新增 belief 参数：

```Python
belief_hidden_dim: int = 256
belief_dropout: float = 0.0
belief_beta_kl: float = 0.1
belief_use_reward_probs: bool = True
belief_loglik_normalize_by_n: bool = True
belief_prior_type: str = "uniform"
```

其中 `belief_beta_kl` 就是你前面公式里的 $\beta_{\mathrm{KL}}$。

建议默认：

```
belief_beta_kl = 0.1
belief_loglik_normalize_by_n = True
belief_use_reward_probs = True
```

原因是如果直接用 raw count likelihood，$N=16$ 时 likelihood 项尺度可能明显大于 KL 项，早期容易让 belief 权重变尖。先用除以 $N$ 的 normalized log-likelihood 更稳。论文公式可以保留原始 count likelihood，代码里把 normalize 做成开关。

---

## 新增 belief head 模块

建议新增文件：

```
src/internvl/model/internvl_chat/bayesian_prm_head.py
```

模块名：

```Python
class BayesianBeliefHead(nn.Module):
```

输入有两种模式：

### 模式 A：只用 hidden state

$$w_i = q_\phi(z_i\mid h_i)$$

输入 shape：

```Python
prm_h: [P, H]
```

输出：

```Python
belief_logits: [P, E]
belief_weights: [P, E]
```

### 模式 B：hidden state + ensemble probabilities

更推荐：

$$w_i = q_\phi(z_i\mid h_i, \mu_{i,1},\dots,\mu_{i,E})$$

输入拼接：

```Python
belief_input = torch.cat([prm_h, ensemble_probs.transpose(0, 1)], dim=-1)
```

其中：

```Python
prm_h: [P, H]
ensemble_probs: [E, P]
ensemble_probs.transpose(0, 1): [P, E]
```

belief head 结构：

```Python
LayerNorm(input_dim)
Linear(input_dim, belief_hidden_dim)
GELU
Dropout
Linear(belief_hidden_dim, num_heads)
```

注意这里的 `num_heads` 必须等于 `ensemble_prm_num_heads`。

---

## 修改 InternVLChatModel 初始化逻辑

当前模型只有在：

```Python
self.prm_loss_type == "ensemble_prm"
```

时初始化 `ensemble_prm_head`。[GitHub](https://raw.githubusercontent.com/zq2r/BaPRM/main/src/internvl/model/internvl_chat/modeling_internvl_chat_beta_binom.py) 需要改成：

```Python
if self.prm_loss_type in ("ensemble_prm", "bayesian_prm"):
    self.init_ensemble_prm_head()

if self.prm_loss_type == "bayesian_prm":
    self.init_belief_head()
```

并新增：

```Python
self.belief_head = None
```

以及：

```Python
def init_belief_head(self, force_reinit: bool = False):
    ...
```

需要确保从 ensemble checkpoint 加载时，checkpoint 里没有 `belief_head` 也不会失败。也就是说，加载 ensemble checkpoint 进入 `bayesian_prm` 模式时，`belief_head` 是新初始化的 missing keys，这是正常情况。

---

## Stage 2 必须冻结 reward ensemble

在训练脚本里，当：

```Python
model_args.prm_loss_type == "bayesian_prm"
```

加载完模型并初始化 belief head 后，必须执行：

```Python
for name, p in model.named_parameters():
    p.requires_grad = False

for p in model.belief_head.parameters():
    p.requires_grad = True
```

不要只依赖 `--freeze_llm True --freeze_mlp True --freeze_backbone True`，因为那样可能还会漏掉 `ensemble_prm_head` 或其他模块。Bayesian 阶段应该强制全冻结，然后只打开 belief head。

训练日志里必须打印 trainable parameters，确认只包含：

```
belief_head.*
```

不应该出现：

```
language_model.*
mlp1.*
vision_model.*
ensemble_prm_head.*
kappa_head.*
```

这一步是 double check 的重点。

---

## 修改 forward：新增 bayesian_prm 分支

当前 forward 已经能取 `<prm>` hidden states，并且 ensemble 分支中使用：

```Python
prm_h = last_h[flat_prm_mask]
ensemble_logits = self.ensemble_prm_head(prm_h)
target = selected_labels
cls_loss = BCEWithLogits(ensemble_logits, target)
```

需要新增：

```Python
use_bayesian_prm = getattr(self, "prm_loss_type", "beta_binom") == "bayesian_prm"
```

并且 `need_prm_hidden` 条件也要包含：

```Python
or getattr(self, "prm_loss_type", "beta_binom") in ("ensemble_prm", "bayesian_prm")
```

Bayesian 分支逻辑：

```Python
selected_k = prm_counts_k.contiguous().view(-1)[placeholder_mask]
selected_n = prm_counts_n.contiguous().view(-1)[placeholder_mask]

valid = (selected_n > 0) & (selected_k >= 0) & (selected_k <= selected_n)

prm_h = prm_h[valid]
selected_k = selected_k[valid]
selected_n = selected_n[valid]
```

然后冻结 ensemble prediction：

```Python
with torch.no_grad():
    ensemble_logits = self.ensemble_prm_head(prm_h.detach())   # [E, P]
    ensemble_probs = torch.sigmoid(ensemble_logits.float()).clamp(eps, 1 - eps)
```

belief weights：

```Python
belief_logits = self.belief_head(prm_h.detach(), ensemble_probs)
weights = torch.softmax(belief_logits.float(), dim=-1)  # [P, E]
```

注意 shape 对齐：

```Python
mu = ensemble_probs.transpose(0, 1)  # [P, E]
```

count log-likelihood：

```Python
log_lik = (
    selected_k[:, None] * torch.log(mu)
    + (selected_n[:, None] - selected_k[:, None]) * torch.log(1 - mu)
)
```

如果启用 normalize：

```Python
log_lik = log_lik / selected_n[:, None].clamp_min(1.0)
```

KL to uniform prior：

```Python
log_w = torch.log(weights.clamp_min(eps))
kl = (weights * (log_w + math.log(num_heads))).sum(dim=-1)
```

loss：

```Python
expected_log_lik = (weights * log_lik).sum(dim=-1)
loss = (-expected_log_lik + belief_beta_kl * kl).mean()
```

记录 stats：

```Python
self._beta_last_stats = {
    "bayesian_belief_loss": ...,
    "bayesian_expected_loglik": ...,
    "bayesian_kl": ...,
    "bayesian_entropy": ...,
    "bayesian_weight_max": ...,
    "bayesian_weight_min": ...,
    "bayesian_weight_top1_mean": ...,
    "bayesian_reward_mean": ...,
    "bayesian_reward_std": ...,
    "valid_prm_count": ...,
}
```

这里不要更新 ensemble head，所以 `ensemble_logits` 必须 no-grad 或 detach。

---

## belief loss 对应公式

Codex 实现时对应这个 loss：

$$\mathcal{L}_{\text{belief}}  
=  
-\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\sum_{m=1}^{M}  
w_{i,m}  
\left[  
K_i\log \mu_{i,m}  
+  
(N_i-K_i)\log(1-\mu_{i,m})  
\right]  
+  
\beta_{\mathrm{KL}}  
\frac{1}{|\mathcal{P}|}  
\sum_{i\in\mathcal{P}}  
\sum_{m=1}^{M}  
w_{i,m}\log(Mw_{i,m})$$

其中：

$$w_{i,m}=q_\phi(z_i=m\mid c_i)$$ $$\mu_{i,m}=f_m(c_i)$$

并且 $\mu_{i,m}$ 来自 frozen ensemble PRM，不参与梯度更新。

---

## 修改 callback 逻辑

当前有 `BetaBinomStatsCallback` 和 `PRMStatsCallback`。`PRMStatsCallback` 只是把 `model._beta_last_stats` 注入日志，不会重置 kappa head。[GitHub](https://raw.githubusercontent.com/zq2r/BaPRM/main/src/internvl/train/internvl_chat_finetune_beta_binom.py)

BayesianPRM 阶段应该使用 `PRMStatsCallback`，不能使用 `BetaBinomStatsCallback`。否则会出现无意义的 kappa reset 或 beta-binom 相关日志。

逻辑应为：

```Python
if prm_loss_type == "beta_binom":
    callbacks.append(BetaBinomStatsCallback())
else:
    callbacks.append(PRMStatsCallback())
```

包括：

```Python
ensemble_prm
bayesian_prm
normal_prm
```

都走 `PRMStatsCallback`。

---

## checkpoint 保存策略

Stage 1 ensemble：

```
log/ensemble-${model_name}-visualprm400k/checkpoint-*
```

保持现有逻辑不变。当前 ensemble 脚本已经有 `OUTPUT_DIR`、`RESUME_TRAINING`、`save_steps`、`save_total_limit` 等逻辑。[GitHub](https://github.com/zq2r/BaPRM/raw/refs/heads/main/shell/scripts/visualprm400k_train_ensemble_prm.sh)

Stage 2 Bayesian belief：

```
log/bayesian-${model_name}-visualprm400k/checkpoint-*
```

保存完整模型 config + belief head 参数。因为 reward ensemble frozen 但仍作为模型一部分保存，最省事的方案是整个 model 保存。之后 evaluation 可以直接 load bayesian checkpoint。

建议额外写一个 metadata 文件：

```
bayesian_training_meta.json
```

内容包括：

```JSON
{
  "ensemble_checkpoint": ".../log/ensemble-.../checkpoint-xxx",
  "belief_beta_kl": 0.1,
  "belief_use_reward_probs": true,
  "belief_loglik_normalize_by_n": true,
  "ensemble_prm_num_heads": 5
}
```

这个不影响训练，但后面 debug 很有用。

---

## 对当前代码的几个需要 double check 的点

第一，**raw GitHub 显示 shell 和 Python 文件里存在超长行 / 换行异常**。例如 ensemble 训练脚本在 raw view 里只有 6 行，而且很多命令和注释被压在同一行。[GitHub](https://github.com/zq2r/BaPRM/raw/refs/heads/main/shell/scripts/visualprm400k_train_ensemble_prm.sh) 这可能只是网页解析问题，但 Codex 实现前必须在本地跑：

```Bash
bash -n shell/scripts/visualprm400k_train_ensemble_prm.sh
python -m py_compile src/internvl/model/internvl_chat/modeling_internvl_chat_beta_binom.py
python -m py_compile src/internvl/train/internvl_chat_finetune_beta_binom.py
```

如果真的是文件换行损坏，需要先修复格式，否则后面改 BayesianPRM 会很难定位错误。

第二，**当前 ensemble loss 用的是 ratio soft label，而不是 count likelihood**。这对 Stage 1 ensemble PRM 是可以接受的；但 Stage 2 belief training 应该使用 `prm_counts_k/n`，因为当前 preprocess 已经会构造 ratio labels、`prm_counts_k`、`prm_counts_n` 三个通道。[GitHub](https://raw.githubusercontent.com/zq2r/BaPRM/main/src/internvl/train/internvl_chat_finetune_beta_binom.py)

第三，**Bayesian 阶段不要让 LLM 参与梯度**。即使脚本传了 `--freeze_llm True`，也建议代码里在 `prm_loss_type == bayesian_prm` 时强制所有参数 `requires_grad=False`，再只打开 `belief_head`。

第四，**`need_prm_hidden` 必须包含 bayesian_prm**。否则 Stage 2 forward 可能拿不到 `<prm>` hidden state。

第五，**`init_ensemble_prm_head` 必须在 bayesian_prm 下执行**。否则加载 ensemble checkpoint 时没有对应模块，checkpoint 里的 `ensemble_prm_head.*` 会加载不上。

第六，**belief head 要在 optimizer 构建前初始化**。否则 Trainer optimizer 不会看到 `belief_head` 参数。

---

## 最小 smoke test 规划

Codex 实现后，先不要直接大规模跑。按这个顺序测：

### 1. 语法检查

```Bash
PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" python -m py_compile \
  src/internvl/model/internvl_chat/bayesian_prm_head.py \
  src/internvl/model/internvl_chat/modeling_internvl_chat_beta_binom.py \
  src/internvl/train/internvl_chat_finetune_beta_binom.py
```

### 2. 只加载已有 ensemble checkpoint 训练 belief

```Bash
LOAD_ENSEMBLE_CHECKPOINT=1 \
ENSEMBLE_CHECKPOINT=/path/to/ensemble/checkpoint-xxx \
BAYESIAN_MAX_STEPS=2 \
bash shell/scripts/visualprm400k_train_bayesian_prm.sh
```

重点看日志：

```
PRM_LOSS_TYPE: bayesian_prm
Loaded ensemble checkpoint: ...
Freeze reward ensemble: true
Trainable parameters:
belief_head.*
```

不能出现：

```
language_model.*
ensemble_prm_head.*
vision_model.*
mlp1.*
```

### 3. 检查 WandB / log stats

应该出现：

```
train/bayesian_belief_loss
train/bayesian_expected_loglik
train/bayesian_kl
train/bayesian_entropy
train/bayesian_weight_top1_mean
train/valid_prm_count
```

### 4. 检查 checkpoint

确认：

```Bash
ls log/bayesian-${model_name}-visualprm400k/checkpoint-*
```

并检查 state dict 里有：

```
belief_head.*
ensemble_prm_head.*
```

### 5. 再测试完整两阶段

```Bash
LOAD_ENSEMBLE_CHECKPOINT=0 \
RESUME_ENSEMBLE_TRAINING=0 \
bash shell/scripts/visualprm400k_train_bayesian_prm.sh
```

预期流程：

```
Stage 1: train ensemble PRM
Stage 1 done, latest checkpoint found
Stage 2: train belief network from ensemble checkpoint
```

---

## 这次 Codex 的实现边界

这次建议只实现 **训练链路**，不要同时改 evaluator。也就是先做到：

1. 可以从头训练 ensemble；
2. 可以加载 ensemble checkpoint；
3. 可以冻结 ensemble；
4. 可以训练 belief network；
5. 可以保存 bayesian checkpoint；
6. 日志确认 loss 和权重正常。

BayesianPRM 的 evaluator 可以下一步再做，把 ensemble evaluator 改成：

$$r_i=\sum_m w_{i,m}\mu_{i,m}$$

以及：

$$u_i=\sum_m w_{i,m}(\mu_{i,m}-r_i)^2$$

但不要和这次训练实现混在一起，否则 debug 面太大。