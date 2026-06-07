那接下来的一步要干嘛

下一步先不要继续加新功能，先做一轮最小 smoke test，确认 ensemble 训练链路真的通了。

重点验证 4 件事：

1. ensemble_prm_head 能进入 trainable parameters
2. forward 真的走 ensemble_prm loss
3. WandB 能记录 ensemble stats
4. checkpoint 能被 ensemble evaluator 正常加载并评测
第一步：语法检查

在代码库根目录跑：

PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" python -m py_compile \
  src/internvl/model/internvl_chat/ensemble_prm_head.py \
  src/internvl/model/internvl_chat/modeling_internvl_chat_beta_binom.py \
  src/internvl/train/internvl_chat_finetune_beta_binom.py \
  eval/prm/evaluate_mathvision_prm_ensemble.py \
  eval/prm/evaluate_mathverse_prm_ensemble.py \
  eval/prm/evaluate_mathvista_prm_ensemble.py \
  eval/prm/evaluate_olympiadbench_prm_ensemble.py

这一步没报错，再继续。

第二步：启动 ensemble 训练，先看日志

先跑：

bash shell/scripts/visualprm400k_train_ensemble_prm.sh

启动后重点看有没有这些日志：

Using PRM loss type: ensemble_prm
Using ensemble PRM head: num_heads=8, hidden_dim=128, dropout=0.0
ensemble_prm mode: skip kappa_head reset and kappa_head gradient multiplier.
Ensemble PRM mode: add PRMStatsCallback before WandbCallback.

然后在 trainable parameters 里看有没有：

ensemble_prm_head.norm.weight
ensemble_prm_head.norm.bias
ensemble_prm_head.w1
ensemble_prm_head.b1
ensemble_prm_head.w2
ensemble_prm_head.b2

只要这些参数出现在 trainable list 里，就说明 head 已经被 optimizer 看到了。

第三步：看 WandB 是否记录 ensemble stats

训练到第一个 logging step 后，WandB 里应该有：

train/ensemble_prm_loss
train/ensemble_reward_mean
train/ensemble_reward_std
train/ensemble_target_mean
train/valid_prm_count

其中最关键的是：

train/ensemble_prm_loss
train/ensemble_reward_std

如果这两个出现，说明 forward 确实走了 ensemble 分支，并且 callback 顺序没问题。