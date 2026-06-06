# 下一步：加 ensemble evaluator，先从 MathVision 开始

这一步先只改一个文件：

```Bash
eval/prm/evaluate_mathvision_prm_ensemble.py
```

不要动已有的：

```Bash
eval/prm/evaluate_mathvision_prm_beta_binomial.py
eval/prm/evaluate_mathvision_prm_normal.py
```

## 1. 复制 normal evaluator

```Bash
cp eval/prm/evaluate_mathvision_prm_normal.py \
   eval/prm/evaluate_mathvision_prm_ensemble.py
```

然后打开：

```Bash
vim eval/prm/evaluate_mathvision_prm_ensemble.py
```

---

## 2. 把 `batch_prm_mu` 改成 `batch_prm_ensemble`

找到：

```Python
@torch.no_grad()
def batch_prm_mu(
    model,
    tokenizer,
    pixel_values,
    questions,
    num_patches_list,
    verbose=False,
):
```

改成：

```Python
@torch.no_grad()
def batch_prm_ensemble(
    model,
    tokenizer,
    pixel_values,
    questions,
    num_patches_list,
    verbose=False,
):
```

---

## 3. 函数里 forward 要打开 hidden states

在这个函数里找到模型 forward 那段，normal 版本大概是：

```Python
outputs = model(
    pixel_values=pixel_values,
    input_ids=input_ids,
    attention_mask=attention_mask,
    image_flags=image_flags,
    return_dict=True,
)
```

改成：

```Python
outputs = model(
    pixel_values=pixel_values,
    input_ids=input_ids,
    attention_mask=attention_mask,
    image_flags=image_flags,
    output_hidden_states=True,
    return_dict=True,
)
```

但是注意：你模型 forward 里为了省显存，只有训练时 `need_prm_hidden` 那套逻辑会主动取 last hidden。评测这里我们直接要求 `output_hidden_states=True`，所以后面可以从：

```Python
hidden = outputs.hidden_states[-1]
```

取 hidden。

---

## 4. 替换原来的 `mu` 计算逻辑

在 normal evaluator 里，你现在应该有类似：

```Python
logits = outputs.logits
placeholder_mask = input_ids == prm_token_id
selected_logits = logits[placeholder_mask]
selected_logits = selected_logits[..., reward_token_ids]

if selected_logits.numel() == 0:
    return selected_logits.new_zeros((0,), dtype=torch.float32)

mu = torch.softmax(selected_logits.float(), dim=-1)[..., 0]
return mu
```

把这段替换成：

```Python
placeholder_mask = input_ids == prm_token_id

if outputs.hidden_states is None:
    raise RuntimeError(
        "ensemble PRM evaluation requires output_hidden_states=True."
    )

hidden = outputs.hidden_states[-1]          # [B, L, H]
prm_h = hidden[placeholder_mask]            # [M, H]

if prm_h.numel() == 0:
    empty = hidden.new_zeros((0,), dtype=torch.float32)
    return empty, empty

if not hasattr(model, "ensemble_prm_head") or model.ensemble_prm_head is None:
    raise RuntimeError(
        "ensemble_prm_head is not initialized. "
        "Please make sure the checkpoint is trained with prm_loss_type='ensemble_prm'."
    )

ensemble_logits = model.ensemble_prm_head(prm_h)            # [E, M]
ensemble_probs = torch.sigmoid(ensemble_logits.float())     # [E, M]

mu = ensemble_probs.mean(dim=0)                             # [M]
std = ensemble_probs.std(dim=0, unbiased=False)             # [M]

return mu, std
```

这里就是 ensemble PRM 的预测逻辑：

```
每个 head 输出一个 logit
sigmoid 后得到每个 head 的 reward probability
对 head 维度取平均，得到最终 reward
std 作为 ensemble uncertainty
```

---

## 5. 改主循环里的调用

找到原来的：

```Python
mu = batch_prm_mu(
    model=model,
    tokenizer=tokenizer,
    pixel_values=curr_pixel_values,
    questions=curr_questions,
    num_patches_list=curr_num_patches,
    verbose=False,
)

score = mu

prm_scores_flattened.extend(score.tolist())
prm_mu_flattened.extend(mu.tolist())
```

改成：

```Python
mu, std = batch_prm_ensemble(
    model=model,
    tokenizer=tokenizer,
    pixel_values=curr_pixel_values,
    questions=curr_questions,
    num_patches_list=curr_num_patches,
    verbose=False,
)

score = mu

prm_scores_flattened.extend(score.tolist())
prm_mu_flattened.extend(mu.tolist())
prm_ensemble_std_flattened.extend(std.tolist())
```

---

## 6. 加 `prm_ensemble_std_flattened`

找到：

```Python
prm_scores_flattened = []
prm_mu_flattened = []
```

改成：

```Python
prm_scores_flattened = []
prm_mu_flattened = []
prm_ensemble_std_flattened = []
```

---

## 7. 输出字段加 `prm_ensemble_std`

找到：

```Python
data_item['prm_scores'] = []
data_item['prm_mu'] = []
```

改成：

```Python
data_item['prm_scores'] = []
data_item['prm_mu'] = []
data_item['prm_ensemble_std'] = []
```

然后找到 append 部分：

```Python
data_item['prm_scores'].append(
    prm_scores_flattened[curr_len : curr_len + steps_lens[i]]
)
data_item['prm_mu'].append(
    prm_mu_flattened[curr_len : curr_len + steps_lens[i]]
)
```

改成：

```Python
data_item['prm_scores'].append(
    prm_scores_flattened[curr_len : curr_len + steps_lens[i]]
)
data_item['prm_mu'].append(
    prm_mu_flattened[curr_len : curr_len + steps_lens[i]]
)
data_item['prm_ensemble_std'].append(
    prm_ensemble_std_flattened[curr_len : curr_len + steps_lens[i]]
)
```

---

## 8. 检查有没有残留函数名

保存后跑：

```Bash
grep -n "batch_prm_mu" eval/prm/evaluate_mathvision_prm_ensemble.py
```

预期：没有输出。

再跑：

```Bash
grep -n "batch_prm_ensemble" eval/prm/evaluate_mathvision_prm_ensemble.py
```

应该能看到函数定义和调用。

---

## 9. 语法检查

```Bash
PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" python -m py_compile \
  eval/prm/evaluate_mathvision_prm_ensemble.py
```

---

这一步完成后，MathVision 的 ensemble evaluator 就有了。下一步再按同样方式改 `MathVerse / MathVista / OlympiadBench`，最后再把 `eval_prm.sh` 加上 `PRM_MODE=ensemble`。