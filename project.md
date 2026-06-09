[2026-06-09 05:25:20,002] [INFO] [partition_parameters.py:366:__exit__] finished initializing model - num_params = 695, num_elems = 7.95B
Loading checkpoint shards: 100%|██████████| 4/4 [00:10<00:00,  2.65s/it]
Loading checkpoint shards: 100%|██████████| 4/4 [00:10<00:00,  2.66s/it]
[rank1]: Traceback (most recent call last):
[rank1]:   File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/src/internvl/train/internvl_chat_finetune_beta_binom.py", line 1982, in <module>
[rank1]:     main()
[rank1]:   File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/src/internvl/train/internvl_chat_finetune_beta_binom.py", line 1714, in main
[rank1]:     reinit_ensemble_heads_except_one(model, keep_idx=keep_idx)
[rank1]:   File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/src/internvl/train/internvl_chat_finetune_beta_binom.py", line 227, in reinit_ensemble_heads_except_one
[rank1]:     torch.nn.init.xavier_uniform_(head.w1[i])
[rank1]:                                   ~~~~~~~^^^
[rank1]: IndexError: index 1 is out of bounds for dimension 0 with size 0
Loading checkpoint shards: 100%|██████████| 4/4 [00:10<00:00,  2.66s/it]
[INFO|modeling_utils.py:4849] 2026-06-09 05:25:30,682 >> All model checkpoint weights were used when initializing InternVLChatModel.

[INFO|modeling_utils.py:4857] 2026-06-09 05:25:30,682 >> All the weights of InternVLChatModel were initialized from the model checkpoint at /inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/log/ensemble-InternVL3-8B-visualprm400k/checkpoint-1103.
If your task is similar to the task the model of the checkpoint was trained on, you can already use InternVLChatModel for predictions without further training.
[INFO|configuration_utils.py:1093] 2026-06-09 05:25:30,685 >> loading configuration file /inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/log/ensemble-InternVL3-8B-visualprm400k/checkpoint-1103/generation_config.json
Loading checkpoint shards: 100%|██████████| 4/4 [00:10<00:00,  2.66s/it]
[INFO|configuration_utils.py:1140] 2026-06-09 05:25:30,685 >> Generate config GenerationConfig {}

[rank2]: Traceback (most recent call last):
[rank2]:   File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/src/internvl/train/internvl_chat_finetune_beta_binom.py", line 1982, in <module>
[rank2]:     main()
[rank2]:   File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/src/internvl/train/internvl_chat_finetune_beta_binom.py", line 1714, in main
[rank2]:     reinit_ensemble_heads_except_one(model, keep_idx=keep_idx)
[rank2]:   File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/src/internvl/train/internvl_chat_finetune_beta_binom.py", line 227, in reinit_ensemble_heads_except_one
[rank2]:     torch.nn.init.xavier_uniform_(head.w1[i])
[rank2]:                                   ~~~~~~~^^^
[rank2]: IndexError: index 1 is out of bounds for dimension 0 with size 0
[rank3]: Traceback (most recent call last):
[rank3]:   File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/src/internvl/train/internvl_chat_finetune_beta_binom.py", line 1982, in <module>
[rank3]:     main()
[rank3]:   File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/src/internvl/train/internvl_chat_finetune_beta_binom.py", line 1714, in main
[rank3]:     reinit_ensemble_heads_except_one(model, keep_idx=keep_idx)
[rank3]:   File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/src/internvl/train/internvl_chat_finetune_beta_binom.py", line 227, in reinit_ensemble_heads_except_one
[rank3]:     torch.nn.init.xavier_uniform_(head.w1[i])
[rank3]:                                   ~~~~~~~^^^
[rank3]: IndexError: index 1 is out of bounds for dimension 0 with size 0
[rank0]: Traceback (most recent call last):
[rank0]:   File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/src/internvl/train/internvl_chat_finetune_beta_binom.py", line 1982, in <module>
[rank0]:     main()
[rank0]:   File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/src/internvl/train/internvl_chat_finetune_beta_binom.py", line 1714, in main
[rank0]:     reinit_ensemble_heads_except_one(model, keep_idx=keep_idx)
[rank0]:   File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/src/internvl/train/internvl_chat_finetune_beta_binom.py", line 227, in reinit_ensemble_heads_except_one
[rank0]:     torch.nn.init.xavier_uniform_(head.w1[i])
[rank0]:                                   ~~~~~~~^^^
[rank0]: IndexError: index 1 is out of bounds for dimension 0 with size 0
[rank0]:[W609 05:25:31.677477124 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
W0609 05:25:32.813000 4155820 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 4163124 closing signal SIGTERM
W0609 05:25:32.814000 4155820 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 4163128 closing signal SIGTERM
W0609 05:25:32.814000 4155820 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 4163133 closing signal SIGTERM
E0609 05:25:33.079000 4155820 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 1 (pid: 4163126) of binary: /inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/miniconda3/envs/beta-prm/bin/python
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/miniconda3/envs/beta-prm/lib/python3.11/site-packages/torch/distributed/run.py", line 923, in <module>
    main()
  File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/miniconda3/envs/beta-prm/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/miniconda3/envs/beta-prm/lib/python3.11/site-packages/torch/distributed/run.py", line 919, in main
    run(args)
  File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/miniconda3/envs/beta-prm/lib/python3.11/site-packages/torch/distributed/run.py", line 910, in run
    elastic_launch(
  File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/miniconda3/envs/beta-prm/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/miniconda3/envs/beta-prm/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/src/internvl/train/internvl_chat_finetune_beta_binom.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-06-09_05:25:32
  host      : qzj-llm-run--8d8d0e7aa802-z57tv5juft
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 4163126)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
(beta-prm) [root:Beta-Binomial-PRM]$ 
已思考 12s

这个报错不是 ensemble head 真的没有 5 个 head，而是 DeepSpeed ZeRO-3 把参数切分了。

你现在直接访问：

head.w1[i]

但在 ZeRO-3 下，每个 rank 只持有参数的一部分。有些 rank 上 head.w1 的 local shard 第一维是 0，所以报：

IndexError: index 1 is out of bounds for dimension 0 with size 0

所以这个 debug reinit 函数不能直接切 head.w1[i]，要用 deepspeed.zero.GatheredParameters 先把完整参数 gather 出来。

改这个函数即可

文件：

src/internvl/train/internvl_chat_finetune_beta_binom.py

把你现在的：

def reinit_ensemble_heads_except_one(model, keep_idx: int = 0):
    ...

整段替换成下面这个版本：

def reinit_ensemble_heads_except_one(model, keep_idx: int = 0):
    head = model.ensemble_prm_head
    if head is None:
        raise RuntimeError("ensemble_prm_head is None; cannot reinit heads.")

    num_heads = int(head.num_heads)
    if keep_idx < 0 or keep_idx >= num_heads:
        raise ValueError(f"keep_idx={keep_idx} out of range [0, {num_heads})")

    params = [head.w1, head.b1, head.w2, head.b2]

    # Under DeepSpeed ZeRO-3, parameters are partitioned. Some ranks may see
    # local shards with shape [0, ...], so direct indexing like head.w1[i]
    # can fail. Gather full parameters before modifying selected heads.
    try:
        import deepspeed

        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                with torch.no_grad():
                    for i in range(num_heads):
                        if i == keep_idx:
                            continue

                        torch.nn.init.xavier_uniform_(head.w1.data[i])
                        torch.nn.init.xavier_uniform_(head.w2.data[i])
                        head.b1.data[i].zero_()
                        head.b2.data[i].zero_()

    except Exception as e:
        # Fallback for non-ZeRO or normal single-process training.
        # If this fallback fails under ZeRO-3, re-raise the original error.
        if head.w1.shape[0] == 0:
            raise RuntimeError(
                "Failed to gather ZeRO-3 partitioned ensemble_prm_head parameters "
                "before reinitialization."
            ) from e

        with torch.no_grad():
            for i in range(num_heads):
                if i == keep_idx:
                    continue

                torch.nn.init.xavier_uniform_(head.w1.data[i])
                torch.nn.init.xavier_uniform_(head.w2.data[i])
                head.b1.data[i].zero_()
                head.b2.data[i].zero_()
为什么这样能修

原来你是在每个 rank 上直接改本地 shard：

head.w1[i]

但 ZeRO-3 下某些 rank 的本地 shard 可能是：

head.w1.shape = [0]

所以 head.w1[1] 直接炸。

新的写法会先：

with deepspeed.zero.GatheredParameters(params, modifier_rank=0):

让完整参数在上下文里可访问，然后只在 rank 0 上执行 reinit。退出上下文后 DeepSpeed 会把修改后的参数重新分片。

再跑前先检查
python -m py_compile src/internvl/train/internvl_chat_finetune_beta_binom.py

然后重新跑你的 debug 命令。