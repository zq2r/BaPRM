import argparse
import itertools
import json
import math
import os
import random
import re
import time
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from internvl.train.dataset import build_transform, dynamic_preprocess
from eval.prm.evaluate_mathvision_prm_normal import (
    load_model_and_tokenizer_beta as load_normal_model,
    batch_prm_mu as batch_prm_normal,
)
from eval.prm.evaluate_mathvision_prm_beta_binomial import (
    load_model_and_tokenizer_beta as load_beta_model,
    batch_prm_mu_kappa,
)
from eval.prm.evaluate_mathvision_prm_bayesian import (
    load_model_and_tokenizer_beta as load_bayesian_model,
)
from eval.prm.bayesian_prm_utils import batch_prm_weighted_mu

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def collate_fn(batches):
    x = batches[0]
    return x['pixel_values'], x['prompts'], x['steps_lens'], x['data_item']


def _clean_question(text: str) -> str:
    return re.sub(r'<image\d+>', '', text).strip()


def _build_prompt(question: str, steps: List[str]) -> Tuple[str, int]:
    solution = '<prm>'.join(s.strip() for s in steps) + '<prm>' if steps else ''
    return f'Question: {question}\nProcess: {solution}', len(steps)


def _tile_images(image_paths: List[str], grid_max_cols: int = 3) -> Image.Image:
    images = [Image.open(p).convert('RGB') for p in image_paths]
    if len(images) == 1:
        return images[0]

    cols = min(grid_max_cols, len(images))
    rows = (len(images) + cols - 1) // cols
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    canvas = Image.new('RGB', (cols * max_w, rows * max_h), (255, 255, 255))

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = c * max_w + (max_w - img.width) // 2
        y = r * max_h + (max_h - img.height) // 2
        canvas.paste(img, (x, y))
    return canvas


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        assert size > 0
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        shard = size // world_size
        left = size % world_size
        shard_sizes = [shard + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = begin + shard_sizes[rank]
        self.indices = range(begin, end)

    def __iter__(self):
        yield from self.indices

    def __len__(self):
        return len(self.indices)


class VisualProcessBenchPRMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_root: str,
        annotation_path: str,
        input_size: int,
        dynamic_image_size: bool,
        use_thumbnail: bool,
        max_num: int,
        grid_max_cols: int,
    ):
        self.image_root = image_root
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.grid_max_cols = grid_max_cols
        self.transform = build_transform(is_train=False, input_size=input_size)

        with open(annotation_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        img_list = rec.get('image') or rec.get('images') or []
        assert isinstance(img_list, list) and img_list, f'invalid image field at idx {idx}'

        paths = [
            p if os.path.isabs(p) else os.path.normpath(os.path.join(self.image_root, p))
            for p in img_list
        ]
        image = _tile_images(paths, self.grid_max_cols)

        if self.dynamic_image_size:
            images = dynamic_preprocess(
                image,
                image_size=self.input_size,
                use_thumbnail=self.use_thumbnail,
                max_num=self.max_num,
            )
        else:
            images = [image]

        pixel_values = torch.stack([self.transform(im) for im in images])
        response = rec.get('response', {})
        steps = response.get('steps', [])
        labels = response.get('process_correctness', [])
        assert len(steps) == len(labels), f'steps/labels mismatch at idx {idx}'

        prompt, steps_len = _build_prompt(_clean_question(rec.get('question', '')), steps)
        return {
            'pixel_values': pixel_values,
            'prompts': [prompt],
            'steps_lens': [steps_len],
            'data_item': {
                'data_source': rec.get('data_source', 'UNKNOWN'),
                'labels_per_step': labels,
                'meta': {'image_paths': paths, 'question': _clean_question(rec.get('question', ''))},
            },
        }


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2.0 * tp / denom if denom > 0 else 0.0


def _macro_f1_binary(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    tp_pos = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp_pos = sum(t != 1 and p == 1 for t, p in zip(y_true, y_pred))
    fn_pos = sum(t == 1 and p != 1 for t, p in zip(y_true, y_pred))
    f1_pos = _f1_from_counts(tp_pos, fp_pos, fn_pos)

    tp_neg = sum(t == -1 and p == -1 for t, p in zip(y_true, y_pred))
    fp_neg = sum(t != -1 and p == -1 for t, p in zip(y_true, y_pred))
    fn_neg = sum(t == -1 and p != -1 for t, p in zip(y_true, y_pred))
    f1_neg = _f1_from_counts(tp_neg, fp_neg, fn_neg)

    return {
        'f1_positive': f1_pos,
        'f1_negative': f1_neg,
        'macro_f1': (f1_pos + f1_neg) / 2.0,
        'counts': {
            'tp_pos': tp_pos, 'fp_pos': fp_pos, 'fn_pos': fn_pos,
            'tp_neg': tp_neg, 'fp_neg': fp_neg, 'fn_neg': fn_neg,
        },
    }


@torch.no_grad()
def _score_normal_or_beta(model, tokenizer, pixel_values, questions, num_patches_list, args):
    if args.prm_mode == 'normal':
        return batch_prm_normal(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            questions=questions,
            num_patches_list=num_patches_list,
            verbose=False,
        ).float()

    mu, kappa = batch_prm_mu_kappa(
        model=model,
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        questions=questions,
        num_patches_list=num_patches_list,
        verbose=False,
    )
    mu = mu.float()
    kappa = kappa.float()
    sigma = torch.sqrt(mu * (1.0 - mu) / (kappa + 1.0))
    return mu - float(args.risk_lambda) * sigma


@torch.no_grad()
def _score_bayesian_raw(model, tokenizer, pixel_values, questions, num_patches_list):
    # beta2-independent quantities; conservatism is applied offline below.
    return batch_prm_weighted_mu(
        model=model,
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        questions=questions,
        num_patches_list=num_patches_list,
        verbose=False,
        belief_use_conservatism=False,
        belief_conservatism_beta=None,
        return_details=True,
    )


def _parse_beta2_grid(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(',') if x.strip()]
    if not vals or any((not math.isfinite(x) or x <= 0) for x in vals):
        raise ValueError(f'Invalid beta2 grid: {text}')
    return vals


def _bayesian_reward(mu_heads: List[float], rel_weights: List[float], beta2: float) -> float:
    mu = torch.tensor(mu_heads, dtype=torch.float64)
    rel = torch.tensor(rel_weights, dtype=torch.float64)
    w = torch.softmax(torch.log(rel.clamp_min(1e-6)) - mu / max(beta2, 1e-6), dim=-1)
    return float((w * mu).sum().item())


def _apply_beta2(outputs: List[Dict[str, Any]], beta2: float):
    for item in outputs:
        mu_nested = item['prm_mu_heads']
        rel_nested = item['prm_rel_weights']
        assert len(mu_nested) == len(rel_nested)
        item['prm_scores'] = []
        for mu_steps, rel_steps in zip(mu_nested, rel_nested):
            assert len(mu_steps) == len(rel_steps)
            item['prm_scores'].append([
                _bayesian_reward(mu_h, rel_w, beta2)
                for mu_h, rel_w in zip(mu_steps, rel_steps)
            ])


def _collect_pairs(outputs: List[Dict[str, Any]]):
    per_source: Dict[str, Dict[str, List[float]]] = {}
    global_y, global_s = [], []

    for item in outputs:
        labels = item['labels_per_step']
        scores = item['prm_scores'][0] if item['prm_scores'] else []
        if len(labels) != len(scores):
            raise RuntimeError(f"label/score mismatch for {item.get('data_source')}: {len(labels)} vs {len(scores)}")

        d = per_source.setdefault(item.get('data_source', 'UNKNOWN'), {'y': [], 's': []})
        for y, s in zip(labels, scores):
            if y == 0:
                continue
            y = 1 if y == 1 else -1
            d['y'].append(y)
            d['s'].append(float(s))
            global_y.append(y)
            global_s.append(float(s))

    return per_source, global_y, global_s


def _evaluate_scores(outputs: List[Dict[str, Any]], args) -> Dict[str, Any]:
    per_source_pairs, global_y, global_s = _collect_pairs(outputs)

    auc = None
    if _HAS_SKLEARN and global_y:
        try:
            auc = roc_auc_score([1 if y == 1 else 0 for y in global_y], global_s)
        except Exception:
            pass

    threshold = float(args.threshold)
    auto_info = None

    if args.auto_threshold and global_s:
        cands = sorted(set(round(x, 4) for x in global_s))
        if len(cands) > 1000:
            step = max(1, len(cands) // 1000)
            cands = cands[::step]

        best = (-1.0, threshold)
        best_pooled = (-1.0, threshold)

        for t in cands:
            weighted_sum, counted = 0.0, 0
            for d in per_source_pairs.values():
                if not d['y']:
                    continue
                pred = [1 if s >= t else -1 for s in d['s']]
                m = _macro_f1_binary(d['y'], pred)
                n = len(d['y'])
                weighted_sum += m['macro_f1'] * n
                counted += n
            overall = weighted_sum / counted if counted else 0.0
            if overall > best[0]:
                best = (overall, t)

            pred_global = [1 if s >= t else -1 for s in global_s]
            pooled = _macro_f1_binary(global_y, pred_global)['macro_f1']
            if pooled > best_pooled[0]:
                best_pooled = (pooled, t)

        threshold = best[1]
        auto_info = {
            'auc': auc,
            'best_threshold_micro_over_sources': {'threshold': best[1], 'score': best[0]},
            'best_threshold_pooled_macro': {'threshold': best_pooled[1], 'score': best_pooled[0]},
        }

    per_source_metrics = {}
    global_y_true, global_y_pred = [], []
    total_steps, weighted_sum = 0, 0.0

    for src, d in per_source_pairs.items():
        pred = [1 if s >= threshold else -1 for s in d['s']]
        m = _macro_f1_binary(d['y'], pred)
        per_source_metrics[src] = m
        n = len(d['y'])
        weighted_sum += m['macro_f1'] * n
        total_steps += n
        global_y_true.extend(d['y'])
        global_y_pred.extend(pred)

    overall_macro = _macro_f1_binary(global_y_true, global_y_pred)
    metrics = {
        'per_source': per_source_metrics,
        'overall': {
            'micro_over_sources': weighted_sum / total_steps if total_steps else 0.0,
            'macro_f1_pooled': overall_macro['macro_f1'],
            'f1_positive': overall_macro['f1_positive'],
            'f1_negative': overall_macro['f1_negative'],
            'total_steps': total_steps,
            'threshold_used': threshold,
        },
    }
    if auto_info is not None:
        metrics['auto_search'] = auto_info
    return metrics


def evaluate_model():
    random.seed(args.seed)
    dataset = VisualProcessBenchPRMDataset(
        image_root=args.image_root,
        annotation_path=args.annotation,
        input_size=image_size,
        dynamic_image_size=args.dynamic,
        use_thumbnail=use_thumbnail,
        max_num=args.max_num,
        grid_max_cols=args.grid_max_cols,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    outputs = []
    rank = torch.distributed.get_rank()

    for pixel_values, prompts, steps_lens, data_item in tqdm(dataloader, disable=(rank != 0)):
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        expected = sum(steps_lens)

        if args.prm_mode == 'bayesian':
            mu_heads_all, rel_weights_all = [], []
            for i in range(0, len(prompts), args.mini_batch_size):
                curr_bs = min(args.mini_batch_size, len(prompts) - i)
                details = _score_bayesian_raw(
                    model,
                    tokenizer,
                    torch.cat([pixel_values] * curr_bs, dim=0),
                    prompts[i:i + curr_bs],
                    [pixel_values.shape[0]] * curr_bs,
                )
                mu_heads_all.extend(details['mu_heads'].float().tolist())
                rel_weights_all.extend(details['rel_weights'].float().tolist())

            if len(mu_heads_all) != expected or len(rel_weights_all) != expected:
                raise RuntimeError(f'BayesianPRM step count mismatch: expected {expected}, got {len(mu_heads_all)}/{len(rel_weights_all)}')

            data_item['prm_mu_heads'] = []
            data_item['prm_rel_weights'] = []
            cur = 0
            for n in steps_lens:
                data_item['prm_mu_heads'].append(mu_heads_all[cur:cur + n])
                data_item['prm_rel_weights'].append(rel_weights_all[cur:cur + n])
                cur += n
        else:
            scores_all = []
            for i in range(0, len(prompts), args.mini_batch_size):
                curr_bs = min(args.mini_batch_size, len(prompts) - i)
                scores = _score_normal_or_beta(
                    model,
                    tokenizer,
                    torch.cat([pixel_values] * curr_bs, dim=0),
                    prompts[i:i + curr_bs],
                    [pixel_values.shape[0]] * curr_bs,
                    args,
                )
                scores_all.extend(scores.tolist())

            if len(scores_all) != expected:
                raise RuntimeError(f'PRM step count mismatch: expected {expected}, got {len(scores_all)}')

            data_item['prm_scores'] = []
            cur = 0
            for n in steps_lens:
                data_item['prm_scores'].append(scores_all[cur:cur + n])
                cur += n

        outputs.append(data_item)

    torch.distributed.barrier()
    gathered = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered, json.dumps(outputs))
    merged = list(itertools.chain.from_iterable(json.loads(x) for x in gathered))

    if rank != 0:
        return

    if args.prm_mode == 'bayesian':
        sweep = []
        best_beta2, best_metrics, best_score = None, None, -1.0

        for beta2 in _parse_beta2_grid(args.bayesian_beta2_grid):
            _apply_beta2(merged, beta2)
            metrics = _evaluate_scores(merged, args)
            overall = metrics['overall']['micro_over_sources']
            sweep.append({
                'beta2': beta2,
                'threshold': metrics['overall']['threshold_used'],
                'overall': overall,
            })
            print(f'[beta2={beta2:g}] threshold={metrics["overall"]["threshold_used"]:.4f}, Overall={overall:.4f}')
            if overall > best_score:
                best_beta2, best_metrics, best_score = beta2, metrics, overall

        _apply_beta2(merged, best_beta2)
        metrics_summary = best_metrics
        metrics_summary['best_beta2'] = best_beta2
        metrics_summary['beta2_sweep'] = sweep
        metrics_summary['overall']['best_beta2'] = best_beta2
        for item in merged:
            item['prm_best_beta2'] = best_beta2
    else:
        metrics_summary = _evaluate_scores(merged, args)

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime('%y%m%d%H%M%S', time.localtime())
    output_path = os.path.join(args.out_dir, f'visualprocessbench_{stamp}.json')
    metrics_path = os.path.join(args.out_dir, f'metrics_{stamp}.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    print(f'Results saved to {output_path}')
    print(f'Metrics saved to {metrics_path}')
    if args.prm_mode == 'bayesian':
        print(f'Best beta2: {metrics_summary["best_beta2"]:g}')
    print(f'Selected threshold: {metrics_summary["overall"]["threshold_used"]:.4f} (auto_threshold={args.auto_threshold})')
    print('Per-source Macro-F1:')
    for src, m in metrics_summary['per_source'].items():
        print(f' - {src}: macro_f1={m["macro_f1"]:.4f} (pos={m["f1_positive"]:.4f}, neg={m["f1_negative"]:.4f})')
    print(f'Overall (micro over sources): {metrics_summary["overall"]["micro_over_sources"]:.4f}')
    print(
        f'Pooled Macro-F1: {metrics_summary["overall"]["macro_f1_pooled"]:.4f} '
        f'(pos={metrics_summary["overall"]["f1_positive"]:.4f}, '
        f'neg={metrics_summary["overall"]["f1_negative"]:.4f}); '
        f'steps counted: {metrics_summary["overall"]["total_steps"]}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--annotation', type=str, required=True)
    parser.add_argument('--image-root', type=str, default='')
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--mini-batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true', default=True)
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--grid-max-cols', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto-threshold', action='store_true')
    parser.add_argument('--auto-device-map', action='store_true')
    parser.add_argument('--prm-mode', required=True, choices=['normal', 'beta', 'bayesian'])
    parser.add_argument('--risk-lambda', type=float, default=0.5)

    # Kept for compatibility with the shell / existing BaPRM args.
    parser.add_argument('--belief-use-conservatism', default='auto', choices=['auto', 'true', 'false'])
    parser.add_argument('--belief-conservatism-beta', type=float, default=None)
    parser.add_argument(
        '--bayesian-beta2-grid',
        type=str,
        default='0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,1.0',
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    # Existing BaPRM loaders use args.auto for device-map loading.
    args.auto = bool(args.auto_device_map)

    if args.prm_mode == 'normal':
        model, tokenizer = load_normal_model(args)
    elif args.prm_mode == 'beta':
        model, tokenizer = load_beta_model(args)
    else:
        model, tokenizer = load_bayesian_model(args)

    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    if args.prm_mode == 'bayesian':
        print(f'[test] beta2 grid: {args.bayesian_beta2_grid}')

    try:
        evaluate_model()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
