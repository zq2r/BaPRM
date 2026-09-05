#!/usr/bin/env python3
import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path


def parse_int_grid(s):
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def mean_std(xs):
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.pstdev(xs))


def is_multiple_choice(question):
    q = str(question or '')
    if 'Choices:' in q or 'Options:' in q:
        return True
    hits = re.findall(r'(?m)^\s*[A-H]\s*[:\.\)]\s*\S+', q)
    return len(hits) >= 2


def strip_wrappers(answer):
    s = str(answer or '').strip()
    s = re.sub(r'</?answer>', '', s, flags=re.I).strip()
    for _ in range(3):
        m = re.fullmatch(r'\s*\\(?:boxed|fbox)\s*\{(.*)\}\s*', s, flags=re.S)
        if not m:
            break
        s = m.group(1).strip()
    s = re.sub(
        r'^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*',
        '', s, flags=re.I,
    ).strip()
    return s


def normalize_answer(answer, question):
    s = strip_wrappers(answer)

    if is_multiple_choice(question):
        patterns = [
            r'^\s*(?:option|choice)\s*([A-H])\b',
            r'^\s*\(?([A-H])\)?\s*(?:[:\.\),\-]|$)',
            r'^\s*([A-H])\s+',
        ]
        for pat in patterns:
            m = re.search(pat, s, flags=re.I)
            if m:
                return 'choice:' + m.group(1).lower()
        if re.fullmatch(r'[A-Ha-h]', s):
            return 'choice:' + s.lower()

    s = s.lower().replace('$', '')
    s = re.sub(r'\\(?:left|right)', '', s)
    s = re.sub(r'\s+', ' ', s).strip().rstrip(' .。')
    return 'text:' + s


def final_answer(solution_split):
    return str(solution_split[-1]).strip() if solution_split else ''


def build_permutations(num_items, max_n, seed, repeat):
    # Match evaluate_prm_tts.py: one RNG stream per repeat.
    rng = random.Random(seed + 1000003 * repeat)
    perms = []
    for _ in range(num_items):
        ids = list(range(max_n))
        rng.shuffle(ids)
        perms.append(ids)
    return perms


def evaluate_sc(data, perms, n):
    correct = 0
    ties = 0
    winning_votes = []

    for item, perm in zip(data, perms):
        ids = perm[:n]
        question = item.get('question') or item.get('query_cot') or ''
        keys = [
            normalize_answer(final_answer(item['solutions_splits'][i]), question)
            for i in ids
        ]
        counts = Counter(keys)
        max_votes = max(counts.values())
        winners = {k for k, v in counts.items() if v == max_votes}
        if len(winners) > 1:
            ties += 1

        # Tie-break by the first candidate in the already-randomized subset.
        local_idx = next(j for j, key in enumerate(keys) if key in winners)
        selected_idx = ids[local_idx]
        correct += int(int(item['labels'][selected_idx]) == 1)
        winning_votes.append(max_votes)

    total = len(data)
    return {
        'accuracy': correct / total,
        'correct': correct,
        'total': total,
        'tie_rate': ties / total,
        'mean_winning_votes': sum(winning_votes) / total,
    }


def evaluate_oracle(data, perms, n):
    correct = 0
    for item, perm in zip(data, perms):
        ids = perm[:n]
        correct += int(any(int(item['labels'][i]) == 1 for i in ids))
    return correct / len(data)


def validate(data, max_n):
    if not isinstance(data, list) or not data:
        raise ValueError('Input must be a non-empty JSON list.')
    for i, item in enumerate(data):
        sols = item.get('solutions_splits', [])
        labels = item.get('labels', [])
        if len(sols) < max_n:
            raise ValueError(f'item {i}: solutions_splits < {max_n}')
        if len(labels) < max_n:
            raise ValueError(f'item {i}: labels < {max_n}')
        for j in range(max_n):
            if not sols[j]:
                raise ValueError(f'item {i}, rollout {j}: empty solution')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-json', required=True)
    ap.add_argument('--output-json', required=True)
    ap.add_argument('--n-grid', default='1,2,4,8,16')
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    ns = parse_int_grid(args.n_grid)
    if not ns:
        raise ValueError('n-grid must not be empty')
    if args.repeats < 1:
        raise ValueError('repeats must be >= 1')

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    max_n = max(ns)
    validate(data, max_n)

    output = {
        'input_json': args.input_json,
        'method': 'self_consistency',
        'num_items': len(data),
        'n_grid': ns,
        'pool_max_n': max_n,
        'subset_mode': 'random',
        'repeats': args.repeats,
        'seed': args.seed,
        'tie_break': 'first_in_random_subset',
        'results': {},
    }

    for n in ns:
        rr = []
        for r in range(args.repeats):
            perms = build_permutations(len(data), max_n, args.seed, r)
            sc = evaluate_sc(data, perms, n)
            oracle = evaluate_oracle(data, perms, n)
            rr.append({'repeat': r, **sc, 'oracle_accuracy': oracle})

        acc_mean, acc_std = mean_std([x['accuracy'] for x in rr])
        oracle_mean, oracle_std = mean_std([x['oracle_accuracy'] for x in rr])
        tie_mean, tie_std = mean_std([x['tie_rate'] for x in rr])
        vote_mean, vote_std = mean_std([x['mean_winning_votes'] for x in rr])

        output['results'][str(n)] = {
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'oracle_pass_mean': oracle_mean,
            'oracle_pass_std': oracle_std,
            'tie_rate_mean': tie_mean,
            'tie_rate_std': tie_std,
            'mean_winning_votes': vote_mean,
            'mean_winning_votes_std': vote_std,
            'repeat_results': rr,
        }

        print(
            f'N={n:<2d} SC={acc_mean:.4f} ± {acc_std:.4f} '
            f'oracle={oracle_mean:.4f} tie={tie_mean:.4f} '
            f'win_votes={vote_mean:.3f}'
        )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
