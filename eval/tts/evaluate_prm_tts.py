import argparse
import json
import math
import random
import statistics
from pathlib import Path


def safe_mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def quantile(xs, q):
    if not xs:
        return 0.0

    ys = sorted(float(x) for x in xs)
    q = min(max(float(q), 0.0), 1.0)

    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return ys[lo]

    w = pos - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w


def parse_int_grid(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_grid(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def mean_std(xs):
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.pstdev(xs))


def validate_data(data, max_n, prm_mode):
    if not isinstance(data, list) or not data:
        raise ValueError("Input evaluator JSON must be a non-empty list.")

    for i, item in enumerate(data):
        if len(item.get("labels", [])) < max_n:
            raise ValueError(
                f"item {i}: labels < max N={max_n}"
            )

        if len(item.get("prm_mu", [])) < max_n:
            raise ValueError(
                f"item {i}: prm_mu < max N={max_n}"
            )

        if prm_mode == "beta":
            if len(item.get("prm_sigma", [])) < max_n:
                raise ValueError(
                    f"item {i}: prm_sigma < max N={max_n}"
                )


def build_permutations(data, max_n, seed, repeat, subset_mode):
    perms = []

    if subset_mode == "prefix":
        return [list(range(max_n)) for _ in data]

    # Independent deterministic permutation for each item/repeat.
    rng = random.Random(seed + 1000003 * repeat)

    for item in data:
        ids = list(range(max_n))
        rng.shuffle(ids)
        perms.append(ids)

    return perms


def aggregate_prm(mu_steps):
    # Standard PRM trajectory score: average step reward.
    return safe_mean([float(x) for x in mu_steps])


def evaluate_single_candidate(data, perms):
    correct = 0

    for item, perm in zip(data, perms):
        idx = perm[0]
        correct += int(int(item["labels"][idx]) == 1)

    return correct / len(data), correct


def evaluate_oracle(data, perms, n):
    correct = 0

    for item, perm in zip(data, perms):
        ids = perm[:n]
        labels = [int(item["labels"][i]) for i in ids]
        correct += int(any(x == 1 for x in labels))

    return correct / len(data), correct


def evaluate_standard_prm(data, perms, n):
    correct = 0

    for item, perm in zip(data, perms):
        ids = perm[:n]

        scores = [
            aggregate_prm(item["prm_mu"][i])
            for i in ids
        ]

        local_best = max(
            range(len(scores)),
            key=lambda j: scores[j],
        )

        selected_idx = ids[local_best]

        correct += int(
            int(item["labels"][selected_idx]) == 1
        )

    return correct / len(data), correct


def collect_beta_sigmas(data, perms, n):
    vals = []

    for item, perm in zip(data, perms):
        for idx in perm[:n]:
            vals.extend(
                float(x)
                for x in item["prm_sigma"][idx]
            )

    return vals


def beta_score(mu_steps, sigma_steps, lam, tau):
    if not mu_steps:
        return -1e9

    if not sigma_steps:
        sigma_steps = [0.0] * len(mu_steps)

    risk_ratio = safe_mean(
        [1.0 if float(s) > tau else 0.0 for s in sigma_steps]
    )

    return (
        aggregate_prm(mu_steps)
        - float(lam) * risk_ratio
    )


def evaluate_beta_setting(
    data,
    perms,
    n,
    lam,
    tau,
):
    correct = 0

    for item, perm in zip(data, perms):
        ids = perm[:n]

        scores = [
            beta_score(
                item["prm_mu"][i],
                item["prm_sigma"][i],
                lam,
                tau,
            )
            for i in ids
        ]

        local_best = max(
            range(len(scores)),
            key=lambda j: scores[j],
        )

        selected_idx = ids[local_best]

        correct += int(
            int(item["labels"][selected_idx]) == 1
        )

    return correct / len(data), correct


def evaluate_beta(
    data,
    perms,
    n,
    lambdas,
    qs,
):
    sigmas = collect_beta_sigmas(
        data,
        perms,
        n,
    )

    sweep = []

    for q in qs:
        tau = quantile(sigmas, q)

        for lam in lambdas:
            acc, correct = evaluate_beta_setting(
                data=data,
                perms=perms,
                n=n,
                lam=lam,
                tau=tau,
            )

            sweep.append(
                {
                    "q": q,
                    "lambda": lam,
                    "tau": tau,
                    "accuracy": acc,
                    "correct": correct,
                    "total": len(data),
                }
            )

    best = max(
        sweep,
        key=lambda x: x["accuracy"],
    )

    return best, sweep


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-json",
        required=True,
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    parser.add_argument(
        "--prm-mode",
        required=True,
        choices=[
            "beta",
            "normal",
            "ensemble",
            "bayesian",
        ],
    )

    parser.add_argument(
        "--n-grid",
        default="1,2,4,8,16",
    )

    parser.add_argument(
        "--subset-mode",
        choices=["prefix", "random"],
        default="random",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--lambdas",
        default="0,0.1,0.2,0.35,0.5,0.7,1.0,1.5",
    )

    parser.add_argument(
        "--budget-q-grid",
        default="0.7,0.8,0.9",
    )

    args = parser.parse_args()

    ns = parse_int_grid(args.n_grid)
    lambdas = parse_float_grid(args.lambdas)
    qs = parse_float_grid(args.budget_q_grid)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    max_n = max(ns)

    validate_data(
        data,
        max_n=max_n,
        prm_mode=args.prm_mode,
    )

    repeats = args.repeats

    if args.subset_mode == "prefix":
        repeats = 1

    output = {
        "input_json": args.input_json,
        "prm_mode": args.prm_mode,
        "num_items": len(data),
        "n_grid": ns,
        "subset_mode": args.subset_mode,
        "repeats": repeats,
        "seed": args.seed,
        "results": {},
    }

    for n in ns:
        repeat_results = []

        for r in range(repeats):
            perms = build_permutations(
                data=data,
                max_n=max_n,
                seed=args.seed,
                repeat=r,
                subset_mode=args.subset_mode,
            )

            oracle_acc, oracle_correct = evaluate_oracle(
                data,
                perms,
                n,
            )

            # N=1: no PRM selection.
            if n == 1:
                acc, correct = evaluate_single_candidate(
                    data,
                    perms,
                )

                repeat_result = {
                    "repeat": r,
                    "accuracy": acc,
                    "correct": correct,
                    "total": len(data),
                    "oracle_accuracy": oracle_acc,
                    "oracle_correct": oracle_correct,
                }

            elif args.prm_mode == "beta":
                best, sweep = evaluate_beta(
                    data=data,
                    perms=perms,
                    n=n,
                    lambdas=lambdas,
                    qs=qs,
                )

                repeat_result = {
                    "repeat": r,
                    "accuracy": best["accuracy"],
                    "correct": best["correct"],
                    "total": len(data),
                    "oracle_accuracy": oracle_acc,
                    "oracle_correct": oracle_correct,
                    "best": best,
                    "risk_budget_sweep": sweep,
                }

            else:
                acc, correct = evaluate_standard_prm(
                    data,
                    perms,
                    n,
                )

                repeat_result = {
                    "repeat": r,
                    "accuracy": acc,
                    "correct": correct,
                    "total": len(data),
                    "oracle_accuracy": oracle_acc,
                    "oracle_correct": oracle_correct,
                }

            repeat_results.append(repeat_result)

        accs = [
            x["accuracy"]
            for x in repeat_results
        ]

        oracle_accs = [
            x["oracle_accuracy"]
            for x in repeat_results
        ]

        acc_mean, acc_std = mean_std(accs)
        oracle_mean, oracle_std = mean_std(oracle_accs)

        output["results"][str(n)] = {
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "oracle_pass_mean": oracle_mean,
            "oracle_pass_std": oracle_std,
            "repeat_results": repeat_results,
        }

        if repeats == 1:
            rr = repeat_results[0]
            output["results"][str(n)]["accuracy"] = rr["accuracy"]
            output["results"][str(n)]["correct"] = rr["correct"]

        if args.prm_mode == "beta" and n > 1:
            configs = [
                x["best"]
                for x in repeat_results
            ]

            print(
                f"N={n:<2d} "
                f"acc={acc_mean:.4f} ± {acc_std:.4f} "
                f"oracle={oracle_mean:.4f} "
                f"best(first repeat): "
                f"q={configs[0]['q']} "
                f"lambda={configs[0]['lambda']} "
                f"tau={configs[0]['tau']:.6f}"
            )
        else:
            print(
                f"N={n:<2d} "
                f"acc={acc_mean:.4f} ± {acc_std:.4f} "
                f"oracle={oracle_mean:.4f}"
            )

    out = Path(args.output_json)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()