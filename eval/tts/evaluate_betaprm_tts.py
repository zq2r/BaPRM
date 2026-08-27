import argparse
import json
import math
from pathlib import Path


DEFAULT_NS = [1, 2, 4, 8, 16]
DEFAULT_LAMBDAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5]
DEFAULT_QS = [0.7, 0.8, 0.9]


def safe_mean(xs):
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def quantile(xs, q):
    if not xs:
        return 0.0

    ys = sorted(float(x) for x in xs)
    q = min(max(float(q), 0.0), 1.0)

    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return float(ys[lo])

    w = pos - lo
    return float(
        ys[lo] * (1.0 - w)
        + ys[hi] * w
    )


def parse_int_grid(s):
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Empty int grid: {s}")
    return vals


def parse_float_grid(s):
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Empty float grid: {s}")
    return vals


def validate_data(data, max_n):
    if not isinstance(data, list):
        raise ValueError("Evaluator output must be a JSON list.")

    if not data:
        raise ValueError("Evaluator output is empty.")

    for idx, item in enumerate(data):
        labels = item.get("labels", [])
        mu = item.get("prm_mu", [])
        sigma = item.get("prm_sigma", [])

        if len(labels) < max_n:
            raise ValueError(
                f"item {idx}: only {len(labels)} labels, "
                f"but max N={max_n}"
            )

        if len(mu) < max_n:
            raise ValueError(
                f"item {idx}: only {len(mu)} prm_mu rollouts, "
                f"but max N={max_n}"
            )

        if len(sigma) < max_n:
            raise ValueError(
                f"item {idx}: only {len(sigma)} prm_sigma rollouts, "
                f"but max N={max_n}"
            )


def collect_step_sigmas(data, n):
    """
    Collect sigma from ONLY the first n rollouts of every problem.

    This means each N has its own global sigma distribution and therefore
    its own q-quantile threshold tau, matching the existing diagnosis logic
    applied to an N-sized rollout pool.
    """
    vals = []

    for item in data:
        sigma_rollouts = item["prm_sigma"][:n]

        for sigma_steps in sigma_rollouts:
            vals.extend(float(x) for x in sigma_steps)

    return vals


def score_risk_budget(mu_steps, sigma_steps, lam, tau):
    if not mu_steps:
        return -1e9

    if not sigma_steps:
        sigma_steps = [0.0] * len(mu_steps)

    risk_ratio = safe_mean(
        [1.0 if float(s) > tau else 0.0 for s in sigma_steps]
    )

    return (
        safe_mean([float(x) for x in mu_steps])
        - float(lam) * risk_ratio
    )


def evaluate_single_pass(data):
    """
    N=1 baseline.

    No PRM selection:
    simply evaluate the first rollout of every problem.
    """
    correct = 0
    total = 0

    for item in data:
        labels = item.get("labels", [])

        if not labels:
            continue

        correct += 1 if int(labels[0]) == 1 else 0
        total += 1

    accuracy = correct / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def evaluate_risk_budget(data, n, lam, tau):
    correct = 0
    total = 0

    for item in data:
        labels = item["labels"][:n]
        mu_rollouts = item["prm_mu"][:n]
        sigma_rollouts = item["prm_sigma"][:n]

        scores = []

        for i in range(n):
            score = score_risk_budget(
                mu_steps=mu_rollouts[i],
                sigma_steps=sigma_rollouts[i],
                lam=lam,
                tau=tau,
            )
            scores.append(score)

        best_idx = max(
            range(len(scores)),
            key=lambda i: scores[i],
        )

        correct += 1 if int(labels[best_idx]) == 1 else 0
        total += 1

    accuracy = correct / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def evaluate_mu_only(data, n):
    """
    Diagnostic only:
    BoN selection using mean(mu), equivalent to lambda=0.
    """
    correct = 0
    total = 0

    for item in data:
        labels = item["labels"][:n]
        mu_rollouts = item["prm_mu"][:n]

        scores = [
            safe_mean([float(x) for x in mu_steps])
            for mu_steps in mu_rollouts
        ]

        best_idx = max(
            range(len(scores)),
            key=lambda i: scores[i],
        )

        correct += 1 if int(labels[best_idx]) == 1 else 0
        total += 1

    accuracy = correct / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--n-grid",
        type=str,
        default="1,2,4,8,16",
    )

    parser.add_argument(
        "--lambdas",
        type=str,
        default="0,0.1,0.2,0.35,0.5,0.7,1.0,1.5",
    )

    parser.add_argument(
        "--budget-q-grid",
        type=str,
        default="0.7,0.8,0.9",
    )

    args = parser.parse_args()

    ns = parse_int_grid(args.n_grid)
    lambdas = parse_float_grid(args.lambdas)
    qs = parse_float_grid(args.budget_q_grid)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    validate_data(data, max(ns))

    output = {
        "input_json": args.input_json,
        "num_items": len(data),
        "n_grid": ns,
        "lambdas": lambdas,
        "budget_q_grid": qs,
        "results": {},
    }

    # =========================
    # N = 1
    # =========================
    single_pass = evaluate_single_pass(data)

    output["results"]["1"] = {
        "mode": "single_pass_no_prm",
        **single_pass,
    }

    print(
        f"N=1  "
        f"single-pass "
        f"acc={single_pass['accuracy']:.4f} "
        f"({single_pass['correct']}/{single_pass['total']})"
    )

    # =========================
    # N > 1
    # =========================
    for n in ns:
        if n == 1:
            continue

        all_step_sigmas = collect_step_sigmas(
            data=data,
            n=n,
        )

        sweep = []

        for q in qs:
            tau = quantile(all_step_sigmas, q)

            for lam in lambdas:
                result = evaluate_risk_budget(
                    data=data,
                    n=n,
                    lam=lam,
                    tau=tau,
                )

                sweep.append(
                    {
                        "lambda": lam,
                        "q": q,
                        "tau": tau,
                        **result,
                    }
                )

        best = max(
            sweep,
            key=lambda x: x["accuracy"],
        )

        mu_only = evaluate_mu_only(
            data=data,
            n=n,
        )

        output["results"][str(n)] = {
            "mode": "betaprm_risk_budget",
            "num_step_sigmas": len(all_step_sigmas),
            "mu_only": mu_only,
            "risk_budget_sweep": sweep,
            "best": best,
        }

        print(
            f"N={n:<2d} "
            f"BetaPRM "
            f"acc={best['accuracy']:.4f} "
            f"({best['correct']}/{best['total']}) "
            f"q={best['q']} "
            f"lambda={best['lambda']} "
            f"tau={best['tau']:.6f}"
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()