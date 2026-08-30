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


def validate_data(
    data,
    max_n,
    prm_mode,
    require_ias=False,
):
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

        if prm_mode == "bayesian":
            if len(
                item.get("prm_mu_heads", [])
            ) < max_n:
                raise ValueError(
                    f"item {i}: prm_mu_heads < max N={max_n}"
                )

            if len(
                item.get("prm_rel_weights", [])
            ) < max_n:
                raise ValueError(
                    f"item {i}: prm_rel_weights < max N={max_n}"
                )

        if require_ias:
            if "ias_mu" not in item:
                raise ValueError(
                    f"item {i}: missing ias_mu required by IAS"
                )
        if require_ias and prm_mode == "bayesian":
            if "ias_mu_heads" not in item:
                raise ValueError(
                    f"item {i}: missing ias_mu_heads "
                    "required by Bayesian IAS"
                )

            if "ias_rel_weights" not in item:
                raise ValueError(
                    f"item {i}: missing ias_rel_weights "
                    "required by Bayesian IAS"
                )

            if len(item["ias_mu_heads"]) != len(
                item["ias_rel_weights"]
            ):
                raise ValueError(
                    f"item {i}: IAS Bayesian head/reliability "
                    "dimensions do not match"
                )

            p = float(item["ias_mu"])
            if not math.isfinite(p):
                raise ValueError(
                    f"item {i}: invalid ias_mu={p}"
                )

            if not 0.0 <= p <= 1.0:
                raise ValueError(
                    f"item {i}: ias_mu must be in [0, 1], got {p}"
                )

def compute_ias_budget(
    p,
    confidence=0.99,
    max_n=16,
):
    p = float(p)
    confidence = float(confidence)
    max_n = int(max_n)

    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"IAS confidence must be in (0, 1), got {confidence}"
        )

    if max_n < 1:
        raise ValueError(
            f"IAS max_n must be >= 1, got {max_n}"
        )

    # Defensive clamp against tiny floating-point errors.
    p = min(max(p, 0.0), 1.0)

    if p <= 0.0:
        return max_n

    if p >= 1.0:
        return 1

    n = math.ceil(
        math.log1p(-confidence)
        / math.log1p(-p)
    )

    return min(max(n, 1), max_n)


def build_ias_budgets(
    data,
    confidence,
    max_n,
):
    return [
        compute_ias_budget(
            item["ias_mu"],
            confidence=confidence,
            max_n=max_n,
        )
        for item in data
    ]

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


def evaluate_oracle(data, perms, budgets):
    correct = 0

    for item, perm, n in zip(data, perms, budgets):
        ids = perm[:n]
        labels = [int(item["labels"][i]) for i in ids]
        correct += int(any(x == 1 for x in labels))

    return correct / len(data), correct


def evaluate_standard_prm(
    data,
    perms,
    budgets,
):
    correct = 0

    for item, perm, n in zip(data, perms, budgets):
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

def bayesian_step_reward(
    mu_heads,
    rel_weights,
    beta2,
):
    beta2 = float(beta2)

    if beta2 <= 0.0:
        raise ValueError(
            f"Bayesian beta2 must be > 0, got {beta2}"
        )

    if len(mu_heads) != len(rel_weights):
        raise ValueError(
            "mu_heads and rel_weights have different lengths: "
            f"{len(mu_heads)} vs {len(rel_weights)}"
        )

    if not mu_heads:
        raise ValueError(
            "Empty Bayesian ensemble vector."
        )

    mus = [float(x) for x in mu_heads]
    rels = [float(x) for x in rel_weights]

    if any(not math.isfinite(x) for x in mus):
        raise ValueError(
            f"Non-finite Bayesian head rewards: {mus}"
        )

    if any(
        (not math.isfinite(x)) or x < 0.0
        for x in rels
    ):
        raise ValueError(
            f"Invalid reliability weights: {rels}"
        )

    if sum(rels) <= 0.0:
        raise ValueError(
            "Reliability weights sum to zero."
        )

    # alpha_final_m ∝ alpha_rel_m * exp(-mu_m / beta2)
    log_weights = [
        math.log(max(alpha, 1e-30))
        - mu / beta2
        for mu, alpha in zip(mus, rels)
    ]

    # Numerically stable softmax.
    max_log_weight = max(log_weights)

    exp_weights = [
        math.exp(x - max_log_weight)
        for x in log_weights
    ]

    normalizer = sum(exp_weights)

    post_weights = [
        x / normalizer
        for x in exp_weights
    ]

    return sum(
        w * mu
        for w, mu in zip(post_weights, mus)
    )


def bayesian_trajectory_score(
    mu_head_steps,
    rel_weight_steps,
    beta2,
):
    if len(mu_head_steps) != len(rel_weight_steps):
        raise ValueError(
            "Bayesian trajectory has mismatched "
            "mu-head/reliability step counts."
        )

    if not mu_head_steps:
        return -1e9

    step_rewards = [
        bayesian_step_reward(
            mu_heads,
            rel_weights,
            beta2,
        )
        for mu_heads, rel_weights in zip(
            mu_head_steps,
            rel_weight_steps,
        )
    ]

    # Same trajectory aggregation as standard PRM:
    # average reward over reasoning steps.
    return aggregate_prm(step_rewards)


def evaluate_bayesian_setting(
    data,
    perms,
    budgets,
    beta2,
):
    correct = 0

    for item, perm, n in zip(
        data,
        perms,
        budgets,
    ):
        ids = perm[:n]

        scores = [
            bayesian_trajectory_score(
                item["prm_mu_heads"][i],
                item["prm_rel_weights"][i],
                beta2,
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


def evaluate_bayesian(
    data,
    perms,
    budgets,
    beta2s,
):
    sweep = []

    for beta2 in beta2s:
        acc, correct = evaluate_bayesian_setting(
            data=data,
            perms=perms,
            budgets=budgets,
            beta2=beta2,
        )

        sweep.append(
            {
                "beta2": beta2,
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


def build_bayesian_ias_budgets(
    data,
    beta2,
    confidence,
    max_n,
):
    probs = [
        bayesian_step_reward(
            item["ias_mu_heads"],
            item["ias_rel_weights"],
            beta2,
        )
        for item in data
    ]

    budgets = [
        compute_ias_budget(
            p,
            confidence=confidence,
            max_n=max_n,
        )
        for p in probs
    ]

    return budgets, probs


def evaluate_bayesian_ias(
    data,
    perms,
    beta2s,
    confidence,
    max_n,
):
    sweep = []

    for beta2 in beta2s:
        budgets, probs = (
            build_bayesian_ias_budgets(
                data=data,
                beta2=beta2,
                confidence=confidence,
                max_n=max_n,
            )
        )

        acc, correct = evaluate_bayesian_setting(
            data=data,
            perms=perms,
            budgets=budgets,
            beta2=beta2,
        )

        oracle_acc, oracle_correct = (
            evaluate_oracle(
                data,
                perms,
                budgets,
            )
        )

        average_n = safe_mean(budgets)

        histogram = {
            str(n): budgets.count(n)
            for n in range(1, max_n + 1)
            if budgets.count(n) > 0
        }

        sweep.append(
            {
                "beta2": beta2,
                "accuracy": acc,
                "correct": correct,
                "total": len(data),
                "oracle_accuracy": oracle_acc,
                "oracle_correct": oracle_correct,
                "average_n": average_n,
                "budget_ratio": average_n / max_n,
                "budget_histogram": histogram,
                "ias_prob_mean": safe_mean(probs),
            }
        )

    best = max(
        sweep,
        key=lambda x: x["accuracy"],
    )

    return best, sweep

def collect_beta_sigmas(
    data,
    perms,
    budgets,
):
    vals = []

    for item, perm, n in zip(data, perms, budgets):
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
    budgets,
    lam,
    tau,
):
    correct = 0

    for item, perm, n in zip(data, perms, budgets):
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
    budgets,
    lambdas,
    qs,
):
    sigmas = collect_beta_sigmas(
        data,
        perms,
        budgets,
    )

    sweep = []

    for q in qs:
        tau = quantile(sigmas, q)

        for lam in lambdas:
            acc, correct = evaluate_beta_setting(
                data=data,
                perms=perms,
                budgets=budgets,
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
        "--ias",
        action="store_true",
        help="Enable instance-adaptive BoN evaluation.",
    )

    parser.add_argument(
        "--ias-confidence",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--ias-max-n",
        type=int,
        default=None,
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
    
    parser.add_argument(
        "--bayesian-beta2-grid",
        default="0.05,0.1,0.2,0.3,0.5,1.0",
        help=(
            "Comma-separated beta2 grid for BayesianPRM "
            "inference-time conservatism sweep."
        ),
    )

    args = parser.parse_args()

    ns = parse_int_grid(args.n_grid)
    lambdas = parse_float_grid(args.lambdas)
    qs = parse_float_grid(args.budget_q_grid)
    beta2s = parse_float_grid(
        args.bayesian_beta2_grid
    )

    if args.prm_mode == "bayesian":
        if not beta2s:
            raise ValueError(
                "bayesian-beta2-grid must not be empty"
            )

        if any(beta2 <= 0.0 for beta2 in beta2s):
            raise ValueError(
                "Every Bayesian beta2 must be > 0, got "
                f"{beta2s}"
            )

    if not ns:
        raise ValueError("n-grid must not be empty")

    fixed_max_n = max(ns)

    ias_max_n = (
        args.ias_max_n
        if args.ias_max_n is not None
        else fixed_max_n
    )

    if args.ias and ias_max_n < 1:
        raise ValueError(
            f"ias-max-n must be >= 1, got {ias_max_n}"
        )

    pool_max_n = max(
        fixed_max_n,
        ias_max_n if args.ias else fixed_max_n,
    )

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    validate_data(
        data,
        max_n=pool_max_n,
        prm_mode=args.prm_mode,
        require_ias=args.ias,
    )

    repeats = args.repeats

    if args.subset_mode == "prefix":
        repeats = 1

    output = {
        "input_json": args.input_json,
        "prm_mode": args.prm_mode,
        "num_items": len(data),
        "n_grid": ns,
        "pool_max_n": pool_max_n,
        "subset_mode": args.subset_mode,
        "repeats": repeats,
        "seed": args.seed,
        "ias_enabled": args.ias,
        "results": {},
    }

    for n in ns:
        budgets = [n] * len(data)
        repeat_results = []

        for r in range(repeats):
            perms = build_permutations(
                data=data,
                max_n=pool_max_n,
                seed=args.seed,
                repeat=r,
                subset_mode=args.subset_mode,
            )

            oracle_acc, oracle_correct = evaluate_oracle(
                data,
                perms,
                budgets,
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
                    budgets=budgets,
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
                
            elif args.prm_mode == "bayesian":
                best, sweep = evaluate_bayesian(
                    data=data,
                    perms=perms,
                    budgets=budgets,
                    beta2s=beta2s,
                )

                repeat_result = {
                    "repeat": r,
                    "accuracy": best["accuracy"],
                    "correct": best["correct"],
                    "total": len(data),
                    "oracle_accuracy": oracle_acc,
                    "oracle_correct": oracle_correct,
                    "best": best,
                    "beta2_sweep": sweep,
                }

            else:
                acc, correct = evaluate_standard_prm(
                    data,
                    perms,
                    budgets,
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

        elif args.prm_mode == "bayesian" and n > 1:
            configs = [
                x["best"]
                for x in repeat_results
            ]

            output["results"][str(n)][
                "best_beta2_first_repeat"
            ] = configs[0]["beta2"]

            output["results"][str(n)][
                "beta2_grid"
            ] = beta2s

            print(
                f"N={n:<2d} "
                f"acc={acc_mean:.4f} ± {acc_std:.4f} "
                f"oracle={oracle_mean:.4f} "
                f"best(first repeat): "
                f"beta2={configs[0]['beta2']}"
            )

        else:
            print(
                f"N={n:<2d} "
                f"acc={acc_mean:.4f} ± {acc_std:.4f} "
                f"oracle={oracle_mean:.4f}"
            )

    if args.ias:
        # Normal/Beta use the evaluator-produced ias_mu directly.
        # Bayesian recomputes ias_mu for every beta2.
        if args.prm_mode != "bayesian":
            ias_budgets = build_ias_budgets(
                data=data,
                confidence=args.ias_confidence,
                max_n=ias_max_n,
            )
        else:
            ias_budgets = None

        ias_repeat_results = []

        for r in range(repeats):
            perms = build_permutations(
                data=data,
                max_n=pool_max_n,
                seed=args.seed,
                repeat=r,
                subset_mode=args.subset_mode,
            )

            if args.prm_mode == "bayesian":
                best, sweep = evaluate_bayesian_ias(
                    data=data,
                    perms=perms,
                    beta2s=beta2s,
                    confidence=args.ias_confidence,
                    max_n=ias_max_n,
                )

                repeat_result = {
                    "repeat": r,
                    "accuracy": best["accuracy"],
                    "correct": best["correct"],
                    "total": len(data),
                    "oracle_accuracy": best[
                        "oracle_accuracy"
                    ],
                    "oracle_correct": best[
                        "oracle_correct"
                    ],
                    "average_n": best["average_n"],
                    "budget_ratio": best[
                        "budget_ratio"
                    ],
                    "budget_histogram": best[
                        "budget_histogram"
                    ],
                    "best": best,
                    "beta2_sweep": sweep,
                }

            else:
                oracle_acc, oracle_correct = (
                    evaluate_oracle(
                        data,
                        perms,
                        ias_budgets,
                    )
                )

                if args.prm_mode == "beta":
                    best, sweep = evaluate_beta(
                        data=data,
                        perms=perms,
                        budgets=ias_budgets,
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
                        ias_budgets,
                    )

                    repeat_result = {
                        "repeat": r,
                        "accuracy": acc,
                        "correct": correct,
                        "total": len(data),
                        "oracle_accuracy": oracle_acc,
                        "oracle_correct": oracle_correct,
                    }

            ias_repeat_results.append(
                repeat_result
            )

        ias_accs = [
            x["accuracy"]
            for x in ias_repeat_results
        ]

        ias_oracle_accs = [
            x["oracle_accuracy"]
            for x in ias_repeat_results
        ]

        ias_acc_mean, ias_acc_std = mean_std(
            ias_accs
        )

        ias_oracle_mean, ias_oracle_std = mean_std(
            ias_oracle_accs
        )

        if args.prm_mode == "bayesian":
            average_n = safe_mean(
                [
                    x["average_n"]
                    for x in ias_repeat_results
                ]
            )

            # The selected beta2 can differ across repeats,
            # matching the current BetaPRM best-per-repeat protocol.
            histogram = ias_repeat_results[0][
                "budget_histogram"
            ]

            best_beta2_first_repeat = (
                ias_repeat_results[0]["best"]["beta2"]
            )

        else:
            average_n = safe_mean(
                ias_budgets
            )

            histogram = {
                str(n): ias_budgets.count(n)
                for n in range(
                    1,
                    ias_max_n + 1,
                )
                if ias_budgets.count(n) > 0
            }

            best_beta2_first_repeat = None

        output["ias"] = {
            "confidence": args.ias_confidence,
            "max_n": ias_max_n,
            "average_n": average_n,
            "budget_ratio": (
                average_n / ias_max_n
            ),
            "budget_histogram": histogram,
            "accuracy_mean": ias_acc_mean,
            "accuracy_std": ias_acc_std,
            "oracle_pass_mean": ias_oracle_mean,
            "oracle_pass_std": ias_oracle_std,
            "repeat_results": ias_repeat_results,
        }

        if args.prm_mode == "bayesian":
            output["ias"][
                "best_beta2_first_repeat"
            ] = best_beta2_first_repeat

            output["ias"][
                "beta2_grid"
            ] = beta2s

            print(
                f"IAS    "
                f"C={args.ias_confidence:.4f} "
                f"Nmax={ias_max_n} "
                f"avgN={average_n:.3f} "
                f"budget="
                f"{average_n / ias_max_n:.4f} "
                f"acc={ias_acc_mean:.4f} "
                f"± {ias_acc_std:.4f} "
                f"oracle={ias_oracle_mean:.4f} "
                f"best(first repeat): "
                f"beta2={best_beta2_first_repeat}"
            )

        else:
            print(
                f"IAS    "
                f"C={args.ias_confidence:.4f} "
                f"Nmax={ias_max_n} "
                f"avgN={average_n:.3f} "
                f"budget="
                f"{average_n / ias_max_n:.4f} "
                f"acc={ias_acc_mean:.4f} "
                f"± {ias_acc_std:.4f} "
                f"oracle={ias_oracle_mean:.4f}"
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