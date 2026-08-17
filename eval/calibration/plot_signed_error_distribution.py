#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot signed PRM calibration error distribution."
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="MathVerse",
        help="support MathVision, Geometry3k, Mavis-Geometry, MathV360K"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/calibration/mathvision/predictions_betaprm_checkpoint1103.jsonl",
        help="Prediction JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/calibration/mathvision/MathVision.pdf",
        help="Output figure path, e.g. error_distribution.png",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Optional JSON file to save summary statistics.",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default="target_success_prob",
        help="Field containing empirical/real success probability.",
    )
    parser.add_argument(
        "--pred-key",
        type=str,
        default="pred_prob",
        help="Field containing PRM predicted probability.",
    )
    parser.add_argument(
        "--error-definition",
        type=str,
        default="pred_minus_real",
        choices=["real_minus_pred", "pred_minus_real"],
        help=(
            "Signed error definition. "
            "'real_minus_pred': negative means overestimation. "
            "'pred_minus_real': positive means overestimation."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    return parser.parse_args()


def load_predictions(path, target_key, pred_key):
    targets = []
    preds = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            if target_key not in item:
                raise KeyError(
                    f"Missing target key '{target_key}' at line {line_idx}"
                )
            if pred_key not in item:
                raise KeyError(
                    f"Missing prediction key '{pred_key}' at line {line_idx}"
                )

            target = float(item[target_key])
            pred = float(item[pred_key])

            if not np.isfinite(target):
                raise ValueError(
                    f"Non-finite target at line {line_idx}: {target}"
                )
            if not np.isfinite(pred):
                raise ValueError(
                    f"Non-finite prediction at line {line_idx}: {pred}"
                )

            targets.append(target)
            preds.append(pred)

    if not targets:
        raise ValueError(f"No valid records found in {path}")

    return np.asarray(targets), np.asarray(preds)


def compute_statistics(targets, preds, errors, error_definition):
    tolerance = 1e-12

    if error_definition == "real_minus_pred":
        # real - pred < 0 => pred > real => overestimation
        over_mask = errors < -tolerance
        under_mask = errors > tolerance
    else:
        # pred - real > 0 => pred > real => overestimation
        over_mask = errors > tolerance
        under_mask = errors < -tolerance

    equal_mask = ~(over_mask | under_mask)

    over_magnitude = preds[over_mask] - targets[over_mask]
    under_magnitude = targets[under_mask] - preds[under_mask]

    stats = {
        "num_prefixes": int(len(errors)),
        "mean_target": float(np.mean(targets)),
        "mean_prediction": float(np.mean(preds)),
        "mean_signed_error": float(np.mean(errors)),
        "median_signed_error": float(np.median(errors)),
        "std_signed_error": float(np.std(errors)),
        "mean_absolute_error": float(np.mean(np.abs(errors))),
        "overestimation_rate": float(np.mean(over_mask)),
        "underestimation_rate": float(np.mean(under_mask)),
        "exact_rate": float(np.mean(equal_mask)),
        "mean_overestimation_magnitude": (
            float(np.mean(over_magnitude))
            if len(over_magnitude) > 0
            else 0.0
        ),
        "mean_underestimation_magnitude": (
            float(np.mean(under_magnitude))
            if len(under_magnitude) > 0
            else 0.0
        ),
        "p05_signed_error": float(np.quantile(errors, 0.05)),
        "p25_signed_error": float(np.quantile(errors, 0.25)),
        "p50_signed_error": float(np.quantile(errors, 0.50)),
        "p75_signed_error": float(np.quantile(errors, 0.75)),
        "p95_signed_error": float(np.quantile(errors, 0.95)),
        "error_definition": error_definition,
    }

    return stats


def plot_distribution(errors, stats, output_path, bins, dpi, title, error_definition, task):
    fig, ax = plt.subplots(figsize=(5.0, 4.8))

    # Histogram as probability density.
    ax.hist(
        errors,
        bins=bins,
        density=True,
        alpha=0.3,
        edgecolor="none",
        label="Histogram",
    )

    # Smooth KDE density curve.
    if len(np.unique(errors)) > 1:
        kde = gaussian_kde(errors)

        x_min = max(-1.0, float(np.min(errors)) - 0.05)
        x_max = min(1.0, float(np.max(errors)) + 0.05)

        xs = np.linspace(x_min, x_max, 500)
        ys = kde(xs)

        ax.plot(
            xs,
            ys,
            linewidth=4.5,
            label="Density",
        )

    # Zero-error reference.
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=3,
        label="Zero error",
    )

    # Mean signed error.
    # mean_error = stats["mean_signed_error"]
    # ax.axvline(
    #     mean_error,
    #     linestyle=":",
    #     linewidth=1.5,
    #     label=f"Mean = {mean_error:.3f}",
    # )

    if error_definition == "real_minus_pred":
        xlabel = "Signed Error (Real Success Prob. - PRM Prediction)"
        interpretation = "Negative = Overestimation"
    else:
        xlabel = "Prediction Error"
        interpretation = "Positive = Overestimation"

    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel("Probability Density", fontsize=22)
    ax.tick_params(axis='both', labelsize=19)

    if title is None:
        title = (
            f"{task}"
        )

    ax.set_title(title, fontsize=24)

    if task == "MathVision":
        ax.legend(fontsize=19, framealpha=0.8)
    ax.grid(alpha=0.2)

    # Probability differences are bounded by [-1, 1].
    ax.set_xlim(-1.0, 1.0)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()

    targets, preds = load_predictions(
        args.input,
        args.target_key,
        args.pred_key,
    )

    if args.error_definition == "real_minus_pred":
        errors = targets - preds
    else:
        errors = preds - targets

    stats = compute_statistics(
        targets,
        preds,
        errors,
        args.error_definition,
    )

    plot_distribution(
        errors=errors,
        stats=stats,
        output_path=args.output,
        bins=args.bins,
        dpi=args.dpi,
        title=args.title,
        error_definition=args.error_definition,
        task=args.task,
    )

    print("=== Signed Error Analysis ===")
    print(f"Number of prefixes          : {stats['num_prefixes']}")
    print(f"Mean real value             : {stats['mean_target']:.6f}")
    print(f"Mean prediction             : {stats['mean_prediction']:.6f}")
    print(f"Mean signed error           : {stats['mean_signed_error']:.6f}")
    print(f"Median signed error         : {stats['median_signed_error']:.6f}")
    print(f"Mean absolute error         : {stats['mean_absolute_error']:.6f}")
    print(
        f"Overestimation rate         : "
        f"{100.0 * stats['overestimation_rate']:.2f}%"
    )
    print(
        f"Underestimation rate        : "
        f"{100.0 * stats['underestimation_rate']:.2f}%"
    )
    print(
        f"Mean overestimation amount  : "
        f"{stats['mean_overestimation_magnitude']:.6f}"
    )
    print(
        f"Mean underestimation amount : "
        f"{stats['mean_underestimation_magnitude']:.6f}"
    )

    if args.error_definition == "real_minus_pred":
        if stats["mean_signed_error"] < 0:
            print(
                "\n[Result] Mean signed error < 0: "
                "the PRM is overestimating on average."
            )
        elif stats["mean_signed_error"] > 0:
            print(
                "\n[Result] Mean signed error > 0: "
                "the PRM is underestimating on average."
            )
        else:
            print("\n[Result] Mean signed error is approximately zero.")
    else:
        if stats["mean_signed_error"] > 0:
            print(
                "\n[Result] Mean signed error > 0: "
                "the PRM is overestimating on average."
            )
        elif stats["mean_signed_error"] < 0:
            print(
                "\n[Result] Mean signed error < 0: "
                "the PRM is underestimating on average."
            )
        else:
            print("\n[Result] Mean signed error is approximately zero.")

    print(f"\nFigure saved to: {args.output}")

    if args.summary_output is not None:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        print(f"Summary saved to: {args.summary_output}")


if __name__ == "__main__":
    main()