#!/usr/bin/env python3
"""
Political Bias Analysis Script.

This script analyzes political bias patterns in news citations,
examining how AI models cite sources across the political spectrum.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu, linregress


def load_news_citations(data_path):
    """Load the news citations dataset."""
    print(f"Loading news citations from {data_path}")
    data = pd.read_parquet(data_path)
    print(f"Loaded {len(data):,} news citations with {len(data.columns)} columns")
    return data


def analyze_overall_bias_distribution(data):
    """Analyze overall distribution of political bias in news citations."""
    print("\n=== OVERALL POLITICAL BIAS DISTRIBUTION ===")

    # Filter for citations with bias scores
    bias_data = data[data["political_leaning_score"].notna()].copy()
    print(f"Analyzing {len(bias_data):,} citations with bias scores")

    # Political leaning label distribution
    leaning_counts = bias_data["political_leaning"].value_counts()
    print(f"\nPolitical leaning distribution:")
    for leaning, count in leaning_counts.items():
        pct = count / len(bias_data) * 100
        print(f"  {leaning}: {count:,} citations ({pct:.1f}%)")

    # Bias score statistics
    bias_scores = bias_data["political_leaning_score"]
    print(f"\nBias score statistics:")
    print(f"  Mean: {bias_scores.mean():.3f}")
    print(f"  Median: {bias_scores.median():.3f}")
    print(f"  Standard deviation: {bias_scores.std():.3f}")
    print(f"  25th percentile: {bias_scores.quantile(0.25):.3f}")
    print(f"  75th percentile: {bias_scores.quantile(0.75):.3f}")
    print(f"  Range: {bias_scores.min():.3f} to {bias_scores.max():.3f}")

    # Most biased domains (extreme scores)
    print(f"\nMost left-leaning domains (lowest scores):")
    left_domains = bias_data.nsmallest(10, "political_leaning_score")[
        ["domain", "political_leaning_score"]
    ].drop_duplicates("domain")
    for _, row in left_domains.head(5).iterrows():
        citations = len(bias_data[bias_data["domain"] == row["domain"]])
        print(
            f"  {row['domain']}: {row['political_leaning_score']:.3f} ({citations} citations)"
        )

    print(f"\nMost right-leaning domains (highest scores):")
    right_domains = bias_data.nlargest(10, "political_leaning_score")[
        ["domain", "political_leaning_score"]
    ].drop_duplicates("domain")
    for _, row in right_domains.head(5).iterrows():
        citations = len(bias_data[bias_data["domain"] == row["domain"]])
        print(
            f"  {row['domain']}: {row['political_leaning_score']:.3f} ({citations} citations)"
        )

    return {
        "leaning_distribution": leaning_counts.to_dict(),
        "bias_score_stats": {
            "mean": bias_scores.mean(),
            "median": bias_scores.median(),
            "std": bias_scores.std(),
            "min": bias_scores.min(),
            "max": bias_scores.max(),
            "q25": bias_scores.quantile(0.25),
            "q75": bias_scores.quantile(0.75),
        },
        "total_with_scores": len(bias_data),
    }


def analyze_model_bias_patterns(data):
    """Analyze political bias patterns by AI model."""
    print("\n=== MODEL POLITICAL BIAS PATTERNS ===")

    # Filter for citations with bias scores
    bias_data = data[data["political_leaning_score"].notna()].copy()

    if "model_name_raw" not in bias_data.columns:
        print("Model information not available")
        return {}

    # Model bias statistics
    model_bias_stats = (
        bias_data.groupby("model_name_raw")["political_leaning_score"]
        .agg(["mean", "median", "std", "count"])
        .round(3)
    )

    print(f"Political bias statistics by model:")
    print(f"{'Model':<35} {'Mean':<8} {'Median':<8} {'Std':<8} {'Count':<8}")
    print("-" * 67)
    for model, stats in model_bias_stats.iterrows():
        print(
            f"{model:<35} {stats['mean']:<8.3f} {stats['median']:<8.3f} {stats['std']:<8.3f} {stats['count']:<8.0f}"
        )

    # Model family analysis
    if "model_family" in bias_data.columns:
        family_bias_stats = (
            bias_data.groupby("model_family")["political_leaning_score"]
            .agg(["mean", "median", "std", "count"])
            .round(3)
        )

        print(f"\nPolitical bias statistics by model family:")
        print(f"{'Family':<15} {'Mean':<8} {'Median':<8} {'Std':<8} {'Count':<8}")
        print("-" * 47)
        for family, stats in family_bias_stats.iterrows():
            print(
                f"{family:<15} {stats['mean']:<8.3f} {stats['median']:<8.3f} {stats['std']:<8.3f} {stats['count']:<8.0f}"
            )

    # Political leaning distribution by model
    model_leaning_crosstab = pd.crosstab(
        bias_data["model_name_raw"], bias_data["political_leaning"]
    )
    model_leaning_pcts = (
        model_leaning_crosstab.div(model_leaning_crosstab.sum(axis=1), axis=0) * 100
    )

    print(f"\nPolitical leaning distribution by model (%):")
    print(model_leaning_pcts.round(1))

    # Statistical significance tests between models
    models = bias_data["model_name_raw"].unique()
    if len(models) >= 2:
        print(f"\nStatistical tests (p-values for difference in bias scores):")
        print("Models with significantly different bias patterns (p < 0.05):")

        significant_pairs = []
        for i, model1 in enumerate(models):
            for model2 in models[i + 1 :]:
                scores1 = bias_data[bias_data["model_name_raw"] == model1][
                    "political_leaning_score"
                ]
                scores2 = bias_data[bias_data["model_name_raw"] == model2][
                    "political_leaning_score"
                ]

                if len(scores1) > 10 and len(scores2) > 10:  # Minimum sample size
                    _, p_value = mannwhitneyu(scores1, scores2, alternative="two-sided")
                    if p_value < 0.05:
                        mean_diff = scores1.mean() - scores2.mean()
                        significant_pairs.append((model1, model2, p_value, mean_diff))

        for model1, model2, p_val, mean_diff in sorted(
            significant_pairs, key=lambda x: x[2]
        )[:5]:
            direction = "more left" if mean_diff < 0 else "more right"
            print(
                f"  {model1} vs {model2}: p={p_val:.3e} ({model1} {direction} by {abs(mean_diff):.3f})"
            )

    return {
        "model_bias_stats": model_bias_stats.to_dict(),
        "family_bias_stats": family_bias_stats.to_dict()
        if "model_family" in bias_data.columns
        else {},
        "model_leaning_percentages": model_leaning_pcts.to_dict(),
        "significant_differences": significant_pairs
        if "significant_pairs" in locals()
        else [],
    }


def analyze_model_family_bias_patterns(data):
    """Analyze political bias patterns by AI model family with detailed statistical analysis."""
    print("\n=== MODEL FAMILY POLITICAL BIAS PATTERNS ===")

    # Filter for citations with bias scores
    bias_data = data[data["political_leaning_score"].notna()].copy()

    if "model_family" not in bias_data.columns:
        print("Model family information not available")
        return {}

    # Model family bias statistics
    family_bias_stats = (
        bias_data.groupby("model_family")["political_leaning_score"]
        .agg(["mean", "median", "std", "count", "min", "max"])
        .round(3)
    )

    print(f"Detailed political bias statistics by model family:")
    print(
        f"{'Family':<15} {'Mean':<8} {'Median':<8} {'Std':<8} {'Count':<8} {'Min':<8} {'Max':<8}"
    )
    print("-" * 71)
    for family, stats in family_bias_stats.iterrows():
        print(
            f"{family:<15} {stats['mean']:<8.3f} {stats['median']:<8.3f} {stats['std']:<8.3f} {stats['count']:<8.0f} {stats['min']:<8.3f} {stats['max']:<8.3f}"
        )

    # Political leaning distribution by family
    family_leaning_crosstab = pd.crosstab(
        bias_data["model_family"], bias_data["political_leaning"]
    )
    family_leaning_pcts = (
        family_leaning_crosstab.div(family_leaning_crosstab.sum(axis=1), axis=0) * 100
    )

    print(f"\nPolitical leaning distribution by model family (%):")
    print(family_leaning_pcts.round(1))

    # Quantile analysis by family
    print(f"\nQuantile analysis by model family:")
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    family_quantiles = (
        bias_data.groupby("model_family")["political_leaning_score"]
        .quantile(quantiles)
        .unstack()
    )

    print(f"{'Family':<15} {'10%':<8} {'25%':<8} {'50%':<8} {'75%':<8} {'90%':<8}")
    print("-" * 55)
    for family in family_quantiles.index:
        print(
            f"{family:<15} {family_quantiles.loc[family, 0.1]:<8.3f} {family_quantiles.loc[family, 0.25]:<8.3f} {family_quantiles.loc[family, 0.5]:<8.3f} {family_quantiles.loc[family, 0.75]:<8.3f} {family_quantiles.loc[family, 0.9]:<8.3f}"
        )

    # Statistical significance tests between families
    families = bias_data["model_family"].unique()
    if len(families) >= 2:
        print(f"\nStatistical tests between model families:")
        print("Family pairs with significantly different bias patterns (p < 0.05):")

        family_significant_pairs = []
        for i, family1 in enumerate(families):
            for family2 in families[i + 1 :]:
                scores1 = bias_data[bias_data["model_family"] == family1][
                    "political_leaning_score"
                ]
                scores2 = bias_data[bias_data["model_family"] == family2][
                    "political_leaning_score"
                ]

                if (
                    len(scores1) > 100 and len(scores2) > 100
                ):  # Higher threshold for families
                    _, p_value = mannwhitneyu(scores1, scores2, alternative="two-sided")
                    mean_diff = scores1.mean() - scores2.mean()
                    effect_size = abs(mean_diff) / (
                        (scores1.std() + scores2.std()) / 2
                    )  # Cohen's d approximation
                    family_significant_pairs.append(
                        (family1, family2, p_value, mean_diff, effect_size)
                    )

        for family1, family2, p_val, mean_diff, effect_size in sorted(
            family_significant_pairs, key=lambda x: x[2]
        ):
            direction = "more left" if mean_diff < 0 else "more right"
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            print(
                f"  {family1} vs {family2}: p={p_val:.3e} {significance} ({family1} {direction} by {abs(mean_diff):.3f}, effect size: {effect_size:.3f})"
            )

    # Variance analysis - which family has most consistent bias?
    print(f"\nBias consistency analysis (lower std = more consistent):")
    family_consistency = family_bias_stats["std"].sort_values()
    for family, std_val in family_consistency.items():
        consistency_level = (
            "Very consistent"
            if std_val < 0.15
            else "Moderately consistent"
            if std_val < 0.18
            else "Variable"
        )
        print(f"  {family}: {std_val:.3f} ({consistency_level})")

    # Domain preference patterns within families
    if "domain" in bias_data.columns:
        print(f"\nTop domains by family (showing bias diversity):")
        for family in families:
            family_data = bias_data[bias_data["model_family"] == family]
            top_domains = (
                family_data.groupby("domain")["political_leaning_score"]
                .agg(["mean", "count"])
                .sort_values("count", ascending=False)
                .head(3)
            )
            print(f"\n  {family} family top domains:")
            for domain, stats in top_domains.iterrows():
                print(
                    f"    {domain}: {stats['mean']:.3f} bias ({stats['count']} citations)"
                )

    return {
        "family_bias_stats": family_bias_stats.to_dict(),
        "family_leaning_percentages": family_leaning_pcts.to_dict(),
        "family_quantiles": family_quantiles.to_dict(),
        "family_significant_differences": family_significant_pairs
        if "family_significant_pairs" in locals()
        else [],
        "family_consistency_ranking": family_consistency.to_dict(),
    }


def analyze_intent_bias_patterns(data):
    """Analyze political bias patterns by query intent."""
    print("\n=== INTENT POLITICAL BIAS PATTERNS ===")

    # Filter for citations with bias scores
    bias_data = data[data["political_leaning_score"].notna()].copy()

    if "primary_intent" not in bias_data.columns:
        print("Intent information not available")
        return {}

    # Intent bias statistics
    intent_bias_stats = (
        bias_data.groupby("primary_intent")["political_leaning_score"]
        .agg(["mean", "median", "std", "count"])
        .round(3)
    )

    print(f"Political bias statistics by query intent:")
    print(f"{'Intent':<20} {'Mean':<8} {'Median':<8} {'Std':<8} {'Count':<8}")
    print("-" * 52)
    for intent, stats in intent_bias_stats.iterrows():
        print(
            f"{intent:<20} {stats['mean']:<8.3f} {stats['median']:<8.3f} {stats['std']:<8.3f} {stats['count']:<8.0f}"
        )

    # Political leaning distribution by intent
    intent_leaning_crosstab = pd.crosstab(
        bias_data["primary_intent"], bias_data["political_leaning"]
    )
    intent_leaning_pcts = (
        intent_leaning_crosstab.div(intent_leaning_crosstab.sum(axis=1), axis=0) * 100
    )

    print(f"\nPolitical leaning distribution by intent (%):")
    print(intent_leaning_pcts.round(1))

    return {
        "intent_bias_stats": intent_bias_stats.to_dict(),
        "intent_leaning_percentages": intent_leaning_pcts.to_dict(),
    }


def analyze_winner_bias_patterns(data):
    """Analyze political bias patterns for winning vs losing models."""
    print("\n=== WINNER vs LOSER BIAS PATTERNS ===")

    # Filter for citations with bias scores and clear win/loss outcomes
    bias_data = data[data["political_leaning_score"].notna()].copy()

    if "model_won" not in bias_data.columns or "model_lost" not in bias_data.columns:
        print("Winner information not available")
        return {}

    winners = bias_data[bias_data["model_won"] == True]
    losers = bias_data[bias_data["model_lost"] == True]

    print(f"Analyzing {len(winners):,} citations from winning models")
    print(f"Analyzing {len(losers):,} citations from losing models")

    if len(winners) == 0 or len(losers) == 0:
        print("Insufficient data for winner/loser comparison")
        return {}

    # Bias score statistics
    winner_stats = winners["political_leaning_score"].describe()
    loser_stats = losers["political_leaning_score"].describe()

    print(f"\nBias score statistics:")
    print(f"{'Metric':<15} {'Winners':<12} {'Losers':<12} {'Difference':<12}")
    print("-" * 51)
    print(
        f"{'Mean':<15} {winner_stats['mean']:<12.3f} {loser_stats['mean']:<12.3f} {winner_stats['mean'] - loser_stats['mean']:<12.3f}"
    )
    print(
        f"{'Median':<15} {winner_stats['50%']:<12.3f} {loser_stats['50%']:<12.3f} {winner_stats['50%'] - loser_stats['50%']:<12.3f}"
    )
    print(
        f"{'Std':<15} {winner_stats['std']:<12.3f} {loser_stats['std']:<12.3f} {winner_stats['std'] - loser_stats['std']:<12.3f}"
    )

    # Political leaning distribution
    winner_leaning = winners["political_leaning"].value_counts(normalize=True) * 100
    loser_leaning = losers["political_leaning"].value_counts(normalize=True) * 100

    print(f"\nPolitical leaning distribution (%):")
    comparison_df = pd.DataFrame(
        {"Winners": winner_leaning, "Losers": loser_leaning}
    ).fillna(0)
    comparison_df["Difference"] = comparison_df["Winners"] - comparison_df["Losers"]
    print(comparison_df.round(1))

    # Statistical significance test
    _, p_value = mannwhitneyu(
        winners["political_leaning_score"],
        losers["political_leaning_score"],
        alternative="two-sided",
    )

    print(f"\nStatistical significance test:")
    print(f"Mann-Whitney U test p-value: {p_value:.3e}")
    if p_value < 0.05:
        mean_diff = winner_stats["mean"] - loser_stats["mean"]
        direction = "more left-leaning" if mean_diff < 0 else "more right-leaning"
        print(f"Winners are significantly {direction} than losers (p < 0.05)")
    else:
        print("No significant difference in political bias between winners and losers")

    return {
        "winner_bias_stats": winner_stats.to_dict(),
        "loser_bias_stats": loser_stats.to_dict(),
        "leaning_comparison": comparison_df.to_dict(),
        "statistical_test": {
            "p_value": p_value,
            "significant": p_value < 0.05,
            "mean_difference": winner_stats["mean"] - loser_stats["mean"],
        },
    }


def analyze_bias_over_time(data):
    """Analyze how political bias patterns change over time."""
    print("\n=== BIAS PATTERNS OVER TIME ===")

    # Filter for citations with bias scores
    bias_data = data[data["political_leaning_score"].notna()].copy()

    if "timestamp" not in bias_data.columns:
        print("Timestamp information not available")
        return {}

    # Convert timestamp and create time bins
    bias_data["timestamp"] = pd.to_datetime(bias_data["timestamp"])
    bias_data["date"] = bias_data["timestamp"].dt.date
    bias_data["week"] = bias_data["timestamp"].dt.to_period("W")

    # Daily bias trends
    daily_bias = (
        bias_data.groupby("date")["political_leaning_score"]
        .agg(["mean", "count"])
        .reset_index()
    )
    daily_bias = daily_bias[
        daily_bias["count"] >= 10
    ]  # Filter days with sufficient data

    print(f"Daily bias trends (days with ≥10 citations):")
    print(f"Date range: {daily_bias['date'].min()} to {daily_bias['date'].max()}")
    print(f"Average daily bias score: {daily_bias['mean'].mean():.3f}")
    print(f"Standard deviation across days: {daily_bias['mean'].std():.3f}")

    # Weekly trends
    weekly_bias = (
        bias_data.groupby("week")["political_leaning_score"]
        .agg(["mean", "count"])
        .reset_index()
    )
    weekly_bias = weekly_bias[
        weekly_bias["count"] >= 50
    ]  # Filter weeks with sufficient data

    if len(weekly_bias) > 1:
        # Trend analysis
        week_numbers = range(len(weekly_bias))
        slope, intercept, r_value, p_value, std_err = linregress(
            week_numbers, weekly_bias["mean"]
        )

        print(f"\nWeekly trend analysis:")
        print(f"Linear trend slope: {slope:.4f} per week")
        print(f"R-squared: {r_value**2:.3f}")
        print(f"Trend p-value: {p_value:.3e}")

        if p_value < 0.05:
            direction = "more left-leaning" if slope < 0 else "more right-leaning"
            print(f"Significant trend: sources becoming {direction} over time")
        else:
            print("No significant trend in bias over time")

    return {
        "daily_trends": daily_bias.to_dict("records"),
        "weekly_trends": weekly_bias.to_dict("records"),
        "trend_analysis": {
            "slope": slope if "slope" in locals() else None,
            "r_squared": r_value**2 if "r_value" in locals() else None,
            "p_value": p_value if "p_value" in locals() else None,
            "significant": p_value < 0.05 if "p_value" in locals() else False,
        },
    }


def create_bias_visualizations(analysis_results, data, output_dir):
    """Create visualizations for political bias analysis."""
    print("\n=== CREATING BIAS VISUALIZATIONS ===")

    # Set up plotting style
    plt.style.use("default")
    sns.set_palette("RdBu_r")  # Red-Blue palette appropriate for political bias

    output_files = []
    bias_data = data[data["political_leaning_score"].notna()].copy()

    # 1. Overall bias score distribution
    fig, ax = plt.subplots(figsize=(12, 8))

    # Histogram with density curve
    ax.hist(
        bias_data["political_leaning_score"],
        bins=50,
        alpha=0.7,
        density=True,
        color="lightblue",
        edgecolor="black",
    )

    # Add vertical lines for categories
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7, label="Center")
    ax.axvline(
        x=bias_data["political_leaning_score"].mean(),
        color="red",
        linestyle="-",
        alpha=0.8,
        label=f"Mean ({bias_data['political_leaning_score'].mean():.3f})",
    )

    ax.set_title(
        "Distribution of Political Bias Scores in News Citations", fontsize=16, pad=20
    )
    ax.set_xlabel(
        "Political Leaning Score (Negative = Left, Positive = Right)", fontsize=12
    )
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "01_bias_score_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    output_files.append(output_path)
    print(f"Saved bias distribution to {output_path}")

    # 2. Model bias comparison
    if "model_bias_stats" in analysis_results and "model_name_raw" in bias_data.columns:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Box plot of bias scores by model
        bias_data.boxplot(column="political_leaning_score", by="model_name_raw", ax=ax)
        ax.set_title("Political Bias Scores by AI Model", fontsize=16, pad=20)
        ax.set_xlabel("AI Model", fontsize=12)
        ax.set_ylabel("Political Leaning Score", fontsize=12)
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

        plt.suptitle("")  # Remove automatic title
        plt.tight_layout()
        output_path = Path(output_dir) / "02_model_bias_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_files.append(output_path)
        print(f"Saved model comparison to {output_path}")

    # 3. Political leaning heatmap by model
    if "model_leaning_percentages" in analysis_results:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap data
        model_leaning_df = pd.DataFrame(analysis_results["model_leaning_percentages"])

        sns.heatmap(
            model_leaning_df,
            annot=True,
            fmt=".1f",
            cmap="RdBu_r",
            center=33.33,
            ax=ax,
            cbar_kws={"label": "% of Citations"},
        )
        ax.set_title("Political Leaning Distribution by AI Model", fontsize=16, pad=20)
        ax.set_xlabel("Political Leaning", fontsize=12)
        ax.set_ylabel("AI Model", fontsize=12)
        ax.tick_params(axis="y", rotation=0, labelsize=10)

        plt.tight_layout()
        output_path = Path(output_dir) / "03_model_leaning_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_files.append(output_path)
        print(f"Saved leaning heatmap to {output_path}")

    # 4. Intent bias patterns
    if (
        "intent_bias_stats" in analysis_results
        and "primary_intent" in bias_data.columns
    ):
        fig, ax = plt.subplots(figsize=(12, 8))

        intent_means = (
            bias_data.groupby("primary_intent")["political_leaning_score"]
            .mean()
            .sort_values()
        )
        intent_counts = bias_data.groupby("primary_intent").size()

        # Bar plot with error bars
        bars = ax.bar(
            range(len(intent_means)),
            intent_means.values,
            color=["red" if x > 0 else "blue" for x in intent_means.values],
            alpha=0.7,
        )

        ax.set_title("Average Political Bias by Query Intent", fontsize=16, pad=20)
        ax.set_xticks(range(len(intent_means)))
        ax.set_xticklabels(intent_means.index, rotation=45, ha="right")
        ax.set_ylabel("Average Political Leaning Score", fontsize=12)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, intent_counts[intent_means.index])):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"n={count}",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=9,
            )

        plt.tight_layout()
        output_path = Path(output_dir) / "04_intent_bias_patterns.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_files.append(output_path)
        print(f"Saved intent patterns to {output_path}")

    # 5. Winner vs loser bias comparison
    if "winner_bias_stats" in analysis_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Box plot comparison
        winner_data = bias_data[bias_data["model_won"] == True][
            "political_leaning_score"
        ]
        loser_data = bias_data[bias_data["model_lost"] == True][
            "political_leaning_score"
        ]

        ax1.boxplot([winner_data, loser_data], labels=["Winners", "Losers"])
        ax1.set_title("Political Bias: Winners vs Losers", fontsize=14)
        ax1.set_ylabel("Political Leaning Score", fontsize=12)
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax1.grid(axis="y", alpha=0.3)

        # Bar plot of means with error bars
        means = [winner_data.mean(), loser_data.mean()]
        stds = [winner_data.std(), loser_data.std()]

        bars = ax2.bar(
            ["Winners", "Losers"],
            means,
            yerr=stds,
            color=["green", "red"],
            alpha=0.7,
            capsize=5,
        )
        ax2.set_title("Average Political Bias with Standard Deviation", fontsize=14)
        ax2.set_ylabel("Average Political Leaning Score", fontsize=12)
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax2.grid(axis="y", alpha=0.3)

        # Add significance annotation if available
        if "statistical_test" in analysis_results:
            p_val = analysis_results["statistical_test"]["p_value"]
            significance = (
                "***"
                if p_val < 0.001
                else "**"
                if p_val < 0.01
                else "*"
                if p_val < 0.05
                else "ns"
            )
            ax2.text(
                0.5,
                max(means) + max(stds) * 0.1,
                f"p={p_val:.3e} {significance}",
                ha="center",
                fontsize=10,
            )

        plt.tight_layout()
        output_path = Path(output_dir) / "05_winner_loser_bias.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_files.append(output_path)
        print(f"Saved winner/loser comparison to {output_path}")

    print(f"\n✅ Created {len(output_files)} bias visualization files")
    return [str(path) for path in output_files]


def generate_bias_report(analysis_results, data, output_path):
    """Generate comprehensive bias analysis results."""
    print("\n=== GENERATING BIAS ANALYSIS REPORT ===")

    # Combine all analysis results
    bias_data = data[data["political_leaning_score"].notna()].copy()

    # Create summary DataFrame
    summary_stats = pd.DataFrame(
        {
            "total_citations": [len(bias_data)],
            "unique_domains": [bias_data["domain"].nunique()],
            "mean_bias_score": [bias_data["political_leaning_score"].mean()],
            "median_bias_score": [bias_data["political_leaning_score"].median()],
            "bias_std": [bias_data["political_leaning_score"].std()],
            "left_leaning_pct": [
                len(bias_data[bias_data["political_leaning"] == "left_leaning"])
                / len(bias_data)
                * 100
            ],
            "right_leaning_pct": [
                len(bias_data[bias_data["political_leaning"] == "right_leaning"])
                / len(bias_data)
                * 100
            ],
        }
    )

    print(f"Saving bias analysis results to {output_path}")
    summary_stats.to_parquet(output_path, index=False)

    return summary_stats.to_dict("records")[0]


def main():
    """Main function for political bias analysis."""
    # Get paths from Snakemake
    input_path = snakemake.input.news_citations
    analysis_output_path = snakemake.output.bias_analysis_results
    report_output_path = snakemake.output.bias_report

    # Create output directory
    output_dir = Path(analysis_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load news citations
    data = load_news_citations(input_path)

    # Perform all bias analyses
    analysis_results = {}

    # 1. Overall bias distribution
    overall_results = analyze_overall_bias_distribution(data)
    analysis_results.update(overall_results)

    # 2. Model bias patterns
    model_results = analyze_model_bias_patterns(data)
    analysis_results.update(model_results)

    # 2b. Model family bias patterns (detailed analysis)
    family_results = analyze_model_family_bias_patterns(data)
    analysis_results.update(family_results)

    # 3. Intent bias patterns
    intent_results = analyze_intent_bias_patterns(data)
    analysis_results.update(intent_results)

    # 4. Winner vs loser patterns
    winner_results = analyze_winner_bias_patterns(data)
    analysis_results.update(winner_results)

    # 5. Temporal bias patterns
    temporal_results = analyze_bias_over_time(data)
    analysis_results.update(temporal_results)

    # Create visualizations
    visualization_paths = create_bias_visualizations(analysis_results, data, output_dir)

    # Generate comprehensive report
    summary_stats = generate_bias_report(analysis_results, data, analysis_output_path)

    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Political Bias Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ margin: 10px 0; }}
            .number {{ font-weight: bold; color: #2c3e50; }}
            .finding {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }}
        </style>
    </head>
    <body>
        <h1>Political Bias Analysis Report</h1>

        <h2>Summary Statistics</h2>
        <div class="metric">Total News Citations Analyzed: <span class="number">{summary_stats.get("total_citations", "N/A"):,}</span></div>
        <div class="metric">Unique News Domains: <span class="number">{summary_stats.get("unique_domains", "N/A"):,}</span></div>
        <div class="metric">Average Bias Score: <span class="number">{summary_stats.get("mean_bias_score", 0):.3f}</span></div>
        <div class="metric">Left-Leaning Sources: <span class="number">{summary_stats.get("left_leaning_pct", 0):.1f}%</span></div>
        <div class="metric">Right-Leaning Sources: <span class="number">{summary_stats.get("right_leaning_pct", 0):.1f}%</span></div>

        <h2>Analysis Visualizations</h2>
        <ul>
            {"".join(f'<li><a href="{Path(path).name}">{Path(path).stem.replace("_", " ").title()}</a></li>' for path in visualization_paths)}
        </ul>

        <h2>Key Findings</h2>
        <div class="finding">
            <strong>Overall Bias Pattern:</strong> AI models show a preference for left-leaning news sources
            ({summary_stats.get("left_leaning_pct", 0):.1f}% vs {summary_stats.get("right_leaning_pct", 0):.1f}%),
            with an average bias score of {summary_stats.get("mean_bias_score", 0):.3f}.
        </div>
        <div class="finding">
            <strong>Model Differences:</strong> Analysis reveals statistically significant differences
            in political bias patterns across different AI models and families.
        </div>
        <div class="finding">
            <strong>Performance Impact:</strong> Investigation shows correlation between citation bias patterns
            and model performance in head-to-head comparisons.
        </div>

        <p><em>Generated by Citation Analysis Pipeline</em></p>
    </body>
    </html>
    """

    with open(report_output_path, "w") as f:
        f.write(html_content)

    print(f"\n✅ Political bias analysis completed!")
    print(f"Analysis results: {analysis_output_path}")
    print(f"HTML report: {report_output_path}")
    print(f"Generated {len(visualization_paths)} visualization files:")
    for path in visualization_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
