#!/usr/bin/env python3
"""
Response Signal Computation Script for News Citation Preference Analysis.

This script calculates response-level metrics including response length,
citation counts, quality metrics, and bias metrics for news citation analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_competition_data(threads_path, responses_path, citations_path):
    """Load the news competition data from prepare_data phase."""
    print("Loading news competition data...")

    threads_df = pd.read_parquet(threads_path)
    responses_df = pd.read_parquet(responses_path)
    citations_df = pd.read_parquet(citations_path)

    print(f"Loaded {len(threads_df):,} threads")
    print(f"Loaded {len(responses_df):,} responses")
    print(f"Loaded {len(citations_df):,} citations")

    return threads_df, responses_df, citations_df


def compute_response_signals(responses_df, citations_df):
    """Compute response-level signals for preference analysis."""

    print("Computing response-level signals...")

    # Start with responses data
    signals_df = responses_df.copy()

    # 1. Response length metrics
    print("  Computing response length metrics...")
    signals_df["response_length"] = signals_df["response_text"].str.len()
    signals_df["response_word_count"] = (
        signals_df["response_text"].str.split().str.len()
    )

    # 2. Citation count metrics per response
    print("  Computing citation count metrics...")
    citation_counts = citations_df.groupby("response_id").size().rename("num_citations")
    signals_df = signals_df.merge(
        citation_counts, left_on="response_id", right_index=True, how="left"
    )
    signals_df["num_citations"] = signals_df["num_citations"].fillna(0)

    # 3. Quality metrics
    print("  Computing quality metrics...")
    quality_stats = compute_quality_metrics(citations_df)
    signals_df = signals_df.merge(quality_stats, on="response_id", how="left")

    # 4. Political bias metrics
    print("  Computing political bias metrics...")
    bias_stats = compute_bias_metrics(citations_df)
    signals_df = signals_df.merge(bias_stats, on="response_id", how="left")

    # 5. Domain category metrics
    print("  Computing domain category metrics...")
    domain_stats = compute_domain_metrics(citations_df)
    signals_df = signals_df.merge(domain_stats, on="response_id", how="left")

    # Fill NaN values for responses without citations
    numeric_cols = [
        col
        for col in signals_df.columns
        if col.startswith(("proportion_", "avg_", "num_"))
    ]
    signals_df[numeric_cols] = signals_df[numeric_cols].fillna(0)

    print(f"Computed signals for {len(signals_df):,} responses")
    print(
        f"Signal columns: {[col for col in signals_df.columns if col not in responses_df.columns]}"
    )

    return signals_df


def compute_quality_metrics(citations_df):
    """Compute quality-related metrics per response."""

    quality_metrics = []

    for response_id, group in citations_df.groupby("response_id"):
        metrics = {"response_id": response_id}

        # Total citations for this response
        total_cites = len(group)

        if total_cites > 0:
            # Quality distribution
            quality_counts = group["domain_quality"].value_counts()
            metrics["proportion_high_quality"] = (
                quality_counts.get("high_quality", 0) / total_cites
            )
            metrics["proportion_low_quality"] = (
                quality_counts.get("low_quality", 0) / total_cites
            )
            metrics["proportion_unknown_quality"] = (
                quality_counts.get("unknown", 0) / total_cites
            )

            # Average quality score (for numeric scores)
            if "domain_quality_score" in group.columns:
                quality_scores = group["domain_quality_score"].dropna()
                if len(quality_scores) > 0:
                    metrics["avg_quality_score"] = quality_scores.mean()
                    metrics["min_quality_score"] = quality_scores.min()
                    metrics["max_quality_score"] = quality_scores.max()
                else:
                    metrics["avg_quality_score"] = np.nan
                    metrics["min_quality_score"] = np.nan
                    metrics["max_quality_score"] = np.nan
        else:
            # No citations case
            metrics["proportion_high_quality"] = 0
            metrics["proportion_low_quality"] = 0
            metrics["proportion_unknown_quality"] = 0
            metrics["avg_quality_score"] = np.nan
            metrics["min_quality_score"] = np.nan
            metrics["max_quality_score"] = np.nan

        quality_metrics.append(metrics)

    return pd.DataFrame(quality_metrics)


def compute_bias_metrics(citations_df):
    """Compute political bias metrics per response."""

    bias_metrics = []

    for response_id, group in citations_df.groupby("response_id"):
        metrics = {"response_id": response_id}

        # Total citations for this response
        total_cites = len(group)

        if total_cites > 0:
            # Political leaning distribution
            leaning_counts = group["political_leaning"].value_counts()
            metrics["proportion_left_leaning"] = (
                leaning_counts.get("left_leaning", 0) / total_cites
            )
            metrics["proportion_right_leaning"] = (
                leaning_counts.get("right_leaning", 0) / total_cites
            )
            metrics["proportion_unknown_leaning"] = (
                leaning_counts.get("unknown", 0) / total_cites
            )

            # Average bias score (for numeric scores)
            if "political_leaning_score" in group.columns:
                bias_scores = group["political_leaning_score"].dropna()
                if len(bias_scores) > 0:
                    metrics["avg_bias_score"] = bias_scores.mean()
                    metrics["min_bias_score"] = bias_scores.min()
                    metrics["max_bias_score"] = bias_scores.max()
                    metrics["bias_score_std"] = bias_scores.std()
                else:
                    metrics["avg_bias_score"] = np.nan
                    metrics["min_bias_score"] = np.nan
                    metrics["max_bias_score"] = np.nan
                    metrics["bias_score_std"] = np.nan
        else:
            # No citations case
            metrics["proportion_left_leaning"] = 0
            metrics["proportion_right_leaning"] = 0
            metrics["proportion_unknown_leaning"] = 0
            metrics["avg_bias_score"] = np.nan
            metrics["min_bias_score"] = np.nan
            metrics["max_bias_score"] = np.nan
            metrics["bias_score_std"] = np.nan

        bias_metrics.append(metrics)

    return pd.DataFrame(bias_metrics)


def compute_domain_metrics(citations_df):
    """Compute domain category metrics per response."""

    domain_metrics = []

    for response_id, group in citations_df.groupby("response_id"):
        metrics = {"response_id": response_id}

        # Total citations for this response
        total_cites = len(group)

        # Define domain categories once
        domain_categories = [
            "news",
            "academic",
            "social_media",
            "unclassified",
            "wiki",
            "gov_edu",
            "tech",
            "search_engine",
            "community_blog",
            "other",
        ]

        if total_cites > 0:
            # Domain classification distribution
            domain_counts = group["domain_classification"].value_counts()

            # Compute proportions for each category
            for category in domain_categories:
                metrics[f"proportion_{category}"] = (
                    domain_counts.get(category, 0) / total_cites
                )
        else:
            # No citations case - set all proportions to 0
            for category in domain_categories:
                metrics[f"proportion_{category}"] = 0

        domain_metrics.append(metrics)

    return pd.DataFrame(domain_metrics)


def add_competition_level_signals(signals_df, threads_df):
    """Add thread-level competition signals to responses."""

    print("Adding competition-level signals...")

    # Merge with thread information
    signals_with_threads = signals_df.merge(
        threads_df[
            ["thread_id", "winner", "primary_intent", "secondary_intent", "total_turns"]
        ],
        on="thread_id",
        how="left",
    )

    # Add model outcome indicators
    signals_with_threads["model_won"] = (
        (signals_with_threads["model_side"] == "a")
        & (signals_with_threads["winner"] == "model_a")
    ) | (
        (signals_with_threads["model_side"] == "b")
        & (signals_with_threads["winner"] == "model_b")
    )

    signals_with_threads["model_lost"] = (
        (signals_with_threads["model_side"] == "a")
        & (signals_with_threads["winner"] == "model_b")
    ) | (
        (signals_with_threads["model_side"] == "b")
        & (signals_with_threads["winner"] == "model_a")
    )

    return signals_with_threads


def validate_signals(signals_df):
    """Validate computed signals for consistency."""

    print("Validating computed signals...")

    # Check for missing values in key columns
    key_cols = [
        "response_length",
        "num_citations",
        "proportion_low_quality",
        "proportion_right_leaning",
    ]
    for col in key_cols:
        if col in signals_df.columns:
            missing = signals_df[col].isna().sum()
            if missing > 0:
                print(f"  WARNING: {missing:,} missing values in {col}")

    # Check proportions are between 0 and 1
    proportion_cols = [
        col for col in signals_df.columns if col.startswith("proportion_")
    ]
    for col in proportion_cols:
        if col in signals_df.columns:
            out_of_range = ((signals_df[col] < 0) | (signals_df[col] > 1)).sum()
            if out_of_range > 0:
                print(f"  WARNING: {out_of_range:,} out-of-range values in {col}")

    # Summary statistics
    print("Signal summary statistics:")
    numeric_signals = [
        col
        for col in signals_df.columns
        if col.startswith(("response_", "num_", "proportion_", "avg_"))
    ]
    if numeric_signals:
        summary = signals_df[numeric_signals].describe()
        print(summary.round(3))

    return True


def main():
    # Get input/output paths from Snakemake
    try:
        # Snakemake provides these variables
        input_threads = snakemake.input.threads
        input_responses = snakemake.input.responses
        input_citations = snakemake.input.citations

        output_signals = snakemake.output[0]

    except NameError:
        # Fallback for running outside Snakemake (for testing)
        print("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        input_dir = base_dir / "data/intermediate/preference_analysis"

        input_threads = input_dir / "news_competitions.parquet"
        input_responses = input_dir / "news_competitions_responses.parquet"
        input_citations = input_dir / "news_competitions_citations.parquet"

        output_signals = input_dir / "news_competitions_response_signals.parquet"

    try:
        # Load data
        threads_df, responses_df, citations_df = load_competition_data(
            input_threads, input_responses, input_citations
        )

        # Compute response signals
        signals_df = compute_response_signals(responses_df, citations_df)

        # Add competition-level information
        enriched_signals = add_competition_level_signals(signals_df, threads_df)

        # Validate results
        is_valid = validate_signals(enriched_signals)
        if not is_valid:
            print("WARNING: Signal validation failed")

        # Ensure output directory exists
        Path(output_signals).parent.mkdir(parents=True, exist_ok=True)

        # Save enriched signals
        enriched_signals.to_parquet(output_signals, index=False)

        print(f"\nSaved response signals to: {output_signals}")
        print(f"Dataset shape: {enriched_signals.shape}")

        # Summary statistics
        signal_cols = [
            col
            for col in enriched_signals.columns
            if col.startswith(("response_", "num_", "proportion_", "avg_"))
        ]
        print(f"Computed {len(signal_cols)} signal features")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
