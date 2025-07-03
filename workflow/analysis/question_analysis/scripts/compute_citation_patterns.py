#!/usr/bin/env python3
"""
Citation Patterns Computation Script for Question Analysis Pipeline.

This script computes response-level citation metrics for analyzing how
question features relate to AI models' news citation patterns.
Adapted from preference_analysis/compute_response_signals.py.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_analysis_data(questions_path, responses_path, citations_path):
    """Load the data needed for citation pattern analysis."""
    logger.info("Loading analysis data...")

    questions_df = pd.read_parquet(questions_path)
    responses_df = pd.read_parquet(responses_path)
    citations_df = pd.read_parquet(citations_path)

    logger.info(f"Loaded {len(questions_df):,} questions")
    logger.info(f"Loaded {len(responses_df):,} responses")
    logger.info(f"Loaded {len(citations_df):,} citations")

    return questions_df, responses_df, citations_df


def compute_citation_patterns(questions_df, responses_df, citations_df):
    """Compute citation patterns for each response."""
    logger.info("Computing citation patterns...")

    # Filter to only responses for English questions
    english_question_ids = set(questions_df["question_id"])
    english_responses = responses_df[
        responses_df["question_id"].isin(english_question_ids)
    ]

    logger.info(
        f"Processing {len(english_responses):,} responses for English questions"
    )

    # Start with response data
    patterns_df = english_responses.copy()

    # 1. Response length metrics
    logger.info("  Computing response length metrics...")
    patterns_df["response_length"] = patterns_df["response_text"].str.len()
    patterns_df["response_word_count"] = (
        patterns_df["response_text"].str.split().str.len()
    )

    # 2. Citation count metrics per response
    logger.info("  Computing citation count metrics...")
    citation_counts = citations_df.groupby("response_id").size().rename("num_citations")
    patterns_df = patterns_df.merge(
        citation_counts, left_on="response_id", right_index=True, how="left"
    )
    patterns_df["num_citations"] = patterns_df["num_citations"].fillna(0)

    # 3. Quality metrics (proportion of all citations)
    logger.info("  Computing quality metrics...")
    quality_stats = compute_quality_metrics(citations_df)
    patterns_df = patterns_df.merge(quality_stats, on="response_id", how="left")

    # 4. Political bias metrics (proportion of all citations)
    logger.info("  Computing political bias metrics...")
    bias_stats = compute_bias_metrics(citations_df)
    patterns_df = patterns_df.merge(bias_stats, on="response_id", how="left")

    # 5. News-specific quality metrics
    logger.info("  Computing news-specific quality metrics...")
    news_quality_stats = compute_news_quality_metrics(citations_df)
    patterns_df = patterns_df.merge(news_quality_stats, on="response_id", how="left")

    # 6. News-specific bias metrics
    logger.info("  Computing news-specific bias metrics...")
    news_bias_stats = compute_news_bias_metrics(citations_df)
    patterns_df = patterns_df.merge(news_bias_stats, on="response_id", how="left")

    # 7. Domain diversity metrics
    logger.info("  Computing domain diversity metrics...")
    domain_stats = compute_domain_metrics(citations_df)
    patterns_df = patterns_df.merge(domain_stats, on="response_id", how="left")

    # Fill NaN values for responses without citations
    numeric_cols = [
        col
        for col in patterns_df.columns
        if col.startswith(("proportion_", "news_proportion_", "num_"))
    ]
    patterns_df[numeric_cols] = patterns_df[numeric_cols].fillna(0)

    logger.info(f"Computed citation patterns for {len(patterns_df):,} responses")
    return patterns_df


def compute_quality_metrics(citations_df):
    """Compute quality-related metrics per response."""
    quality_metrics = []

    for response_id, group in citations_df.groupby("response_id"):
        metrics = {"response_id": response_id}

        # Filter news citations for quality assessment
        news_citations = group[group["domain_classification"] == "news"]
        total_cites = len(group)

        # Define quality categories
        quality_categories = ["high_quality", "low_quality", "unknown_quality"]

        if total_cites > 0:
            # Quality distribution (proportions relative to all citations)
            quality_counts = news_citations["domain_quality"].value_counts()

            for category in quality_categories:
                metrics[f"proportion_{category}"] = (
                    quality_counts.get(category, 0) / total_cites
                )
        else:
            for category in quality_categories:
                metrics[f"proportion_{category}"] = 0

        quality_metrics.append(metrics)

    return pd.DataFrame(quality_metrics)


def compute_bias_metrics(citations_df):
    """Compute political bias metrics per response."""
    bias_metrics = []

    for response_id, group in citations_df.groupby("response_id"):
        metrics = {"response_id": response_id}

        # Filter news citations for bias assessment
        news_citations = group[group["domain_classification"] == "news"]
        total_cites = len(group)

        # Define political leaning categories
        leaning_categories = [
            "left_leaning",
            "right_leaning",
            "center_leaning",
            "unknown_leaning",
        ]

        if total_cites > 0:
            # Political leaning distribution (proportions relative to all citations)
            leaning_counts = news_citations["political_leaning"].value_counts()

            for category in leaning_categories:
                metrics[f"proportion_{category}"] = (
                    leaning_counts.get(category, 0) / total_cites
                )
        else:
            for category in leaning_categories:
                metrics[f"proportion_{category}"] = 0

        bias_metrics.append(metrics)

    return pd.DataFrame(bias_metrics)


def compute_news_quality_metrics(citations_df):
    """Compute quality metrics specifically for news citations only."""
    news_quality_metrics = []

    for response_id, group in citations_df.groupby("response_id"):
        metrics = {"response_id": response_id}

        # Filter to only news citations
        news_citations = group[group["domain_classification"] == "news"]
        total_news_cites = len(news_citations)

        # Define quality categories
        quality_categories = ["high_quality", "low_quality", "unknown_quality"]

        if total_news_cites > 0:
            # Quality distribution among news citations only
            news_quality_counts = news_citations["domain_quality"].value_counts()

            for category in quality_categories:
                metrics[f"news_proportion_{category}"] = (
                    news_quality_counts.get(category, 0) / total_news_cites
                )
        else:
            for category in quality_categories:
                metrics[f"news_proportion_{category}"] = 0

        news_quality_metrics.append(metrics)

    return pd.DataFrame(news_quality_metrics)


def compute_news_bias_metrics(citations_df):
    """Compute political bias metrics specifically for news citations only."""
    news_bias_metrics = []

    for response_id, group in citations_df.groupby("response_id"):
        metrics = {"response_id": response_id}

        # Filter to only news citations
        news_citations = group[group["domain_classification"] == "news"]
        total_news_cites = len(news_citations)

        # Define political leaning categories
        leaning_categories = [
            "left_leaning",
            "right_leaning",
            "center_leaning",
            "unknown_leaning",
        ]

        if total_news_cites > 0:
            # Political leaning distribution among news citations only
            news_leaning_counts = news_citations["political_leaning"].value_counts()

            for category in leaning_categories:
                metrics[f"news_proportion_{category}"] = (
                    news_leaning_counts.get(category, 0) / total_news_cites
                )
        else:
            for category in leaning_categories:
                metrics[f"news_proportion_{category}"] = 0

        news_bias_metrics.append(metrics)

    return pd.DataFrame(news_bias_metrics)


def compute_domain_metrics(citations_df):
    """Compute domain category metrics per response."""
    domain_metrics = []

    for response_id, group in citations_df.groupby("response_id"):
        metrics = {"response_id": response_id}

        total_cites = len(group)

        # Define domain categories
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

            for category in domain_categories:
                metrics[f"proportion_{category}"] = (
                    domain_counts.get(category, 0) / total_cites
                )
        else:
            for category in domain_categories:
                metrics[f"proportion_{category}"] = 0

        domain_metrics.append(metrics)

    return pd.DataFrame(domain_metrics)


def create_minimal_output(patterns_df):
    """Create minimal output with just response_id and pattern features."""
    logger.info("Creating minimal citation patterns output...")

    # Start with response_id
    output_cols = ["response_id", "question_id"]

    # Add citation pattern features
    pattern_features = [
        col
        for col in patterns_df.columns
        if col.startswith(("proportion_", "news_proportion_", "num_citations"))
    ]

    # Add response length features
    length_features = ["response_length", "response_word_count"]

    # Combine all features
    all_features = output_cols + pattern_features + length_features

    # Filter to only available columns
    available_features = [col for col in all_features if col in patterns_df.columns]

    minimal_df = patterns_df[available_features].copy()

    logger.info(
        f"Created minimal output with {len(minimal_df):,} rows and {len(minimal_df.columns)} columns"
    )
    logger.info(f"Citation pattern features: {pattern_features}")

    return minimal_df


def analyze_citation_patterns(df):
    """Analyze the computed citation patterns."""
    logger.info("Analyzing citation patterns...")

    # Citation count analysis
    if "num_citations" in df.columns:
        citation_stats = df["num_citations"].describe()
        logger.info(f"Citation count statistics:\n{citation_stats}")

    # Quality distribution analysis
    quality_cols = [
        col for col in df.columns if "proportion_" in col and "quality" in col
    ]
    if quality_cols:
        quality_stats = df[quality_cols].describe()
        logger.info(f"Quality distribution statistics:\n{quality_stats}")

    # Political bias analysis
    bias_cols = [col for col in df.columns if "proportion_" in col and "leaning" in col]
    if bias_cols:
        bias_stats = df[bias_cols].describe()
        logger.info(f"Political bias statistics:\n{bias_stats}")

    # News-specific metrics
    news_cols = [col for col in df.columns if col.startswith("news_proportion_")]
    if news_cols:
        news_stats = df[news_cols].describe()
        logger.info(f"News-specific metrics:\n{news_stats}")


def validate_citation_patterns(df):
    """Validate the computed citation patterns."""
    logger.info("Validating citation patterns...")

    # Check required columns
    if "response_id" not in df.columns:
        raise ValueError("Missing required column: response_id")

    # Check for duplicates
    duplicates = df["response_id"].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate response IDs")

    # Check proportions are between 0 and 1
    proportion_cols = [col for col in df.columns if col.startswith("proportion_")]
    for col in proportion_cols:
        out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
        if out_of_range > 0:
            logger.warning(f"Found {out_of_range} out-of-range values in {col}")

    # Check numeric columns
    numeric_cols = ["num_citations", "response_length", "response_word_count"]
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column {col} is not numeric: {df[col].dtype}")

    # Sample output
    logger.info("Sample citation patterns output:")
    sample_row = df.iloc[0]
    for col in df.columns:
        logger.info(f"  {col}: {sample_row[col]}")

    logger.info("✅ Citation patterns validation completed")


def main():
    """Main function for citation patterns computation."""
    try:
        # Get input/output paths from Snakemake
        questions_path = snakemake.input.questions  # type: ignore
        responses_path = snakemake.input.responses  # type: ignore
        citations_path = snakemake.input.citations  # type: ignore
        output_path = snakemake.output[0]  # type: ignore

    except NameError:
        # Fallback for running outside Snakemake (for testing)
        logger.info("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        questions_path = (
            base_dir / "data/intermediate/question_analysis/english_questions.parquet"
        )
        responses_path = (
            base_dir / "data/intermediate/cleaned_arena_data/responses.parquet"
        )
        citations_path = (
            base_dir
            / "data/intermediate/citation_analysis/integrated_citations.parquet"
        )
        output_path = (
            base_dir / "data/intermediate/question_analysis/citation_patterns.parquet"
        )

    # Load data
    questions_df, responses_df, citations_df = load_analysis_data(
        questions_path, responses_path, citations_path
    )

    # Compute citation patterns
    patterns_df = compute_citation_patterns(questions_df, responses_df, citations_df)

    # Create minimal output
    minimal_patterns = create_minimal_output(patterns_df)

    # Analyze patterns
    analyze_citation_patterns(minimal_patterns)

    # Validate output
    validate_citation_patterns(minimal_patterns)

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save citation patterns
    minimal_patterns.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(minimal_patterns):,} citation patterns to {output_path}")

    # Summary statistics
    logger.info("=== CITATION PATTERNS COMPUTATION SUMMARY ===")
    logger.info(f"Input questions: {len(questions_df):,}")
    logger.info(f"Input responses: {len(responses_df):,}")
    logger.info(f"Input citations: {len(citations_df):,}")
    logger.info(f"Output patterns: {len(minimal_patterns):,}")
    logger.info(f"Pattern features: {list(minimal_patterns.columns)}")
    logger.info(f"Output file size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    logger.info("✅ Citation patterns computation completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
