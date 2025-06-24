#!/usr/bin/env python3
"""
Data Preparation Script for News Citation Preference Analysis.

This script filters the cleaned arena data to focus on competitions where
both models have cited news sources, preparing the dataset for preference analysis.
"""

import pandas as pd
from pathlib import Path
import sys


def load_data(citations_path, responses_path, threads_path):
    """Load the cleaned arena data tables from specific paths."""
    print("Loading enriched arena data...")

    # Load enriched citations data with all classification signals
    citations_df = pd.read_parquet(citations_path)
    responses_df = pd.read_parquet(responses_path)
    threads_df = pd.read_parquet(threads_path)

    print(f"Loaded {len(citations_df):,} citations")
    print(f"Loaded {len(responses_df):,} responses")
    print(f"Loaded {len(threads_df):,} threads")

    # Check available enriched features
    enriched_cols = [
        "political_leaning_score",
        "political_leaning",
        "domain_quality_score",
        "domain_quality",
        "domain_classification",
    ]
    available_cols = [col for col in enriched_cols if col in citations_df.columns]
    print(f"Available enriched features: {available_cols}")

    return citations_df, responses_df, threads_df


def get_news_citations(citations_df):
    """Get news citations using existing domain classification."""

    # Use domain classification from enriched citations
    if "domain_classification" not in citations_df.columns:
        raise ValueError(
            "Expected 'domain_classification' column in enriched citations data"
        )

    print("Using domain classification from enriched citations")
    news_citations = citations_df[
        citations_df["domain_classification"] == "news"
    ].copy()

    print(
        f"Found {len(news_citations):,} news citations from {news_citations['domain'].nunique():,} domains"
    )

    # Show breakdown of news citations by political leaning and quality if available
    if "political_leaning" in news_citations.columns:
        bias_counts = news_citations["political_leaning"].value_counts()
        print(f"Political leaning distribution: {dict(bias_counts)}")

    if "domain_quality" in news_citations.columns:
        quality_counts = news_citations["domain_quality"].value_counts()
        print(f"Quality distribution: {dict(quality_counts)}")

    return news_citations


def filter_news_competitions(citations_df, responses_df, threads_df):
    """Filter to threads where both models have news citations."""

    # Get news citations using existing classification
    news_citations = get_news_citations(citations_df)

    # Find responses with news citations
    responses_with_news = set(news_citations["response_id"])
    responses_df["has_news"] = responses_df["response_id"].isin(responses_with_news)

    print(f"Found {len(responses_with_news):,} responses with news citations")

    # Group by thread to find threads where both models cite news
    thread_news_counts = (
        responses_df.groupby("thread_id")["has_news"]
        .agg(["sum", "count"])
        .reset_index()
    )
    thread_news_counts.columns = ["thread_id", "responses_with_news", "total_responses"]

    # Filter for threads with at least one response with news
    valid_threads = thread_news_counts[(thread_news_counts["responses_with_news"] > 0)][
        "thread_id"
    ]

    print(f"Found {len(valid_threads):,} threads where at least one model cites news")

    # Filter threads data
    filtered_threads = threads_df[threads_df["thread_id"].isin(valid_threads)].copy()

    # Remove ties and invalid outcomes
    valid_winners = ["model_a", "model_b"]
    filtered_threads = filtered_threads[filtered_threads["winner"].isin(valid_winners)]

    print(f"After removing ties: {len(filtered_threads):,} valid competitions")

    # Filter responses and citations to match
    filtered_responses = responses_df[
        responses_df["thread_id"].isin(filtered_threads["thread_id"])
    ].copy()
    filtered_citations = citations_df[
        citations_df["response_id"].isin(filtered_responses["response_id"])
    ].copy()

    print(
        f"Final dataset: {len(filtered_threads):,} threads, {len(filtered_responses):,} responses, {len(filtered_citations):,} news citations"
    )

    return filtered_threads, filtered_responses, filtered_citations


def validate_competition_structure(threads_df, responses_df):
    """Validate that each thread has exactly 2 competing models."""
    print("\nValidating competition structure...")

    # Check responses per thread
    responses_per_thread = responses_df.groupby("thread_id").size()
    print(f"Responses per thread distribution:")
    print(responses_per_thread.value_counts().sort_index())

    # Check model pairs
    thread_models = (
        responses_df.groupby("thread_id")["model_name_llm"].apply(list).reset_index()
    )
    thread_models["num_models"] = thread_models["model_name_llm"].apply(len)
    thread_models["model_pair"] = thread_models["model_name_llm"].apply(
        lambda x: tuple(sorted(x)) if len(x) == 2 else None
    )

    valid_pairs = thread_models[thread_models["num_models"] == 2]
    print(f"Valid model pairs: {len(valid_pairs):,}")

    # Most common model pairs
    pair_counts = valid_pairs["model_pair"].value_counts().head(10)
    print("\nMost common model pairs:")
    for pair, count in pair_counts.items():
        print(f"  {pair[0]} vs {pair[1]}: {count:,}")

    print(f"Valid pairs length: {len(valid_pairs)}")
    print(f"Threads length: {len(threads_df)}")
    return len(valid_pairs) == len(threads_df)


def main():
    # Get input/output paths from Snakemake
    try:
        # Snakemake provides these variables
        input_citations = snakemake.input.citations
        input_responses = snakemake.input.responses
        input_threads = snakemake.input.threads

        output_threads = snakemake.output.threads
        output_responses = snakemake.output.responses
        output_citations = snakemake.output.citations

    except NameError:
        # Fallback for running outside Snakemake (for testing)
        print("Running outside Snakemake - using default paths")
        # Navigate from scripts/ to project root: scripts -> preference_analysis -> analysis -> workflow -> project_root
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        input_dir = base_dir / "data/intermediate/cleaned_arena_data"
        output_dir = base_dir / "data/intermediate/preference_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        input_citations = input_dir / "citations_enriched.parquet"
        input_responses = input_dir / "responses.parquet"
        input_threads = input_dir / "threads.parquet"

        output_threads = output_dir / "news_competitions.parquet"
        output_responses = output_dir / "news_competitions_responses.parquet"
        output_citations = output_dir / "news_competitions_citations.parquet"

    try:
        # Load data
        citations_df, responses_df, threads_df = load_data(
            input_citations, input_responses, input_threads
        )

        # Filter to news competitions
        filtered_threads, filtered_responses, filtered_citations = (
            filter_news_competitions(citations_df, responses_df, threads_df)
        )

        # Validate structure
        is_valid = validate_competition_structure(filtered_threads, filtered_responses)
        if not is_valid:
            print("WARNING: Competition structure validation failed")

        # Ensure output directories exist
        Path(output_threads).parent.mkdir(parents=True, exist_ok=True)
        Path(output_responses).parent.mkdir(parents=True, exist_ok=True)
        Path(output_citations).parent.mkdir(parents=True, exist_ok=True)

        # Save filtered data
        filtered_threads.to_parquet(output_threads, index=False)
        filtered_responses.to_parquet(output_responses, index=False)
        filtered_citations.to_parquet(output_citations, index=False)

        print(f"\nSaved prepared data:")
        print(f"  Threads: {output_threads}")
        print(f"  Responses: {output_responses}")
        print(f"  Citations: {output_citations}")

        # Generate summary statistics
        summary = {
            "total_threads": len(filtered_threads),
            "total_responses": len(filtered_responses),
            "total_citations": len(filtered_citations),
            "unique_models": filtered_responses["model_name_llm"].nunique(),
            "unique_domains": filtered_citations["domain"].nunique(),
            "winner_distribution": filtered_threads["winner"].value_counts().to_dict(),
        }

        print(f"\nSummary statistics:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
