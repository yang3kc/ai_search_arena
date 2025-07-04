#!/usr/bin/env python3
"""
Data Integration Script for Question Analysis Pipeline.

This script merges question features, embeddings, and citation patterns
with response and thread metadata to create an analysis-ready dataset.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_input_data(
    question_embeddings_path,
    question_features_path,
    citation_patterns_path,
    threads_path,
    responses_path,
    question_topics_path,
    question_topic_probabilities_path,
    topic_info_path,
):
    """Load all input data files for integration."""
    logger.info("Loading input data files...")

    # Load question embeddings
    embeddings_df = pd.read_parquet(question_embeddings_path)
    logger.info(f"Loaded {len(embeddings_df):,} question embeddings")

    # Load question features
    features_df = pd.read_parquet(question_features_path)
    logger.info(f"Loaded {len(features_df):,} question features")

    # Load citation patterns
    patterns_df = pd.read_parquet(citation_patterns_path)
    logger.info(f"Loaded {len(patterns_df):,} citation patterns")

    # Load responses (needed for thread_id and model info)
    responses_df = pd.read_parquet(responses_path)
    logger.info(f"Loaded {len(responses_df):,} responses")

    # Load thread metadata
    threads_df = pd.read_parquet(threads_path)
    logger.info(f"Loaded {len(threads_df):,} threads")

    # Load question topics
    topics_df = pd.read_parquet(question_topics_path)
    logger.info(f"Loaded {len(topics_df):,} question topics")

    # Load question topic probabilities
    topic_probs_df = pd.read_parquet(question_topic_probabilities_path)
    logger.info(f"Loaded {len(topic_probs_df):,} question topic probabilities")

    # Load topic info
    with open(topic_info_path, "r") as f:
        topic_info = json.load(f)
    logger.info(f"Loaded topic info with {len(topic_info['topics'])} topics")

    return (
        embeddings_df,
        features_df,
        patterns_df,
        responses_df,
        threads_df,
        topics_df,
        topic_probs_df,
        topic_info,
    )


def merge_question_data(embeddings_df, features_df):
    """Merge question embeddings with question features."""
    logger.info("Merging question embeddings with features...")

    # Merge on question_id
    question_data = embeddings_df.merge(features_df, on="question_id", how="inner")

    logger.info(f"Merged question data: {len(question_data):,} rows")

    # Check for data loss
    if len(question_data) != len(embeddings_df):
        logger.warning(
            f"Data loss during question merge: {len(embeddings_df)} -> {len(question_data)}"
        )

    return question_data


def merge_topic_data(question_data, topics_df, topic_probs_df, topic_info):
    """Merge question data with topic modeling results."""
    logger.info("Merging question data with topic modeling results...")

    # Create topic label mapping
    topic_labels = {}
    for topic_data in topic_info["topics"]:
        topic_id = topic_data["topic"]
        topic_labels[topic_id] = topic_data.get("openai_label", f"Topic {topic_id}")

    # Merge topics with probabilities
    topics_with_probs = topics_df.merge(topic_probs_df, on="question_id", how="inner")

    # Add topic labels
    topics_with_probs["topic_label"] = topics_with_probs["topic"].map(topic_labels)

    logger.info(
        f"Merged {len(topics_with_probs):,} topic assignments with probabilities"
    )

    # Merge with question data (left join to preserve all questions)
    question_data_with_topics = question_data.merge(
        topics_with_probs, on="question_id", how="left"
    )

    logger.info(
        f"Merged question data with topics: {len(question_data_with_topics):,} rows"
    )

    # Check coverage
    questions_with_topics = (~question_data_with_topics["topic"].isna()).sum()
    coverage_pct = (questions_with_topics / len(question_data_with_topics)) * 100
    logger.info(
        f"Topic coverage: {questions_with_topics:,}/{len(question_data_with_topics):,} ({coverage_pct:.1f}%)"
    )

    return question_data_with_topics


def add_response_metadata(patterns_df, responses_df):
    """Add response metadata including thread_id and model information."""
    logger.info("Adding response metadata...")

    # Select relevant response metadata
    response_metadata_cols = [
        "response_id",
        "thread_id",
        "model_name_raw",
        "model_side",
    ]

    # Filter to available columns
    available_cols = [
        col for col in response_metadata_cols if col in responses_df.columns
    ]
    response_metadata = responses_df[available_cols].copy()

    # Merge patterns with response metadata
    patterns_with_metadata = patterns_df.merge(
        response_metadata, on="response_id", how="left"
    )

    # Check for missing response metadata
    missing_metadata = (
        patterns_with_metadata["thread_id"].isna().sum()
        if "thread_id" in patterns_with_metadata.columns
        else 0
    )
    if missing_metadata > 0:
        logger.warning(
            f"Citation patterns with missing response metadata: {missing_metadata:,}"
        )

    logger.info(
        f"Added response metadata: {[col for col in available_cols if col != 'response_id']}"
    )
    return patterns_with_metadata


def merge_with_citation_patterns(question_data, patterns_df):
    """Merge question data with citation patterns."""
    logger.info("Merging question data with citation patterns...")

    # Merge on question_id
    integrated_data = question_data.merge(patterns_df, on="question_id", how="left")

    logger.info(
        f"Integrated data after citation patterns merge: {len(integrated_data):,} rows"
    )

    # Check for questions without responses/citations
    no_citations = integrated_data["response_id"].isna().sum()
    if no_citations > 0:
        logger.info(f"Questions without citation patterns: {no_citations:,}")

    return integrated_data


def add_thread_metadata(integrated_data, threads_df):
    """Add thread-level metadata to the integrated dataset."""
    logger.info("Adding thread metadata...")

    # Select relevant thread metadata
    thread_cols = ["thread_id"]

    # Add available thread metadata columns
    available_thread_cols = [
        "winner",
        "primary_intent",
        "secondary_intent",
        "total_turns",
        "timestamp",
    ]

    for col in available_thread_cols:
        if col in threads_df.columns:
            thread_cols.append(col)
            logger.info(f"Adding thread metadata: {col}")

    thread_metadata = threads_df[thread_cols].copy()

    # Merge thread metadata
    initial_count = len(integrated_data)
    integrated_with_threads = integrated_data.merge(
        thread_metadata, on="thread_id", how="left"
    )

    if len(integrated_with_threads) != initial_count:
        logger.warning(
            f"Row count changed during thread merge: {initial_count} -> {len(integrated_with_threads)}"
        )

    # Check for missing thread metadata
    missing_threads = integrated_with_threads["thread_id"].isna().sum()
    if missing_threads > 0:
        logger.warning(f"Responses with missing thread metadata: {missing_threads:,}")

    logger.info(f"Final integrated dataset: {len(integrated_with_threads):,} rows")
    return integrated_with_threads


def handle_missing_data(integrated_data):
    """Handle missing data and create analysis-ready dataset."""
    logger.info("Handling missing data...")

    # Count missing values
    missing_summary = integrated_data.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]

    if len(missing_cols) > 0:
        logger.info("Missing value summary:")
        for col, count in missing_cols.items():
            pct = (count / len(integrated_data)) * 100
            logger.info(f"  {col}: {count:,} ({pct:.1f}%)")

    # Handle citation pattern nulls (questions without responses)
    citation_cols = [
        col
        for col in integrated_data.columns
        if col.startswith(("proportion_", "news_proportion_", "num_citations"))
    ]

    if citation_cols:
        # Fill NaN citation patterns with 0 (no citations)
        integrated_data[citation_cols] = integrated_data[citation_cols].fillna(0)
        logger.info(
            f"Filled {len(citation_cols)} citation pattern columns with 0 for missing values"
        )

    # Handle response length nulls
    response_length_cols = ["response_length", "response_word_count"]
    for col in response_length_cols:
        if col in integrated_data.columns:
            missing_count = integrated_data[col].isna().sum()
            if missing_count > 0:
                integrated_data[col] = integrated_data[col].fillna(0)
                logger.info(f"Filled {missing_count:,} missing values in {col} with 0")

    # Handle topic data nulls (questions without topic assignments)
    topic_cols = [
        col
        for col in integrated_data.columns
        if col.startswith("topic_") and col.endswith("_prob")
    ]

    if topic_cols:
        # Fill NaN topic probabilities with 0 (no topic assignment)
        integrated_data[topic_cols] = integrated_data[topic_cols].fillna(0)
        logger.info(
            f"Filled {len(topic_cols)} topic probability columns with 0 for missing values"
        )

    # Handle topic and topic_label nulls
    if "topic" in integrated_data.columns:
        topic_missing = integrated_data["topic"].isna().sum()
        if topic_missing > 0:
            integrated_data["topic"] = integrated_data["topic"].fillna(
                -2
            )  # Use -2 to distinguish from outliers (-1)
            logger.info(f"Filled {topic_missing:,} missing topic assignments with -2")

    if "topic_label" in integrated_data.columns:
        topic_label_missing = integrated_data["topic_label"].isna().sum()
        if topic_label_missing > 0:
            integrated_data["topic_label"] = integrated_data["topic_label"].fillna(
                "No Topic Assignment"
            )
            logger.info(
                f"Filled {topic_label_missing:,} missing topic labels with 'No Topic Assignment'"
            )

    return integrated_data


def validate_integrated_data(integrated_data):
    """Validate the final integrated dataset."""
    logger.info("Validating integrated dataset...")

    # Check required columns
    required_cols = ["question_id"]
    missing_required = [
        col for col in required_cols if col not in integrated_data.columns
    ]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Check for duplicates
    duplicates = integrated_data["question_id"].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate question IDs")

    # Check embedding dimensions
    embedding_cols = [
        col for col in integrated_data.columns if col.startswith("embedding_dim_")
    ]
    if embedding_cols:
        logger.info(f"Dataset includes {len(embedding_cols)} embedding dimensions")

    # Check feature coverage
    feature_cols = [
        col
        for col in integrated_data.columns
        if col
        in [
            "client_country",
            "question_length_chars",
            "question_length_words",
            "turn_number",
        ]
    ]
    logger.info(
        f"Dataset includes {len(feature_cols)} question features: {feature_cols}"
    )

    # Check citation pattern coverage
    citation_cols = [
        col
        for col in integrated_data.columns
        if col.startswith(("proportion_", "news_proportion_", "num_citations"))
    ]
    logger.info(f"Dataset includes {len(citation_cols)} citation pattern features")

    # Check topic coverage
    topic_cols = [
        col
        for col in integrated_data.columns
        if col.startswith("topic_") and col.endswith("_prob")
    ]
    if topic_cols:
        logger.info(f"Dataset includes {len(topic_cols)} topic probability features")

    if "topic" in integrated_data.columns:
        topic_stats = integrated_data["topic"].value_counts()
        logger.info(f"Topic distribution: {len(topic_stats)} unique topics")
        logger.info(
            f"Most common topic: {topic_stats.index[0]} ({topic_stats.iloc[0]:,} questions)"
        )
        if -2 in topic_stats.index:
            logger.info(f"Questions without topic assignment: {topic_stats[-2]:,}")
        if -1 in topic_stats.index:
            logger.info(f"Topic outliers: {topic_stats[-1]:,}")

    # Data quality checks
    if "num_citations" in integrated_data.columns:
        citation_stats = integrated_data["num_citations"].describe()
        logger.info(f"Citation count statistics:\n{citation_stats}")

    # Sample output
    logger.info("Sample integrated data:")
    sample_cols = [
        "question_id",
        "client_country",
        "num_citations",
        "response_length",
        "topic",
        "topic_label",
    ]
    available_sample_cols = [
        col for col in sample_cols if col in integrated_data.columns
    ]
    if available_sample_cols:
        sample_row = integrated_data[available_sample_cols].iloc[0]
        for col in available_sample_cols:
            logger.info(f"  {col}: {sample_row[col]}")

    logger.info("✅ Integrated data validation completed")


def analyze_data_coverage(integrated_data):
    """Analyze data coverage and completeness."""
    logger.info("Analyzing data coverage...")

    total_rows = len(integrated_data)

    # Question feature coverage
    if "client_country" in integrated_data.columns:
        country_coverage = (~integrated_data["client_country"].isna()).sum()
        logger.info(
            f"Client country coverage: {country_coverage:,}/{total_rows:,} ({country_coverage / total_rows * 100:.1f}%)"
        )

    # Citation coverage
    if "response_id" in integrated_data.columns:
        response_coverage = (~integrated_data["response_id"].isna()).sum()
        logger.info(
            f"Response coverage: {response_coverage:,}/{total_rows:,} ({response_coverage / total_rows * 100:.1f}%)"
        )

    if "num_citations" in integrated_data.columns:
        has_citations = (integrated_data["num_citations"] > 0).sum()
        logger.info(
            f"Questions with citations: {has_citations:,}/{total_rows:,} ({has_citations / total_rows * 100:.1f}%)"
        )

    # Thread metadata coverage
    if "primary_intent" in integrated_data.columns:
        intent_coverage = (~integrated_data["primary_intent"].isna()).sum()
        logger.info(
            f"Primary intent coverage: {intent_coverage:,}/{total_rows:,} ({intent_coverage / total_rows * 100:.1f}%)"
        )

    # Topic coverage
    if "topic" in integrated_data.columns:
        topic_coverage = (
            integrated_data["topic"] >= 0
        ).sum()  # Topics >= 0 (exclude -1 outliers and -2 missing)
        logger.info(
            f"Topic coverage: {topic_coverage:,}/{total_rows:,} ({topic_coverage / total_rows * 100:.1f}%)"
        )


def main():
    """Main function for data integration."""
    try:
        # Get input/output paths from Snakemake
        question_embeddings_path = snakemake.input.question_embeddings  # type: ignore
        question_features_path = snakemake.input.question_features  # type: ignore
        citation_patterns_path = snakemake.input.citation_patterns  # type: ignore
        threads_path = snakemake.input.threads  # type: ignore
        responses_path = snakemake.input.responses  # type: ignore
        question_topics_path = snakemake.input.question_topics  # type: ignore
        question_topic_probabilities_path = snakemake.input.question_topic_probabilities  # type: ignore
        topic_info_path = snakemake.input.topic_info  # type: ignore
        output_path = snakemake.output[0]  # type: ignore

    except NameError:
        # Fallback for running outside Snakemake (for testing)
        logger.info("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent

        question_embeddings_path = (
            base_dir / "data/intermediate/question_analysis/question_embeddings.parquet"
        )
        question_features_path = (
            base_dir / "data/intermediate/question_analysis/question_features.parquet"
        )
        citation_patterns_path = (
            base_dir / "data/intermediate/question_analysis/citation_patterns.parquet"
        )
        threads_path = base_dir / "data/intermediate/cleaned_arena_data/threads.parquet"
        responses_path = (
            base_dir / "data/intermediate/cleaned_arena_data/responses.parquet"
        )
        question_topics_path = (
            base_dir / "data/intermediate/question_analysis/question_topics.parquet"
        )
        question_topic_probabilities_path = (
            base_dir
            / "data/intermediate/question_analysis/question_topic_probabilities.parquet"
        )
        topic_info_path = (
            base_dir / "data/intermediate/question_analysis/topic_info.json"
        )
        output_path = (
            base_dir
            / "data/intermediate/question_analysis/integrated_analysis_data.parquet"
        )

    # Load input data
    (
        embeddings_df,
        features_df,
        patterns_df,
        responses_df,
        threads_df,
        topics_df,
        topic_probs_df,
        topic_info,
    ) = load_input_data(
        question_embeddings_path,
        question_features_path,
        citation_patterns_path,
        threads_path,
        responses_path,
        question_topics_path,
        question_topic_probabilities_path,
        topic_info_path,
    )

    # Step 1: Merge question embeddings with features
    question_data = merge_question_data(embeddings_df, features_df)

    # Step 1.5: Merge question data with topic modeling results
    question_data_with_topics = merge_topic_data(
        question_data, topics_df, topic_probs_df, topic_info
    )

    # Step 2: Add response metadata to citation patterns
    patterns_with_metadata = add_response_metadata(patterns_df, responses_df)

    # Step 3: Merge question data with citation patterns
    integrated_data = merge_with_citation_patterns(
        question_data_with_topics, patterns_with_metadata
    )

    # Step 4: Add thread metadata
    integrated_with_threads = add_thread_metadata(integrated_data, threads_df)

    # Step 5: Handle missing data
    analysis_ready_data = handle_missing_data(integrated_with_threads)

    # Step 6: Validate integrated data
    validate_integrated_data(analysis_ready_data)

    # Step 7: Analyze data coverage
    analyze_data_coverage(analysis_ready_data)

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save integrated dataset
    analysis_ready_data.to_parquet(output_path, index=False)
    logger.info(
        f"Saved {len(analysis_ready_data):,} integrated records to {output_path}"
    )

    # Summary statistics
    logger.info("=== DATA INTEGRATION SUMMARY ===")
    logger.info(f"Input question embeddings: {len(embeddings_df):,}")
    logger.info(f"Input question features: {len(features_df):,}")
    logger.info(f"Input citation patterns: {len(patterns_df):,}")
    logger.info(f"Input threads: {len(threads_df):,}")
    logger.info(f"Input topic assignments: {len(topics_df):,}")
    logger.info(f"Input topic probabilities: {len(topic_probs_df):,}")
    logger.info(f"Topic metadata: {len(topic_info['topics'])} topics")
    logger.info(
        f"Final integrated dataset: {len(analysis_ready_data):,} rows, {len(analysis_ready_data.columns)} columns"
    )
    logger.info(f"Output file size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Feature summary
    embedding_cols = [
        col for col in analysis_ready_data.columns if col.startswith("embedding_dim_")
    ]
    feature_cols = [
        col
        for col in analysis_ready_data.columns
        if col
        in [
            "client_country",
            "model_name_raw",
            "question_length_chars",
            "question_length_words",
            "turn_number",
        ]
    ]
    citation_cols = [
        col
        for col in analysis_ready_data.columns
        if col.startswith(("proportion_", "news_proportion_", "num_citations"))
    ]
    topic_cols = [
        col
        for col in analysis_ready_data.columns
        if col.startswith("topic_") and col.endswith("_prob")
    ]
    topic_metadata_cols = [
        col for col in analysis_ready_data.columns if col in ["topic", "topic_label"]
    ]

    logger.info("Features included:")
    logger.info(f"  - {len(embedding_cols)} embedding dimensions")
    logger.info(f"  - {len(feature_cols)} question features")
    logger.info(f"  - {len(citation_cols)} citation pattern features")
    logger.info(f"  - {len(topic_cols)} topic probability features")
    logger.info(f"  - {len(topic_metadata_cols)} topic metadata features")

    logger.info("✅ Data integration completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
