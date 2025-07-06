#!/usr/bin/env python3
"""
Extract Question Features Script for Question Analysis Pipeline.

This script extracts features from questions including question intent,
client country, and basic question characteristics.
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


def extract_question_characteristics(questions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract basic question characteristics.

    Args:
        questions_df: DataFrame with questions

    Returns:
        DataFrame with question characteristics
    """
    logger.info("Extracting question characteristics...")

    df = questions_df.copy()

    # Question length in characters
    df["question_length_chars"] = df["user_query"].str.len()

    # Question length in words (simple split by whitespace)
    df["question_length_words"] = df["user_query"].str.split().str.len()

    # Question turn number (already available)
    # Keep turn_number as is

    logger.info(f"Extracted characteristics for {len(df):,} questions")
    return df


def load_thread_metadata(threads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and prepare thread metadata for merging.

    Args:
        threads_df: DataFrame with thread metadata

    Returns:
        DataFrame with relevant thread metadata
    """
    logger.info("Loading thread metadata...")

    # Select relevant columns for question features
    metadata_cols = ["thread_id"]

    # Add intent if available
    if "intent" in threads_df.columns:
        metadata_cols.append("intent")
        logger.info("Found intent column in threads")
    else:
        logger.warning("Intent column not found in threads")

    # Add country if available
    if "country" in threads_df.columns:
        metadata_cols.append("country")
        logger.info("Found country column in threads")
    elif "client_country" in threads_df.columns:
        metadata_cols.append("client_country")
        logger.info("Found client_country column in threads")
    else:
        logger.warning("Country column not found in threads")

    # Add any other relevant metadata columns
    available_cols = [col for col in metadata_cols if col in threads_df.columns]
    thread_metadata = threads_df[available_cols].copy()

    logger.info(
        f"Loaded metadata for {len(thread_metadata):,} threads with columns: {available_cols}"
    )
    return thread_metadata


def merge_with_thread_metadata(
    questions_df: pd.DataFrame, threads_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge questions with thread metadata.

    Args:
        questions_df: DataFrame with questions
        threads_df: DataFrame with thread metadata

    Returns:
        DataFrame with merged data
    """
    logger.info("Merging questions with thread metadata...")

    # Load thread metadata
    thread_metadata = load_thread_metadata(threads_df)

    # Merge on thread_id
    initial_count = len(questions_df)
    merged_df = questions_df.merge(thread_metadata, on="thread_id", how="left")

    if len(merged_df) != initial_count:
        logger.warning(
            f"Row count changed during merge: {initial_count} -> {len(merged_df)}"
        )

    # Check for missing metadata
    if "intent" in merged_df.columns:
        missing_intent = merged_df["intent"].isna().sum()
        logger.info(
            f"Questions with missing intent: {missing_intent:,} ({missing_intent / len(merged_df) * 100:.1f}%)"
        )

    if "country" in merged_df.columns:
        missing_country = merged_df["country"].isna().sum()
        logger.info(
            f"Questions with missing country: {missing_country:,} ({missing_country / len(merged_df) * 100:.1f}%)"
        )
    elif "client_country" in merged_df.columns:
        missing_country = merged_df["client_country"].isna().sum()
        logger.info(
            f"Questions with missing client_country: {missing_country:,} ({missing_country / len(merged_df) * 100:.1f}%)"
        )

    logger.info(f"Merged {len(merged_df):,} questions with thread metadata")
    return merged_df


def create_features_dataframe(questions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create final features DataFrame with only question_id and features.

    Args:
        questions_df: DataFrame with all question data and metadata

    Returns:
        DataFrame with question_id and features only
    """
    logger.info("Creating features DataFrame...")

    # Start with question_id
    features_cols = ["question_id"]

    # Add available features
    feature_mappings = {
        "intent": "question_intent",
        "country": "client_country",
        "client_country": "client_country",
        "question_length_chars": "question_length_chars",
        "question_length_words": "question_length_words",
        "turn_number": "turn_number",
    }

    features_df = questions_df[["question_id"]].copy()

    for original_col, feature_col in feature_mappings.items():
        if original_col in questions_df.columns:
            features_df[feature_col] = questions_df[original_col]
            features_cols.append(feature_col)
            logger.info(f"Added feature: {feature_col}")

    logger.info(
        f"Created features DataFrame with {len(features_df):,} rows and {len(features_df.columns)} columns"
    )
    return features_df


def analyze_features(df: pd.DataFrame):
    """Analyze the extracted features."""
    logger.info("Analyzing extracted features...")

    # Question characteristics analysis
    if "question_length_chars" in df.columns:
        char_stats = df["question_length_chars"].describe()
        logger.info(f"Question length (chars) statistics:\n{char_stats}")

    if "question_length_words" in df.columns:
        word_stats = df["question_length_words"].describe()
        logger.info(f"Question length (words) statistics:\n{word_stats}")

    # Intent distribution
    if "question_intent" in df.columns:
        intent_counts = df["question_intent"].value_counts()
        logger.info(f"Question intent distribution:\n{intent_counts.head(10)}")

    # Country distribution
    if "client_country" in df.columns:
        country_counts = df["client_country"].value_counts()
        logger.info(f"Client country distribution:\n{country_counts.head(10)}")

    # Turn number distribution
    if "turn_number" in df.columns:
        turn_counts = df["turn_number"].value_counts().sort_index()
        logger.info(f"Turn number distribution:\n{turn_counts.head(10)}")


def validate_output(df: pd.DataFrame):
    """Validate the final features DataFrame."""
    logger.info("Validating features output...")

    # Check required columns
    if "question_id" not in df.columns:
        raise ValueError("Missing required column: question_id")

    # Check for duplicates
    duplicates = df["question_id"].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate question IDs")

    # Check data types
    numeric_cols = ["question_length_chars", "question_length_words", "turn_number"]
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column {col} is not numeric: {df[col].dtype}")

    # Sample output
    logger.info("Sample features output:")
    sample_row = df.iloc[0]
    for col in df.columns:
        logger.info(f"  {col}: {sample_row[col]}")

    logger.info("✅ Features validation completed")


def main():
    """Main function for question feature extraction."""
    try:
        # Get input/output paths from Snakemake
        input_path = snakemake.input.questions  # type: ignore
        threads_path = snakemake.input.threads  # type: ignore
        output_path = snakemake.output[0]  # type: ignore

    except NameError:
        # Fallback for running outside Snakemake (for testing)
        logger.info("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        input_path = (
            base_dir / "data/intermediate/question_analysis/english_questions.parquet"
        )
        output_path = (
            base_dir / "data/intermediate/question_analysis/question_features.parquet"
        )
        threads_path = base_dir / "data/intermediate/cleaned_arena_data/threads.parquet"

    # Load questions
    logger.info(f"Loading questions from {input_path}")
    questions_df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(questions_df):,} questions")

    # Load threads
    logger.info(f"Loading threads from {threads_path}")
    threads_df = pd.read_parquet(threads_path)
    logger.info(f"Loaded {len(threads_df):,} threads")

    # Extract question characteristics
    questions_with_chars = extract_question_characteristics(questions_df)

    # Merge with thread metadata
    questions_with_metadata = merge_with_thread_metadata(
        questions_with_chars, threads_df
    )

    # Create features DataFrame
    features_df = create_features_dataframe(questions_with_metadata)

    # Analyze features
    analyze_features(features_df)

    # Validate output
    validate_output(features_df)

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save features
    features_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(features_df):,} question features to {output_path}")

    # Summary statistics
    logger.info("=== QUESTION FEATURE EXTRACTION SUMMARY ===")
    logger.info(f"Input questions: {len(questions_df):,}")
    logger.info(f"Output features: {len(features_df):,}")
    logger.info(f"Feature columns: {list(features_df.columns)}")
    logger.info(f"Output file size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    logger.info("✅ Question feature extraction completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
