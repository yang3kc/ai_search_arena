#!/usr/bin/env python3
"""
Filter English Questions Script for Question Analysis Pipeline.

This script filters questions to include only English-only questions based on
the language information from threads metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import ast

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_language_list(lang_data):
    """
    Parse language list data to Python list.

    Args:
        lang_data: Language data (could be numpy array, list, or string)

    Returns:
        List of languages or None if parsing fails
    """
    try:
        # Handle numpy arrays
        if hasattr(lang_data, "tolist"):
            return lang_data.tolist()
        # Handle regular lists
        elif isinstance(lang_data, list):
            return lang_data
        # Handle string representation of lists
        elif isinstance(lang_data, str):
            return ast.literal_eval(lang_data)
        else:
            return None
    except (ValueError, SyntaxError, AttributeError):
        return None


def filter_english_only_threads(threads_df):
    """
    Filter threads to include only those with English as the only language.

    Args:
        threads_df: DataFrame with threads data including languages column

    Returns:
        Filtered DataFrame with English-only threads
    """
    logger.info(f"Starting with {len(threads_df):,} threads")

    # Create working copy
    df = threads_df.copy()

    # Parse language lists
    logger.info("Parsing language information...")
    df["parsed_languages"] = df["languages"].apply(parse_language_list)

    # Filter out threads where language parsing failed
    initial_count = len(df)
    df = df[df["parsed_languages"].notna()].copy()
    logger.info(
        f"Removed {initial_count - len(df):,} threads with unparseable languages"
    )

    # Filter to English-only threads
    def is_english_only(lang_list):
        if lang_list is None:
            return False
        # Check if list contains only 'English'
        return len(lang_list) == 1 and lang_list[0] == "English"

    initial_count = len(df)
    english_only_df = df[df["parsed_languages"].apply(is_english_only)].copy()
    logger.info(
        f"Filtered to {len(english_only_df):,} English-only threads ({initial_count - len(english_only_df):,} multilingual/non-English removed)"
    )

    # Remove temporary column
    english_only_df = english_only_df.drop(columns=["parsed_languages"])

    return english_only_df


def filter_questions_by_english_threads(questions_df, english_threads_df):
    """
    Filter questions to include only those from English-only threads.

    Args:
        questions_df: DataFrame with questions
        english_threads_df: DataFrame with English-only threads

    Returns:
        Filtered questions DataFrame
    """
    logger.info(f"Starting with {len(questions_df):,} questions")

    # Get English-only thread IDs
    english_thread_ids = set(english_threads_df["thread_id"].unique())
    logger.info(f"Found {len(english_thread_ids):,} English-only threads")

    # Filter questions to English-only threads
    initial_count = len(questions_df)
    english_questions = questions_df[
        questions_df["thread_id"].isin(english_thread_ids)
    ].copy()
    logger.info(
        f"Filtered to {len(english_questions):,} questions from English-only threads ({initial_count - len(english_questions):,} removed)"
    )

    # Basic quality filters
    logger.info("Applying basic quality filters...")

    # Remove null questions
    initial_count = len(english_questions)
    english_questions = english_questions[
        english_questions["user_query"].notna()
    ].copy()
    logger.info(f"Removed {initial_count - len(english_questions):,} null questions")

    # Add question length
    english_questions["question_length"] = english_questions["user_query"].str.len()

    # Remove too long questions (>1000 characters)
    initial_count = len(english_questions)
    english_questions = english_questions[
        english_questions["question_length"] <= 1000
    ].copy()
    logger.info(
        f"Removed {initial_count - len(english_questions):,} overly long questions"
    )

    return english_questions


def analyze_language_distribution(threads_df):
    """Analyze the distribution of languages in threads."""
    logger.info("Analyzing language distribution...")

    # Parse all language lists
    threads_df["parsed_languages"] = threads_df["languages"].apply(parse_language_list)
    valid_threads = threads_df[threads_df["parsed_languages"].notna()]

    # Count by number of languages
    lang_counts = valid_threads["parsed_languages"].apply(lambda x: len(x) if x else 0)
    logger.info(
        f"Language count distribution:\n{lang_counts.value_counts().sort_index()}"
    )

    # Most common language combinations
    lang_combinations = valid_threads["parsed_languages"].apply(
        lambda x: tuple(sorted(x)) if x else None
    )
    top_combinations = lang_combinations.value_counts().head(10)
    logger.info(f"Top 10 language combinations:\n{top_combinations}")

    # English-only vs others
    english_only_count = (
        valid_threads["parsed_languages"]
        .apply(lambda x: len(x) == 1 and x[0] == "English" if x else False)
        .sum()
    )
    logger.info(
        f"English-only threads: {english_only_count:,} / {len(valid_threads):,} ({english_only_count / len(valid_threads) * 100:.1f}%)"
    )


def validate_filtered_questions(df):
    """Validate the filtered questions dataset."""
    logger.info("Validating filtered questions...")

    # Check for required columns
    required_cols = [
        "question_id",
        "thread_id",
        "turn_number",
        "user_query",
        "question_role",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for duplicates
    duplicates = df["question_id"].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate question IDs")

    # Length statistics
    length_stats = df["question_length"].describe()
    logger.info(f"Question length statistics:\n{length_stats}")

    # Thread distribution
    thread_counts = df["thread_id"].value_counts()
    logger.info(
        f"Questions per thread - Mean: {thread_counts.mean():.1f}, Median: {thread_counts.median():.1f}"
    )

    # Sample questions for manual review
    logger.info("Sample filtered questions:")
    sample_questions = df.sample(n=min(5, len(df)))
    for _, row in sample_questions.iterrows():
        logger.info(f"  {row['question_id']}: {row['user_query'][:100]}...")

    return True


def main():
    """Main function for English question filtering."""
    try:
        # Get input/output paths from Snakemake
        input_questions = snakemake.input.questions
        input_threads = snakemake.input.threads
        output_path = snakemake.output[0]

    except NameError:
        # Fallback for running outside Snakemake (for testing)
        logger.info("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        input_questions = (
            base_dir / "data/intermediate/cleaned_arena_data/questions.parquet"
        )
        input_threads = (
            base_dir / "data/intermediate/cleaned_arena_data/threads.parquet"
        )
        output_path = (
            base_dir / "data/intermediate/question_analysis/english_questions.parquet"
        )

    # Load data
    logger.info(f"Loading questions from {input_questions}")
    questions_df = pd.read_parquet(input_questions)
    logger.info(f"Loaded {len(questions_df):,} questions")

    logger.info(f"Loading threads from {input_threads}")
    threads_df = pd.read_parquet(input_threads)
    logger.info(f"Loaded {len(threads_df):,} threads")

    # Analyze language distribution
    analyze_language_distribution(threads_df)

    # Filter to English-only threads
    english_threads = filter_english_only_threads(threads_df)

    # Filter questions to English-only threads
    english_questions = filter_questions_by_english_threads(
        questions_df, english_threads
    )

    # Validate results
    validate_filtered_questions(english_questions)

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save filtered questions
    english_questions.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(english_questions):,} English questions to {output_path}")

    # Summary statistics
    logger.info("=== ENGLISH QUESTION FILTERING SUMMARY ===")
    logger.info(f"Original questions: {len(questions_df):,}")
    logger.info(f"Original threads: {len(threads_df):,}")
    logger.info(f"English-only threads: {len(english_threads):,}")
    logger.info(f"English questions: {len(english_questions):,}")
    logger.info(
        f"Question filtering rate: {len(english_questions) / len(questions_df) * 100:.1f}%"
    )
    logger.info(
        f"Thread filtering rate: {len(english_threads) / len(threads_df) * 100:.1f}%"
    )
    logger.info(
        f"Average question length: {english_questions['question_length'].mean():.1f} characters"
    )

    logger.info("✅ English question filtering completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
