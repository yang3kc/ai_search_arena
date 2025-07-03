#!/usr/bin/env python3
"""
Filter English Questions Script for Question Analysis Pipeline.

This script filters questions to include only English language questions
with sufficient confidence scores for downstream analysis.
"""

import pandas as pd
from pathlib import Path
import logging
import sys
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import warnings

# Suppress langdetect warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def detect_language_batch(texts, min_confidence=0.8):
    """
    Detect language for a batch of texts with confidence scoring.

    Args:
        texts: List of text strings
        min_confidence: Minimum confidence threshold for language detection

    Returns:
        List of tuples (is_english, confidence_score)
    """
    results = []

    for text in texts:
        try:
            if pd.isna(text) or len(str(text).strip()) < 10:
                # Too short for reliable detection
                results.append((False, 0.0))
                continue

            # Clean text for detection
            clean_text = str(text).strip()

            # Use langdetect
            detected_lang = detect(clean_text)

            # Simple confidence estimation based on text characteristics
            # (langdetect doesn't provide confidence directly)
            confidence = estimate_confidence(clean_text, detected_lang)

            is_english = (detected_lang == "en") and (confidence >= min_confidence)
            results.append((is_english, confidence))

        except (LangDetectException, Exception):
            # Detection failed
            results.append((False, 0.0))

    return results


def estimate_confidence(text, detected_lang):
    """
    Estimate confidence score for language detection.
    This is a simple heuristic since langdetect doesn't provide confidence.
    """
    # Base confidence
    confidence = 0.7

    # Longer texts generally more reliable
    if len(text) > 50:
        confidence += 0.1
    if len(text) > 100:
        confidence += 0.1

    # Check for common English patterns
    english_indicators = [
        "the ",
        "and ",
        "is ",
        "are ",
        "was ",
        "were ",
        "what ",
        "how ",
        "why ",
        "when ",
        "where ",
        "can ",
        "could ",
        "would ",
        "should ",
    ]

    if detected_lang == "en":
        english_count = sum(
            1 for indicator in english_indicators if indicator in text.lower()
        )
        if english_count >= 2:
            confidence += 0.1
        if english_count >= 4:
            confidence += 0.1

    return min(confidence, 1.0)


def filter_english_questions(questions_df, min_confidence=0.8, max_length=10000):
    """
    Filter questions to include only English questions.

    Args:
        questions_df: DataFrame with questions
        min_confidence: Minimum language detection confidence
        max_length: Maximum question length

    Returns:
        Filtered DataFrame with additional language metadata
    """
    logger.info(f"Starting with {len(questions_df):,} questions")

    # Create working copy
    df = questions_df.copy()

    # Basic filtering
    logger.info("Applying basic filters...")

    # Remove null questions
    initial_count = len(df)
    df = df[df["user_query"].notna()].copy()
    logger.info(f"Removed {initial_count - len(df):,} null questions")

    # Remove too long questions
    initial_count = len(df)
    df["question_length"] = df["user_query"].str.len()
    df = df[df["question_length"] <= max_length].copy()
    logger.info(f"Removed {initial_count - len(df):,} overly long questions")

    # Remove too short questions (less than 10 characters)
    initial_count = len(df)
    df = df[df["question_length"] >= 10].copy()
    logger.info(f"Removed {initial_count - len(df):,} too short questions")

    # Language detection
    logger.info("Detecting languages...")
    texts = df["user_query"].tolist()

    # Process in batches for memory efficiency
    batch_size = 1000
    all_results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_results = detect_language_batch(batch, min_confidence)
        all_results.extend(batch_results)

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {i + len(batch):,} questions")

    # Add language detection results
    is_english_list, confidence_list = zip(*all_results)
    df["is_english"] = is_english_list
    df["language_confidence"] = confidence_list

    # Filter to English questions
    initial_count = len(df)
    english_df = df[df["is_english"]].copy()
    logger.info(
        f"Filtered to {len(english_df):,} English questions ({initial_count - len(english_df):,} non-English removed)"
    )

    # Remove the temporary is_english column but keep confidence
    english_df = english_df.drop(columns=["is_english"])

    return english_df


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

    # Language confidence statistics
    conf_stats = df["language_confidence"].describe()
    logger.info(f"Language confidence statistics:\n{conf_stats}")

    # Length statistics
    length_stats = df["question_length"].describe()
    logger.info(f"Question length statistics:\n{length_stats}")

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
        input_path = snakemake.input[0]
        output_path = snakemake.output[0]

        # Get config parameters
        config = snakemake.config
        min_confidence = config.get("question_filtering", {}).get("min_confidence", 0.8)
        max_length = config.get("question_filtering", {}).get("max_length", 10000)

    except NameError:
        # Fallback for running outside Snakemake (for testing)
        logger.info("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        input_path = base_dir / "data/intermediate/cleaned_arena_data/questions.parquet"
        output_path = (
            base_dir / "data/intermediate/question_analysis/english_questions.parquet"
        )
        min_confidence = 0.8
        max_length = 10000

    # Load questions
    logger.info(f"Loading questions from {input_path}")
    questions_df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(questions_df):,} questions")

    # Filter English questions
    english_questions = filter_english_questions(
        questions_df, min_confidence=min_confidence, max_length=max_length
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
    logger.info(f"English questions: {len(english_questions):,}")
    logger.info(
        f"Filtering rate: {len(english_questions) / len(questions_df) * 100:.1f}%"
    )
    logger.info(
        f"Average confidence: {english_questions['language_confidence'].mean():.3f}"
    )
    logger.info(
        f"Average length: {english_questions['question_length'].mean():.1f} characters"
    )

    logger.info("✅ English question filtering completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
