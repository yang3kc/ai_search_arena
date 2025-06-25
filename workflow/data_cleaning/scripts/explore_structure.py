#!/usr/bin/env python3
"""
Exploration script to understand the structure of search arena data.
This script analyzes turn distribution and validates our understanding
of the nested data structure before extraction.
"""

import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def explore_arena_data():
    """Explore the search arena data structure and turn distribution."""

    # Get paths from Snakemake
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]

    logger.info(f"Loading data from {input_file}")

    # Load the data
    try:
        df = pd.read_parquet(input_file)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Capture output for writing to file
    report_lines = []
    report_lines.append(f"=== SEARCH ARENA DATA EXPLORATION REPORT ===\n")
    report_lines.append(f"Generated: {pd.Timestamp.now()}\n")
    report_lines.append(f"Input file: {input_file}\n")

    # Basic statistics
    logger.info("\n=== BASIC DATA OVERVIEW ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Turn distribution analysis
    logger.info("\n=== TURN DISTRIBUTION ANALYSIS ===")
    turn_counts = df["turn"].value_counts().sort_index()
    logger.info(f"Turn distribution:\n{turn_counts}")

    total_turns = (df["turn"] * len(df)).sum()
    logger.info(f"Total conversation turns across all threads: {total_turns}")

    # Analyze multi-turn conversations
    multi_turn_threads = df[df["turn"] > 1]
    logger.info(
        f"Threads with multiple turns: {len(multi_turn_threads)} ({len(multi_turn_threads) / len(df) * 100:.1f}%)"
    )

    if len(multi_turn_threads) > 0:
        logger.info(f"Max turns in a thread: {df['turn'].max()}")
        logger.info(
            f"Average turns per multi-turn thread: {multi_turn_threads['turn'].mean():.2f}"
        )

    # Winner field analysis
    logger.info("\n=== WINNER FIELD ANALYSIS ===")
    winner_counts = df["winner"].value_counts(dropna=False)
    logger.info(f"Winner distribution:\n{winner_counts}")
    logger.info(
        f"Winner completeness: {df['winner'].notna().sum()}/{len(df)} ({df['winner'].notna().mean() * 100:.1f}%)"
    )

    # Judge field analysis
    logger.info("\n=== JUDGE FIELD ANALYSIS ===")
    judge_counts = df["judge"].value_counts()
    logger.info(f"Judge distribution:\n{judge_counts}")

    # Intent analysis
    logger.info("\n=== INTENT ANALYSIS ===")
    if "primary_intent" in df.columns:
        primary_intent_counts = df["primary_intent"].value_counts()
        logger.info(
            f"Primary intent distribution (top 10):\n{primary_intent_counts.head(10)}"
        )

    # Model analysis
    logger.info("\n=== MODEL ANALYSIS ===")
    model_a_counts = df["model_a"].value_counts()
    model_b_counts = df["model_b"].value_counts()
    logger.info(f"Model A distribution (top 10):\n{model_a_counts.head(10)}")
    logger.info(f"Model B distribution (top 10):\n{model_b_counts.head(10)}")

    # Message structure analysis
    logger.info("\n=== MESSAGE STRUCTURE ANALYSIS ===")
    sample_messages_a = df["messages_a"].iloc[0]

    logger.info(f"Sample messages_a type: {type(sample_messages_a)}")
    logger.info(
        f"Sample messages_a length: {len(sample_messages_a) if hasattr(sample_messages_a, '__len__') else 'N/A'}"
    )

    if hasattr(sample_messages_a, "__len__") and len(sample_messages_a) > 0:
        logger.info(
            f"First message_a structure: {sample_messages_a[0] if len(sample_messages_a) > 0 else 'Empty'}"
        )

    # Check for conversation IDs in metadata
    logger.info("\n=== CONVERSATION ID ANALYSIS ===")
    sample_metadata_a = df["system_a_metadata"].iloc[0]
    if isinstance(sample_metadata_a, dict) and "conv_id" in sample_metadata_a:
        conv_ids_a = df["system_a_metadata"].apply(
            lambda x: x.get("conv_id") if isinstance(x, dict) else None
        )
        unique_conv_ids = conv_ids_a.nunique()
        logger.info(f"Unique conversation IDs in system_a_metadata: {unique_conv_ids}")

        # Check if conv_id is unique per row
        if unique_conv_ids == len(df):
            logger.info("✓ Each row has a unique conversation ID")
        else:
            logger.info(
                "⚠ Some rows share conversation IDs - this might indicate multi-turn threads"
            )

    # Languages analysis
    logger.info("\n=== LANGUAGES ANALYSIS ===")
    if "languages" in df.columns:
        # Sample a few language entries to understand structure
        sample_languages = df["languages"].dropna().iloc[:5]
        logger.info(f"Sample language entries: {list(sample_languages)}")

    # Add comprehensive findings to report
    report_lines.append(f"\n=== BASIC DATA OVERVIEW ===\n")
    report_lines.append(f"Shape: {df.shape}\n")
    report_lines.append(f"Columns: {list(df.columns)}\n\n")

    # Column analysis
    report_lines.append("=== COLUMN ANALYSIS ===\n")
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100

        report_lines.append(f"{col}:\n")
        report_lines.append(f"  - Type: {dtype}\n")
        report_lines.append(f"  - Null values: {null_count} ({null_pct:.1f}%)\n")

        # Handle unique count calculation carefully for array/dict columns
        try:
            if col in [
                "messages_a",
                "messages_b",
                "system_a_metadata",
                "system_b_metadata",
                "languages",
            ]:
                # For complex nested columns, just show sample
                sample_values = df[col].dropna().iloc[:2].tolist()
                report_lines.append(f"  - Type: Complex nested structure\n")
                report_lines.append(f"  - Sample values: {sample_values}\n")
            else:
                unique_count = df[col].nunique()
                report_lines.append(f"  - Unique values: {unique_count}\n")

                # Add sample values for simple columns
                if dtype != "object" or unique_count <= 20:
                    sample_values = df[col].value_counts().head(5)
                    report_lines.append(f"  - Top values: {dict(sample_values)}\n")
                else:
                    # For object types with many unique values, show samples
                    sample_values = df[col].dropna().iloc[:3].tolist()
                    report_lines.append(f"  - Sample values: {sample_values}\n")
        except Exception as e:
            report_lines.append(f"  - Analysis error: {str(e)}\n")
            sample_values = df[col].dropna().iloc[:2].tolist()
            report_lines.append(f"  - Sample values: {sample_values}\n")

        report_lines.append("\n")

    report_lines.append("=== TURN DISTRIBUTION ANALYSIS ===\n")
    report_lines.append(f"Turn distribution:\n{turn_counts}\n\n")
    report_lines.append(
        f"Multi-turn threads: {len(multi_turn_threads)} ({len(multi_turn_threads) / len(df) * 100:.1f}%)\n"
    )
    if len(multi_turn_threads) > 0:
        report_lines.append(f"Max turns in thread: {df['turn'].max()}\n")
        report_lines.append(
            f"Average turns per multi-turn thread: {multi_turn_threads['turn'].mean():.2f}\n\n"
        )

    report_lines.append("=== WINNER AND EVALUATION ANALYSIS ===\n")
    report_lines.append(f"Winner distribution:\n{winner_counts}\n\n")
    report_lines.append(
        f"Winner completeness: {df['winner'].notna().sum()}/{len(df)} ({df['winner'].notna().mean() * 100:.1f}%)\n"
    )

    # Judge analysis
    report_lines.append(f"Total unique judges: {df['judge'].nunique()}\n")
    report_lines.append(f"Top 5 judges: {dict(judge_counts.head(5))}\n\n")

    # Model analysis
    report_lines.append("=== MODEL ANALYSIS ===\n")
    report_lines.append(f"Unique Model A count: {df['model_a'].nunique()}\n")
    report_lines.append(f"Unique Model B count: {df['model_b'].nunique()}\n")
    report_lines.append(f"Top 5 Model A: {dict(model_a_counts.head(5))}\n")
    report_lines.append(f"Top 5 Model B: {dict(model_b_counts.head(5))}\n\n")

    # Intent analysis
    report_lines.append("=== INTENT ANALYSIS ===\n")
    if "primary_intent" in df.columns:
        report_lines.append(
            f"Primary intent distribution:\n{dict(primary_intent_counts)}\n"
        )
    if "secondary_intent" in df.columns:
        secondary_intent_counts = df["secondary_intent"].value_counts()
        report_lines.append(
            f"Secondary intent distribution:\n{dict(secondary_intent_counts)}\n\n"
        )

    # Message structure analysis
    report_lines.append("=== MESSAGE STRUCTURE ANALYSIS ===\n")
    report_lines.append(f"Messages_a sample type: {type(sample_messages_a)}\n")
    report_lines.append(
        f"Messages_a sample length: {len(sample_messages_a) if hasattr(sample_messages_a, '__len__') else 'N/A'}\n"
    )
    if hasattr(sample_messages_a, "__len__") and len(sample_messages_a) > 0:
        report_lines.append(f"First message_a structure: {sample_messages_a[0]}\n")

    # System metadata analysis
    report_lines.append("\n=== SYSTEM METADATA ANALYSIS ===\n")
    if isinstance(sample_metadata_a, dict):
        metadata_keys = list(sample_metadata_a.keys())
        report_lines.append(f"System_a_metadata keys: {metadata_keys}\n")

        # Check for conversation IDs
        conv_ids_a = df["system_a_metadata"].apply(
            lambda x: x.get("conv_id") if isinstance(x, dict) else None
        )
        unique_conv_ids = conv_ids_a.nunique()
        report_lines.append(f"Unique conversation IDs: {unique_conv_ids}\n")

        if unique_conv_ids == len(df):
            report_lines.append("✓ Each row has unique conversation ID\n")
        else:
            report_lines.append("⚠ Some rows share conversation IDs\n")

    # Languages analysis
    report_lines.append("\n=== LANGUAGES ANALYSIS ===\n")
    if "languages" in df.columns:
        sample_languages = df["languages"].dropna().iloc[:5]
        report_lines.append(f"Sample language entries: {list(sample_languages)}\n")

        # Flatten language arrays to get language distribution
        all_languages = []
        for lang_array in df["languages"].dropna():
            if hasattr(lang_array, "__iter__"):
                all_languages.extend(lang_array)

        if all_languages:
            lang_counts = pd.Series(all_languages).value_counts()
            report_lines.append(
                f"Language distribution: {dict(lang_counts.head(10))}\n"
            )

    # Write report to output file
    with open(output_file, "w") as f:
        f.writelines(report_lines)

    logger.info(f"\n=== EXPLORATION COMPLETE ===")
    logger.info(f"Report written to {output_file}")


if __name__ == "__main__":
    explore_arena_data()
