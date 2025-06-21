#!/usr/bin/env python3
"""
Thread extraction script for search arena data.
Extracts thread-level metadata and creates the threads table
according to the defined schema.
"""

import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_threads():
    """Extract thread-level data from search arena data."""

    # Get paths from Snakemake
    input_data = snakemake.input.data
    output_file = snakemake.output[0]
    config = snakemake.config

    logger.info(f"Loading data from {input_data}")

    # Load the raw data
    try:
        df = pd.read_parquet(input_data)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    logger.info("Extracting thread-level data...")

    # Create threads table according to schema
    threads_data = []

    for idx, row in df.iterrows():
        # Generate thread_id
        thread_id = f"{config['thread_id_prefix']}{idx:08d}"

        # Extract basic thread metadata
        thread_record = {
            "thread_id": thread_id,
            "original_row_id": idx,
            "timestamp": row.get("timestamp"),
            "total_turns": row.get("turn", 1),  # Use turn field as total_turns
            "winner": row.get("winner"),
            "judge": row.get("judge"),
            "primary_intent": row.get("primary_intent"),
            "secondary_intent": row.get("secondary_intent"),
            "languages": row.get("languages"),
            "client_country": None,  # Will extract from system metadata if available
        }

        # Extract client_country from system metadata if available
        system_a_metadata = row.get("system_a_metadata")
        if isinstance(system_a_metadata, dict):
            client_country = system_a_metadata.get("client_country")
            if client_country:
                thread_record["client_country"] = client_country

        threads_data.append(thread_record)

        # Log progress every 1000 rows
        if (idx + 1) % 1000 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} threads")

    # Create DataFrame
    threads_df = pd.DataFrame(threads_data)

    # Data validation
    logger.info("Validating thread data...")

    # Check for required fields
    required_fields = ["thread_id", "original_row_id", "total_turns"]
    missing_required = []
    for field in required_fields:
        if threads_df[field].isna().any():
            missing_count = threads_df[field].isna().sum()
            missing_required.append(f"{field}: {missing_count} missing")

    if missing_required:
        logger.warning(f"Missing required fields: {', '.join(missing_required)}")

    # Validate thread_id uniqueness
    if threads_df["thread_id"].nunique() != len(threads_df):
        logger.error("Thread IDs are not unique!")
        return

    # Validate original_row_id range
    if threads_df["original_row_id"].min() < 0 or threads_df[
        "original_row_id"
    ].max() >= len(df):
        logger.error("Original row IDs are out of range!")
        return

    # Log statistics
    logger.info(f"\n=== THREAD EXTRACTION STATISTICS ===")
    logger.info(f"Total threads extracted: {len(threads_df)}")
    logger.info(
        f"Threads with winners: {threads_df['winner'].notna().sum()} ({threads_df['winner'].notna().mean() * 100:.1f}%)"
    )
    logger.info(f"Turn distribution:")
    turn_dist = threads_df["total_turns"].value_counts().sort_index()
    for turns, count in turn_dist.items():
        logger.info(
            f"  {turns} turn(s): {count} threads ({count / len(threads_df) * 100:.1f}%)"
        )

    logger.info(f"Primary intent distribution (top 10):")
    if threads_df["primary_intent"].notna().any():
        intent_dist = threads_df["primary_intent"].value_counts().head(10)
        for intent, count in intent_dist.items():
            logger.info(f"  {intent}: {count}")

    # Save to parquet
    logger.info(f"Saving threads table to {output_file}")
    try:
        threads_df.to_parquet(output_file, index=False)
        logger.info(f"Successfully saved {len(threads_df)} thread records")
    except Exception as e:
        logger.error(f"Failed to save threads table: {e}")
        return

    # Validation check - reload and verify
    try:
        validation_df = pd.read_parquet(output_file)
        if len(validation_df) == len(threads_df):
            logger.info("✓ Thread table validation successful")
        else:
            logger.error(
                f"✗ Thread table validation failed: expected {len(threads_df)}, got {len(validation_df)}"
            )
    except Exception as e:
        logger.error(f"✗ Thread table validation failed: {e}")


if __name__ == "__main__":
    extract_threads()
