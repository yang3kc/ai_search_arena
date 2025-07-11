#!/usr/bin/env python3
"""
Questions extraction script for search arena data.
Extracts user questions from multi-turn conversations and creates the questions table
according to the defined schema.
"""

import logging

import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_questions():
    """Extract questions from search arena data multi-turn conversations."""

    # Get paths from Snakemake
    input_data = snakemake.input.data
    threads_file = snakemake.input.threads
    output_file = snakemake.output[0]
    config = snakemake.config

    logger.info(f"Loading data from {input_data}")
    logger.info(f"Loading threads table from {threads_file}")

    # Load the raw data and threads table
    try:
        df = pd.read_parquet(input_data)
        threads_df = pd.read_parquet(threads_file)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        logger.info(f"Loaded {len(threads_df)} thread records")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    logger.info("Extracting questions from message arrays...")

    # Create questions table according to schema
    questions_data = []
    total_questions = 0
    mismatch_count = 0

    for idx, row in df.iterrows():
        # Get thread_id from threads table
        thread_id = f"{config['thread_id_prefix']}{idx:08d}"
        total_turns = row.get("turn", 1)

        # Extract messages from system metadata instead of direct message arrays
        system_a_metadata = row.get("system_a_metadata", {})
        system_b_metadata = row.get("system_b_metadata", {})

        # Get formatted_messages from system_a_metadata (primary source)
        if (
            not isinstance(system_a_metadata, dict)
            or "formatted_messages" not in system_a_metadata
        ):
            logger.warning(
                f"No formatted_messages in system_a_metadata for row {idx}, skipping"
            )
            continue

        messages_a = system_a_metadata.get("formatted_messages", [])
        messages_b = (
            system_b_metadata.get("formatted_messages", [])
            if isinstance(system_b_metadata, dict)
            else []
        )

        if not hasattr(messages_a, "__len__") or len(messages_a) == 0:
            logger.warning(
                f"Empty formatted_messages in system_a_metadata for row {idx}, skipping"
            )
            continue

        # Process messages to extract user questions
        # In the messages array, user and assistant messages alternate
        # For N turns, we expect N user messages and N assistant messages (total 2*N)
        user_messages_a = []
        for message in messages_a:
            if isinstance(message, dict) and message.get("role") == "user":
                user_messages_a.append(message)

        # Consistency check: verify that user messages are the same in both system_a and system_b metadata
        if len(messages_b) > 0:
            user_messages_b = []
            for message in messages_b:
                if isinstance(message, dict) and message.get("role") == "user":
                    user_messages_b.append(message)

            # Check if user messages match between system_a and system_b
            if len(user_messages_a) != len(user_messages_b):
                logger.warning(
                    f"Row {idx}: User message count mismatch between system_a ({len(user_messages_a)}) "
                    f"and system_b ({len(user_messages_b)})"
                )
            else:
                # Check if user message content matches
                for turn_idx, (msg_a, msg_b) in enumerate(
                    zip(user_messages_a, user_messages_b)
                ):
                    content_a = msg_a.get("content", "")
                    content_b = msg_b.get("content", "")
                    if content_a != content_b:
                        logger.warning(
                            f"Row {idx}, Turn {turn_idx + 1}: User message content mismatch between system_a and system_b"
                        )
                        if (idx + 1) % 1000 == 0:  # Log examples occasionally
                            logger.info(f"  system_a: '{content_a[:100]}...'")
                            logger.info(f"  system_b: '{content_b[:100]}...'")

        # Create question records for each user message (using system_a as primary source)
        for turn_idx, user_message in enumerate(user_messages_a):
            # Generate question_id
            question_id = f"{config['question_id_prefix']}{total_questions:08d}"

            question_record = {
                "question_id": question_id,
                "thread_id": thread_id,
                "turn_number": turn_idx + 1,  # Turn numbers start at 1
                "user_query": user_message.get("content", ""),
                "question_role": user_message.get("role", "user"),
            }

            questions_data.append(question_record)
            total_questions += 1

        # Validate that we found the expected number of questions based on total_turns
        # Note: The turn field might not always match the actual number of user messages
        # This could be due to incomplete conversations, system messages, or other factors
        expected_questions = total_turns
        actual_questions = len(user_messages_a)

        if actual_questions != expected_questions:
            mismatch_count += 1
            if (idx + 1) % 1000 == 0:
                # Only log a few examples to avoid spam
                logger.info(
                    f"Example mismatch - Row {idx}: Expected {expected_questions} questions, "
                    f"found {actual_questions} user messages (formatted_messages length: {len(messages_a)})"
                )

        # Log progress every 1000 rows
        if (idx + 1) % 1000 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} threads")

    # Create DataFrame
    questions_df = pd.DataFrame(questions_data)

    # Data validation
    logger.info("Validating questions data...")

    # Check for required fields
    required_fields = ["question_id", "thread_id", "turn_number", "user_query"]
    missing_required = []
    for field in required_fields:
        if questions_df[field].isna().any():
            missing_count = questions_df[field].isna().sum()
            missing_required.append(f"{field}: {missing_count} missing")

    if missing_required:
        logger.warning(f"Missing required fields: {', '.join(missing_required)}")

    # Validate question_id uniqueness
    if questions_df["question_id"].nunique() != len(questions_df):
        logger.error("Question IDs are not unique!")
        return

    # Validate thread_id references
    valid_thread_ids = set(threads_df["thread_id"])
    invalid_thread_ids = set(questions_df["thread_id"]) - valid_thread_ids
    if invalid_thread_ids:
        logger.error(f"Invalid thread_id references found: {len(invalid_thread_ids)}")
        return

    # Log statistics
    logger.info("\n=== QUESTION EXTRACTION STATISTICS ===")
    logger.info(f"Total questions extracted: {len(questions_df)}")
    logger.info(f"Total threads represented: {questions_df['thread_id'].nunique()}")
    logger.info(
        f"Average questions per thread: {len(questions_df) / questions_df['thread_id'].nunique():.2f}"
    )
    logger.info(
        f"Turn count mismatches: {mismatch_count}/{len(df)} ({mismatch_count / len(df) * 100:.1f}%)"
    )

    # Turn distribution
    turn_dist = questions_df["turn_number"].value_counts().sort_index()
    logger.info("Turn distribution:")
    for turn, count in turn_dist.head(10).items():
        logger.info(
            f"  Turn {turn}: {count} questions ({count / len(questions_df) * 100:.1f}%)"
        )

    # Question length statistics
    questions_df["query_length"] = questions_df["user_query"].str.len()
    logger.info("Question length statistics:")
    logger.info(f"  Mean length: {questions_df['query_length'].mean():.1f} characters")
    logger.info(
        f"  Median length: {questions_df['query_length'].median():.1f} characters"
    )
    logger.info(f"  Min length: {questions_df['query_length'].min()}")
    logger.info(f"  Max length: {questions_df['query_length'].max()}")

    # Language distribution (based on sample)
    sample_questions = questions_df["user_query"].head(100).tolist()
    logger.info("Sample questions:")
    for i, q in enumerate(sample_questions[:5]):
        logger.info(f"  {i + 1}. {q[:100]}{'...' if len(q) > 100 else ''}")

    # Save to parquet
    logger.info(f"Saving questions table to {output_file}")
    try:
        # Drop temporary column before saving
        questions_df = questions_df.drop("query_length", axis=1)
        questions_df.to_parquet(output_file, index=False)
        logger.info(f"Successfully saved {len(questions_df)} question records")
    except Exception as e:
        logger.error(f"Failed to save questions table: {e}")
        return

    # Validation check - reload and verify
    try:
        validation_df = pd.read_parquet(output_file)
        if len(validation_df) == len(questions_df):
            logger.info("✓ Questions table validation successful")
        else:
            logger.error(
                f"✗ Questions table validation failed: expected {len(questions_df)}, got {len(validation_df)}"
            )
    except Exception as e:
        logger.error(f"✗ Questions table validation failed: {e}")


if __name__ == "__main__":
    extract_questions()
