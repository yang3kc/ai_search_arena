#!/usr/bin/env python3
"""
Responses extraction script for search arena data.
Extracts AI model responses from multi-turn conversations and creates the responses table
according to the defined schema.
"""

import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_responses():
    """Extract responses from search arena data multi-turn conversations."""

    # Get paths from Snakemake
    input_data = snakemake.input.data
    questions_file = snakemake.input.questions
    output_file = snakemake.output[0]
    config = snakemake.config

    logger.info(f"Loading data from {input_data}")
    logger.info(f"Loading questions table from {questions_file}")

    # Load the raw data and questions table
    try:
        df = pd.read_parquet(input_data)
        questions_df = pd.read_parquet(questions_file)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        logger.info(f"Loaded {len(questions_df)} question records")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    logger.info("Extracting responses from system metadata...")

    # Create responses table according to schema
    responses_data = []
    total_responses = 0
    missing_metadata_count = 0
    mismatch_count = 0

    for idx, row in df.iterrows():
        # Get thread_id and corresponding questions for this thread
        thread_id = f"{config['thread_id_prefix']}{idx:08d}"
        thread_questions = questions_df[questions_df["thread_id"] == thread_id]

        if len(thread_questions) == 0:
            logger.warning(f"No questions found for thread {thread_id}, skipping")
            continue

        # Extract system metadata for both models
        system_a_metadata = row.get("system_a_metadata", {})
        system_b_metadata = row.get("system_b_metadata", {})

        if not isinstance(system_a_metadata, dict) or not isinstance(
            system_b_metadata, dict
        ):
            missing_metadata_count += 1
            logger.warning(f"Missing system metadata for row {idx}, skipping")
            continue

        # Get formatted_messages for both models
        messages_a = system_a_metadata.get("formatted_messages", [])
        messages_b = system_b_metadata.get("formatted_messages", [])

        if len(messages_a) == 0 or len(messages_b) == 0:
            logger.warning(f"Empty formatted_messages for row {idx}, skipping")
            continue

        # Extract assistant responses from both models
        assistant_responses_a = []
        assistant_responses_b = []

        for message in messages_a:
            if isinstance(message, dict) and message.get("role") == "assistant":
                assistant_responses_a.append(message)

        for message in messages_b:
            if isinstance(message, dict) and message.get("role") == "assistant":
                assistant_responses_b.append(message)

        # Validate response counts match questions
        expected_responses = len(thread_questions)
        if (
            len(assistant_responses_a) != expected_responses
            or len(assistant_responses_b) != expected_responses
        ):
            mismatch_count += 1
            if (idx + 1) % 1000 == 0:
                logger.info(
                    f"Response count mismatch - Row {idx}: Expected {expected_responses}, "
                    f"found {len(assistant_responses_a)} (A) and {len(assistant_responses_b)} (B)"
                )

        # Extract model names from both system metadata and raw data
        model_a_llm_name = system_a_metadata.get("llm_config", {}).get(
            "name", "unknown"
        )
        model_b_llm_name = system_b_metadata.get("llm_config", {}).get(
            "name", "unknown"
        )
        model_a_raw = row.get("model_a", "unknown")
        model_b_raw = row.get("model_b", "unknown")

        # Extract LLM configuration parameters
        llm_config_a = system_a_metadata.get("llm_config", {}).get("params", {})
        llm_config_b = system_b_metadata.get("llm_config", {}).get("params", {})

        web_search_config_a = system_a_metadata.get("web_search_config", {})
        web_search_config_b = system_b_metadata.get("web_search_config", {})

        # Create response records for each question-response pair
        for question_idx, question_row in thread_questions.iterrows():
            question_id = question_row["question_id"]
            turn_number = question_row["turn_number"]

            # Ensure we have responses for this turn (turn_number is 1-indexed)
            response_idx = turn_number - 1

            if response_idx < len(assistant_responses_a) and response_idx < len(
                assistant_responses_b
            ):
                # Model A response
                response_a_id = f"{config['response_id_prefix']}{total_responses:08d}"
                response_a = {
                    "response_id": response_a_id,
                    "question_id": question_id,
                    "thread_id": thread_id,
                    "turn_number": turn_number,
                    "model_name_llm": model_a_llm_name,
                    "model_name_raw": model_a_raw,
                    "model_side": "a",
                    "response_text": assistant_responses_a[response_idx].get(
                        "content", ""
                    ),
                    "response_role": assistant_responses_a[response_idx].get(
                        "role", "assistant"
                    ),
                    "citation_format": system_a_metadata.get(
                        "citation_format_standardized", ""
                    ),
                    "llm_temperature": llm_config_a.get("temperature"),
                    "llm_top_p": llm_config_a.get("top_p"),
                    "llm_max_tokens": llm_config_a.get("max_tokens")
                    or llm_config_a.get("max_completion_tokens"),
                    "search_context_size": llm_config_a.get(
                        "web_search_options", {}
                    ).get("search_context_size")
                    if llm_config_a.get("web_search_options")
                    else None,
                    "user_location_country": llm_config_a.get(
                        "web_search_options", {}
                    ).get("user_location")
                    if llm_config_a.get("web_search_options")
                    else None,
                    "search_engine": web_search_config_a.get("search_engine"),
                    "scrape_engine": web_search_config_a.get("scrape_engine"),
                    "context_manager": web_search_config_a.get("context_manager"),
                }
                responses_data.append(response_a)
                total_responses += 1

                # Model B response
                response_b_id = f"{config['response_id_prefix']}{total_responses:08d}"
                response_b = {
                    "response_id": response_b_id,
                    "question_id": question_id,
                    "thread_id": thread_id,
                    "turn_number": turn_number,
                    "model_name_llm": model_b_llm_name,
                    "model_name_raw": model_b_raw,
                    "model_side": "b",
                    "response_text": assistant_responses_b[response_idx].get(
                        "content", ""
                    ),
                    "response_role": assistant_responses_b[response_idx].get(
                        "role", "assistant"
                    ),
                    "citation_format": system_b_metadata.get(
                        "citation_format_standardized", ""
                    ),
                    "llm_temperature": llm_config_b.get("temperature"),
                    "llm_top_p": llm_config_b.get("top_p"),
                    "llm_max_tokens": llm_config_b.get("max_tokens")
                    or llm_config_b.get("max_completion_tokens"),
                    "search_context_size": llm_config_b.get(
                        "web_search_options", {}
                    ).get("search_context_size")
                    if llm_config_b.get("web_search_options")
                    else None,
                    "user_location_country": llm_config_b.get(
                        "web_search_options", {}
                    ).get("user_location")
                    if llm_config_b.get("web_search_options")
                    else None,
                    "search_engine": web_search_config_b.get("search_engine"),
                    "scrape_engine": web_search_config_b.get("scrape_engine"),
                    "context_manager": web_search_config_b.get("context_manager"),
                }
                responses_data.append(response_b)
                total_responses += 1
            else:
                logger.warning(
                    f"Missing response for turn {turn_number} in thread {thread_id}"
                )

        # Log progress every 1000 rows
        if (idx + 1) % 1000 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} threads")

    # Create DataFrame
    responses_df = pd.DataFrame(responses_data)

    # Data validation
    logger.info("Validating responses data...")

    # Check for required fields
    required_fields = [
        "response_id",
        "question_id",
        "thread_id",
        "turn_number",
        "model_name_llm",
        "model_name_raw",
        "model_side",
        "response_text",
    ]
    missing_required = []
    for field in required_fields:
        if responses_df[field].isna().any():
            missing_count = responses_df[field].isna().sum()
            missing_required.append(f"{field}: {missing_count} missing")

    if missing_required:
        logger.warning(f"Missing required fields: {', '.join(missing_required)}")

    # Validate response_id uniqueness
    if responses_df["response_id"].nunique() != len(responses_df):
        logger.error("Response IDs are not unique!")
        return

    # Validate question_id references
    valid_question_ids = set(questions_df["question_id"])
    invalid_question_ids = set(responses_df["question_id"]) - valid_question_ids
    if invalid_question_ids:
        logger.error(
            f"Invalid question_id references found: {len(invalid_question_ids)}"
        )
        return

    # Validate model_side distribution
    model_side_counts = responses_df["model_side"].value_counts()
    if (
        len(model_side_counts) != 2
        or "a" not in model_side_counts
        or "b" not in model_side_counts
    ):
        logger.warning("Unexpected model_side distribution")
    else:
        if (
            abs(model_side_counts["a"] - model_side_counts["b"]) > 10
        ):  # Allow small differences
            logger.warning(
                f"Unbalanced model_side distribution: A={model_side_counts['a']}, B={model_side_counts['b']}"
            )

    # Log statistics
    logger.info("\n=== RESPONSE EXTRACTION STATISTICS ===")
    logger.info(f"Total responses extracted: {len(responses_df)}")
    logger.info(f"Total questions represented: {responses_df['question_id'].nunique()}")
    logger.info(f"Total threads represented: {responses_df['thread_id'].nunique()}")
    logger.info(
        f"Responses per question: {len(responses_df) / responses_df['question_id'].nunique():.2f}"
    )
    logger.info(
        f"Missing metadata count: {missing_metadata_count}/{len(df)} ({missing_metadata_count / len(df) * 100:.1f}%)"
    )
    logger.info(
        f"Response count mismatches: {mismatch_count}/{len(df)} ({mismatch_count / len(df) * 100:.1f}%)"
    )

    # Model distribution (LLM config names)
    model_llm_counts = responses_df["model_name_llm"].value_counts()
    logger.info("Model distribution (LLM config) (top 10):")
    for model, count in model_llm_counts.head(10).items():
        logger.info(
            f"  {model}: {count} responses ({count / len(responses_df) * 100:.1f}%)"
        )

    # Model distribution (raw names)
    model_raw_counts = responses_df["model_name_raw"].value_counts()
    logger.info("Model distribution (raw data) (top 10):")
    for model, count in model_raw_counts.head(10).items():
        logger.info(
            f"  {model}: {count} responses ({count / len(responses_df) * 100:.1f}%)"
        )

    # Model side distribution
    side_dist = responses_df["model_side"].value_counts()
    logger.info("Model side distribution:")
    for side, count in side_dist.items():
        logger.info(
            f"  Side {side}: {count} responses ({count / len(responses_df) * 100:.1f}%)"
        )

    # Response length statistics
    responses_df["response_length"] = responses_df["response_text"].str.len()
    logger.info("Response length statistics:")
    logger.info(
        f"  Mean length: {responses_df['response_length'].mean():.1f} characters"
    )
    logger.info(
        f"  Median length: {responses_df['response_length'].median():.1f} characters"
    )
    logger.info(f"  Min length: {responses_df['response_length'].min()}")
    logger.info(f"  Max length: {responses_df['response_length'].max()}")

    # LLM configuration statistics
    logger.info("LLM configuration statistics:")
    temp_stats = responses_df["llm_temperature"].describe()
    logger.info(
        f"  Temperature: mean={temp_stats['mean']:.2f}, min={temp_stats['min']:.2f}, max={temp_stats['max']:.2f}"
    )

    # Save to parquet
    logger.info(f"Saving responses table to {output_file}")
    try:
        # Drop temporary column before saving
        responses_df = responses_df.drop("response_length", axis=1)
        responses_df.to_parquet(output_file, index=False)
        logger.info(f"Successfully saved {len(responses_df)} response records")
    except Exception as e:
        logger.error(f"Failed to save responses table: {e}")
        return

    # Validation check - reload and verify
    try:
        validation_df = pd.read_parquet(output_file)
        if len(validation_df) == len(responses_df):
            logger.info("✓ Responses table validation successful")
        else:
            logger.error(
                f"✗ Responses table validation failed: expected {len(responses_df)}, got {len(validation_df)}"
            )
    except Exception as e:
        logger.error(f"✗ Responses table validation failed: {e}")


if __name__ == "__main__":
    extract_responses()
