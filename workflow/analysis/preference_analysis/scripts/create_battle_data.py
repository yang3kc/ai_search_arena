#!/usr/bin/env python3
"""
Battle Data Creation Script for News Citation Preference Analysis.

This script creates battle-style data from response signals, transforming individual
responses into model_a vs model_b comparisons for statistical preference analysis.
"""

import pandas as pd
from pathlib import Path
import sys


def load_data(threads_path, signals_path):
    """Load threads and response signals data."""
    print("Loading threads and response signals data...")

    threads_df = pd.read_parquet(threads_path)
    signals_df = pd.read_parquet(signals_path)

    print(f"Loaded {len(threads_df):,} threads")
    print(f"Loaded {len(signals_df):,} response signals")

    return threads_df, signals_df


def create_battle_data(threads_df, signals_df):
    """
    Create battle-style data compatible with citation_style_analysis.py.

    Each row represents a thread-level comparison between model_a and model_b,
    with signal features aggregated across all responses in the thread and suffixed by _a and _b.
    """
    print("Creating battle data...")

    battle_data = []

    # Get signal feature columns (exclude metadata columns)
    metadata_cols = {
        "response_id",
        "question_id",
        "thread_id",
        "turn_number",
        "model_name_llm",
        "model_name_raw",
        "model_side",
        "response_text",
        "response_role",
        "citation_format",
        "llm_temperature",
        "llm_top_p",
        "llm_max_tokens",
        "search_context_size",
        "user_location_country",
        "search_engine",
        "scrape_engine",
        "context_manager",
        "has_news",
        "winner",
        "primary_intent",
        "secondary_intent",
        "total_turns",
        "model_won",
        "model_lost",
    }

    signal_features = [col for col in signals_df.columns if col not in metadata_cols]
    print(f"Signal features to aggregate: {len(signal_features)} features")
    print(f"Features: {signal_features[:10]}...")  # Show first 10

    for _, thread in threads_df.iterrows():
        thread_id = thread["thread_id"]

        # Get all responses for this thread
        thread_responses = signals_df[signals_df["thread_id"] == thread_id]

        if len(thread_responses) < 2:
            print(f"Skipping thread {thread_id} with less than 2 responses")
            continue  # Skip threads with less than 2 responses

        # Group responses by model_side to properly assign model_a and model_b
        side_groups = thread_responses.groupby("model_side")

        # Check if we have both model_a and model_b sides
        if "a" not in side_groups.groups or "b" not in side_groups.groups:
            print(f"Skipping thread {thread_id} with no model_a or model_b")
            continue  # Skip if we don't have both sides

        # Get responses for each side
        model_a_responses = side_groups.get_group("a")
        model_b_responses = side_groups.get_group("b")

        # Get model names from the responses
        model_a_name = model_a_responses["model_name_raw"].iloc[0]
        model_b_name = model_b_responses["model_name_raw"].iloc[0]

        # Aggregate signals for each model (take mean across all responses in thread)
        model_a_signals = {}
        model_b_signals = {}

        for feature in signal_features:
            if feature in model_a_responses.columns:
                # Use mean for numeric features, first value for categorical
                if pd.api.types.is_numeric_dtype(model_a_responses[feature]):
                    model_a_signals[feature] = model_a_responses[feature].mean()
                    model_b_signals[feature] = model_b_responses[feature].mean()
                else:
                    # For non-numeric features, take the most common value or first value
                    model_a_signals[feature] = (
                        model_a_responses[feature].iloc[0]
                        if len(model_a_responses) > 0
                        else 0
                    )
                    model_b_signals[feature] = (
                        model_b_responses[feature].iloc[0]
                        if len(model_b_responses) > 0
                        else 0
                    )

        # Create battle row
        battle_row = {
            "thread_id": thread_id,
            "model_a": model_a_name,
            "model_b": model_b_name,
            "winner": thread["winner"],  # Thread-level winner
            "num_responses_a": len(
                model_a_responses
            ),  # Track how many responses were aggregated
            "num_responses_b": len(model_b_responses),
        }

        # Add thread-level metadata
        for col in ["primary_intent", "secondary_intent", "total_turns"]:
            if col in thread:
                battle_row[col] = thread[col]

        # Add aggregated signal features for both models with _a and _b suffixes
        for feature in signal_features:
            battle_row[f"{feature}_a"] = model_a_signals.get(feature, 0)
            battle_row[f"{feature}_b"] = model_b_signals.get(feature, 0)

        # Add model pair information for analysis
        battle_row["model_pair"] = f"{model_a_name}_vs_{model_b_name}"

        battle_data.append(battle_row)

    battle_df = pd.DataFrame(battle_data)
    print(f"Created {len(battle_df):,} thread-level battle comparisons")

    # Show aggregation statistics
    if len(battle_df) > 0:
        print(
            f"  Average responses per model A: {battle_df['num_responses_a'].mean():.1f}"
        )
        print(
            f"  Average responses per model B: {battle_df['num_responses_b'].mean():.1f}"
        )
        print(
            f"  Total responses aggregated: {battle_df['num_responses_a'].sum() + battle_df['num_responses_b'].sum():,}"
        )

    return battle_df


def compute_signal_differences(battle_df):
    """Compute differences between model_a and model_b for numeric signals only."""
    print("Computing signal differences...")

    # Find all _a columns
    a_columns = [col for col in battle_df.columns if col.endswith("_a")]

    difference_cols = []

    for a_col in a_columns:
        # Get corresponding _b column
        feature_name = a_col[:-2]  # Remove "_a" suffix
        b_col = f"{feature_name}_b"

        if b_col in battle_df.columns:
            # Only compute differences for numeric columns
            if pd.api.types.is_numeric_dtype(
                battle_df[a_col]
            ) and pd.api.types.is_numeric_dtype(battle_df[b_col]):
                # Compute difference (model_a - model_b)
                diff_col = f"{feature_name}_diff"
                battle_df[diff_col] = battle_df[a_col] - battle_df[b_col]
                difference_cols.append(diff_col)
            else:
                print(f"  Skipping non-numeric feature: {feature_name}")

    print(f"Computed {len(difference_cols)} signal differences")
    return battle_df, difference_cols


def validate_battle_data(battle_df):
    """Validate the battle data format and content."""
    print("Validating battle data...")

    # Check required columns
    required_cols = ["thread_id", "model_a", "model_b", "winner"]
    missing_cols = [col for col in required_cols if col not in battle_df.columns]
    if missing_cols:
        print(f"  ERROR: Missing required columns: {missing_cols}")
        return False

    # Check winner distribution
    winner_counts = battle_df["winner"].value_counts()
    print(f"  Winner distribution: {dict(winner_counts)}")

    # Check for valid winner values
    valid_winners = ["model_a", "model_b", "tie", "tie (bothbad)"]
    invalid_winners = battle_df[~battle_df["winner"].isin(valid_winners)][
        "winner"
    ].unique()
    if len(invalid_winners) > 0:
        print(f"  WARNING: Invalid winner values: {invalid_winners}")

    # Check model pairs
    unique_models = set(battle_df["model_a"]) | set(battle_df["model_b"])
    print(f"  Unique models: {len(unique_models)}")
    print(f"  Models: {list(unique_models)}")

    # Check for same model competing against itself
    self_battles = battle_df[battle_df["model_a"] == battle_df["model_b"]]
    if len(self_battles) > 0:
        print(
            f"  WARNING: {len(self_battles)} battles where model competes against itself"
        )

    # Check signal features
    signal_a_cols = [col for col in battle_df.columns if col.endswith("_a")]
    signal_b_cols = [col for col in battle_df.columns if col.endswith("_b")]
    print(
        f"  Signal features: {len(signal_a_cols)} _a features, {len(signal_b_cols)} _b features"
    )

    # Check for missing values in key signal columns
    key_signals = [
        "response_length_a",
        "num_citations_a",
        "proportion_low_quality_a",
        "proportion_right_leaning_a",
    ]
    for col in key_signals:
        if col in battle_df.columns:
            missing = battle_df[col].isna().sum()
            if missing > 0:
                print(f"  WARNING: {missing:,} missing values in {col}")

    print("  Validation completed")
    return True


def add_analysis_metadata(battle_df):
    """Add metadata useful for analysis."""
    print("Adding analysis metadata...")

    # Model family information (extract from model names)
    def get_model_family(model_name):
        if "gpt" in model_name.lower():
            return "openai"
        elif "sonar" in model_name.lower():
            return "perplexity"
        elif "claude" in model_name.lower():
            return "anthropic"
        else:
            return "other"

    battle_df["model_a_family"] = battle_df["model_a"].apply(get_model_family)
    battle_df["model_b_family"] = battle_df["model_b"].apply(get_model_family)

    # Family vs family comparison
    battle_df["family_matchup"] = (
        battle_df["model_a_family"] + "_vs_" + battle_df["model_b_family"]
    )

    # Cross-family vs same-family battles
    battle_df["cross_family_battle"] = (
        battle_df["model_a_family"] != battle_df["model_b_family"]
    )
    return battle_df


def main():
    # Get input/output paths from Snakemake
    try:
        # Snakemake provides these variables
        input_threads = snakemake.input.threads
        input_signals = snakemake.input.signals

        output_battle = snakemake.output[0]

    except NameError:
        # Fallback for running outside Snakemake (for testing)
        print("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        input_dir = base_dir / "data/intermediate/preference_analysis"

        input_threads = input_dir / "news_competitions.parquet"
        input_signals = input_dir / "news_competitions_response_signals.parquet"

        output_battle = input_dir / "battle_data.parquet"

    try:
        # Load data
        threads_df, signals_df = load_data(input_threads, input_signals)

        # Create battle data
        battle_df = create_battle_data(threads_df, signals_df)

        if len(battle_df) == 0:
            print("ERROR: No battle data created")
            return 1

        # Compute signal differences
        battle_df, diff_cols = compute_signal_differences(battle_df)

        # Add analysis metadata
        battle_df = add_analysis_metadata(battle_df)

        # Validate results
        is_valid = validate_battle_data(battle_df)
        if not is_valid:
            print("WARNING: Battle data validation failed")

        # Ensure output directory exists
        Path(output_battle).parent.mkdir(parents=True, exist_ok=True)

        # Save battle data
        battle_df.to_parquet(output_battle, index=False)

        print(f"\nSaved battle data to: {output_battle}")
        print(f"Dataset shape: {battle_df.shape}")

        # Summary statistics
        signal_pairs = len([col for col in battle_df.columns if col.endswith("_a")])
        print(
            f"Battle features: {signal_pairs} signal pairs + {len(diff_cols)} differences + metadata"
        )

        # Show sample of key metrics
        if len(battle_df) > 0:
            print("\nSample battle statistics:")
            key_diff_cols = [
                col
                for col in diff_cols
                if any(
                    key in col
                    for key in [
                        "response_length",
                        "num_citations",
                        "proportion_low_quality",
                        "proportion_right_leaning",
                    ]
                )
            ]
            if key_diff_cols:
                sample_stats = battle_df[key_diff_cols[:4]].describe()
                print(sample_stats.round(3))

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
