"""
Test the full pipeline: feature engineering + leaderboard computation.
"""

import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from feature_engineering import create_leaderboard_format
from leaderboard_core import run_leaderboard


def test_full_pipeline():
    """Test complete pipeline from cleaned data to leaderboard."""

    print("Loading cleaned data...")
    threads_df = pd.read_parquet(
        "/Users/yangkc/working/llm/ai_search_arena/data/intermediate/cleaned_arena_data/threads.parquet"
    )
    responses_df = pd.read_parquet(
        "/Users/yangkc/working/llm/ai_search_arena/data/intermediate/cleaned_arena_data/responses.parquet"
    )
    citations_df = pd.read_parquet(
        "/Users/yangkc/working/llm/ai_search_arena/data/intermediate/cleaned_arena_data/citations.parquet"
    )

    print(
        f"Loaded {len(threads_df)} threads, {len(responses_df)} responses, {len(citations_df)} citations"
    )

    # Test on larger subset but still manageable
    sample_threads = threads_df.head(1000)
    sample_thread_ids = set(sample_threads["thread_id"])
    sample_responses = responses_df[responses_df["thread_id"].isin(sample_thread_ids)]
    sample_citations = citations_df[
        citations_df["response_id"].isin(sample_responses["response_id"])
    ]

    print(
        f"Testing with {len(sample_threads)} threads, {len(sample_responses)} responses, {len(sample_citations)} citations"
    )

    print("\nTransforming to leaderboard format...")
    leaderboard_df = create_leaderboard_format(
        sample_threads, sample_responses, sample_citations
    )

    print(f"Created leaderboard format with {len(leaderboard_df)} conversations")

    # Filter out ties for cleaner testing
    battles_no_ties = leaderboard_df[
        ~leaderboard_df["winner"].isin(["tie", "tie (bothbad)"])
    ]
    print(f"Conversations without ties: {len(battles_no_ties)}")

    if len(battles_no_ties) < 10:
        print("Not enough battles without ties, using all data...")
        battles_no_ties = leaderboard_df

    # Check data structure
    print("\nData structure check:")
    print(f"Models A: {sorted(battles_no_ties['model_a'].unique())}")
    print(f"Models B: {sorted(battles_no_ties['model_b'].unique())}")
    print(f"Winner distribution: {battles_no_ties['winner'].value_counts().to_dict()}")

    # Test basic leaderboard
    print("\nTesting basic leaderboard...")
    try:
        anchor_model = (
            battles_no_ties["model_a"].value_counts().index[0]
        )  # Use most common model as anchor
        print(f"Using anchor model: {anchor_model}")

        leaderboard_table, bt_bootstrap, style_coef = run_leaderboard(
            battles_no_ties,
            anchor_model=anchor_model,
            anchor_rating=1000,
            num_bootstrap_samples=5,  # Small number for testing
        )

        print("Basic leaderboard results:")
        print(leaderboard_table[["rating", "final_ranking"]])

        # Test with style control
        print("\nTesting style control...")
        style_elements = ["standardized_citations_a", "standardized_citations_b"]

        leaderboard_style, bt_bootstrap_style, style_coef_bootstrap = run_leaderboard(
            battles_no_ties,
            anchor_model=anchor_model,
            anchor_rating=1000,
            style_elements=style_elements,
            num_bootstrap_samples=5,
        )

        print("Style-controlled leaderboard results:")
        print(leaderboard_style[["rating", "final_ranking"]])

        print("\nTest completed successfully!")
        return leaderboard_df

    except Exception as e:
        print(f"Error in leaderboard computation: {e}")
        print("Debugging information:")
        print(
            f"  Sample conv_metadata: {list(battles_no_ties['conv_metadata'].iloc[0].keys())}"
        )
        print("  First few rows:")
        print(battles_no_ties[["model_a", "model_b", "winner"]].head())
        raise


if __name__ == "__main__":
    test_full_pipeline()
