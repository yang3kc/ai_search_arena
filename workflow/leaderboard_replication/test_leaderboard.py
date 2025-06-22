"""
Test the leaderboard implementation with the 7k preference dataset.
"""

import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from leaderboard_core import compute_bt, run_leaderboard


def test_leaderboard():
    """Test leaderboard with preference dataset."""
    print("Loading preference dataset...")
    df = pd.read_parquet('/Users/yangkc/working/llm/ai_search_arena/data/raw_data/search-arena-v1-preference-7k.parquet')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Models: {sorted(df['model_a'].unique())}")
    print(f"Winner distribution: {df['winner'].value_counts().to_dict()}")
    
    # Filter out ties for basic test
    battles_no_ties = df[~df['winner'].isin(['tie', 'tie (bothbad)'])]
    print(f"Battles without ties: {len(battles_no_ties)}")
    
    # Test basic Bradley-Terry
    print("\nTesting basic Bradley-Terry...")
    bt_ratings = compute_bt(battles_no_ties)
    print("BT Ratings:")
    print(bt_ratings)
    
    # Test full leaderboard
    print("\nTesting full leaderboard...")
    anchor_model = 'api-gpt-4o-search'
    anchor_rating = 1000
    
    leaderboard_table, bt_bootstrap, style_coef = run_leaderboard(
        df, 
        anchor_model=anchor_model,
        anchor_rating=anchor_rating,
        num_bootstrap_samples=10  # Small number for testing
    )
    
    print("Leaderboard table:")
    print(leaderboard_table)
    
    print("\nBootstrap confidence intervals:")
    print(bt_bootstrap.quantile([0.025, 0.975]).T)
    
    # Test with style control 
    print("\nTesting with citation style control...")
    style_elements = ["standardized_citations_a", "standardized_citations_b"]
    
    leaderboard_style, bt_bootstrap_style, style_coef_bootstrap = run_leaderboard(
        df,
        anchor_model=anchor_model,
        anchor_rating=anchor_rating, 
        style_elements=style_elements,
        num_bootstrap_samples=10
    )
    
    print("Style-controlled leaderboard:")
    print(leaderboard_style)
    
    print("\nStyle coefficients:")
    if style_coef_bootstrap is not None:
        print(f"Standardized citations coefficient: {style_coef_bootstrap.mean(axis=0)[0]:.3f}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_leaderboard()