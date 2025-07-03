#!/usr/bin/env python3
"""
Bradley-Terry Ratings Calculator.

This script calculates basic Bradley-Terry model ratings for AI models
without any citation style controls. It's focused on model performance ranking.
"""

import sys

# Import utility functions
from bt_utils import (
    load_battle_data,
    filter_battle_data,
    compute_bradley_terry_ratings,
    save_results,
    print_summary,
)


def main():
    """Calculate Bradley-Terry ratings for AI models."""
    # Configuration for BT ratings only
    config = {
        "filtering": {
            "exclude_ties": True,
            "valid_winners": ["model_a", "model_b"],
            "min_battles_per_analysis": 50,
        },
        "statistical_analysis": {
            "anchor_model": "gpt-4o-search-preview",
            "anchor_rating": 1000.0,
        },
    }

    try:
        # Get input/output paths from Snakemake or command line
        if "snakemake" in globals():
            input_battle_data = snakemake.input[0]
            output_results = snakemake.output.results
            output_coefficients = snakemake.output.coefficients
        else:
            # For standalone execution
            if len(sys.argv) < 4:
                print(
                    "Usage: python bt_ratings.py <battle_data.parquet> <output_results.json> <output_coefficients.csv>"
                )
                return 1
            input_battle_data = sys.argv[1]
            output_results = sys.argv[2]
            output_coefficients = sys.argv[3]

        print("=" * 60)
        print("BRADLEY-TERRY RATINGS CALCULATOR")
        print("=" * 60)

        # Load battle data
        battle_df = load_battle_data(input_battle_data)

        # Filter data
        filtered_battle_df = filter_battle_data(battle_df, config)

        # Calculate Bradley-Terry ratings
        print("\n=== Computing Bradley-Terry Ratings ===")
        bt_results = compute_bradley_terry_ratings(
            filtered_battle_df,
            config["statistical_analysis"]["anchor_model"],
            config["statistical_analysis"]["anchor_rating"],
        )

        if not bt_results:
            print("ERROR: Failed to compute Bradley-Terry ratings")
            return 1

        # Organize results
        results = {"bradley_terry_ratings": bt_results}

        # Print model rankings
        print("\nModel Rankings:")
        sorted_models = sorted(
            bt_results["model_ratings"].items(), key=lambda x: x[1], reverse=True
        )
        for i, (model, rating) in enumerate(sorted_models):
            print(f"  {i + 1}. {model}: {rating:.1f}")

        # Save results
        save_results(results, output_results, output_coefficients)

        # Print summary
        print_summary(results)

        print("\nBradley-Terry ratings calculated successfully!")
        print(f"Results saved to: {output_results}")
        print(f"Coefficients saved to: {output_coefficients}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
