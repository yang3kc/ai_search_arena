#!/usr/bin/env python3
"""
Preference Analysis Script for News Citation Effects.

This script performs comprehensive preference analysis using modular components.
It coordinates Bradley-Terry ratings, individual feature effects, and citation style analysis.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Import modular components
from utils import load_battle_data, filter_battle_data, save_results, print_summary
from bt_ratings import main as bt_main
from individual_effects import analyze_individual_effects, print_individual_summary
from citation_style_effects import (
    create_default_model_specifications,
    analyze_citation_style_effects,
    compare_models,
)


def analyze_preferences(battle_df, config):
    """Run complete preference analysis using modular components."""
    print("Running comprehensive preference analysis...")

    results = {}

    # 1. Basic Bradley-Terry ratings (no style controls)
    print("\n=== Basic Bradley-Terry Ratings ===")
    from utils import compute_bradley_terry_ratings

    bt_results = compute_bradley_terry_ratings(
        battle_df,
        config["statistical_analysis"]["anchor_model"],
        config["statistical_analysis"]["anchor_rating"],
    )

    if bt_results:
        results["bradley_terry_ratings"] = bt_results
        print("Model ratings:")
        for model, rating in sorted(
            bt_results["model_ratings"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {model}: {rating:.1f}")

    # 2. Individual feature analysis
    print("\n=== Individual Feature Effects ===")
    features = config["response_signals"]["primary_features"]
    individual_results = analyze_individual_effects(battle_df, features, config)

    if individual_results:
        results["individual_feature_effects"] = individual_results

    # 3. Citation style effects with multiple model specifications
    print("\n=== Citation Style Effects (Multiple Models) ===")
    model_specs = create_default_model_specifications()

    # Run analysis for selected model specifications
    selected_models = [
        "basic_response",
        "political_bias",
        "source_quality",
        "full_primary",
    ]
    citation_results = analyze_citation_style_effects(
        battle_df, model_specs, config, selected_models
    )

    if citation_results:
        results["citation_style_effects"] = citation_results
        compare_models(citation_results)

    return results


def main():
    """Main entry point for comprehensive preference analysis."""
    # Get input/output paths from Snakemake
    input_battle_data = snakemake.input[0]
    output_results = snakemake.output.results
    output_coefficients = snakemake.output.coefficients

    # Get parameters from Snakemake
    bootstrap_samples = snakemake.params.bootstrap_samples
    random_seed = snakemake.params.random_seed

    # Create config from Snakemake parameters
    config = {
        "filtering": {
            "exclude_ties": True,
            "valid_winners": ["model_a", "model_b"],
            "min_battles_per_analysis": 50,
        },
        "statistical_analysis": {
            "anchor_model": "gpt-4o-search-preview",
            "anchor_rating": 1000.0,
            "bootstrap_samples": bootstrap_samples,
            "random_seed": random_seed,
        },
        "response_signals": {
            "primary_features": [
                "response_word_count",
                "num_citations",
                "proportion_news",
                "proportion_left_leaning",
                "proportion_right_leaning",
                "proportion_high_quality",
                "proportion_low_quality",
            ]
        },
    }

    try:
        print("=" * 80)
        print("COMPREHENSIVE PREFERENCE ANALYSIS")
        print("=" * 80)

        # Load battle data
        battle_df = load_battle_data(input_battle_data)

        # Filter data
        filtered_battle_df = filter_battle_data(battle_df, config)

        # Run comprehensive analysis using modular components
        results = analyze_preferences(filtered_battle_df, config)

        # Save results
        coefficients_df = save_results(results, output_results, output_coefficients)

        # Print summary
        print_summary(results)

        print(f"\nComprehensive analysis completed successfully!")
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
