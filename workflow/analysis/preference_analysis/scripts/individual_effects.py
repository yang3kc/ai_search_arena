#!/usr/bin/env python3
"""
Individual Feature Effects Analysis.

This script examines the effects of individual citation features on user preferences,
running separate analyses for each feature to understand their isolated impact.
"""

import sys

# Import utility functions
from bt_utils import (
    load_battle_data,
    filter_battle_data,
    compute_style_coefficients,
    save_results,
)


def analyze_individual_effects(battle_df, features, config):
    """Analyze individual feature effects separately."""
    print("Analyzing individual feature effects...")

    individual_results = {}

    for feature in features:
        print(f"\n--- Analyzing {feature} ---")

        # Check if feature data exists
        diff_col = f"{feature}_diff"
        if diff_col not in battle_df.columns:
            print(f"  WARNING: {diff_col} not found in battle data, skipping")
            continue

        # Analyze single feature
        feature_result = compute_style_coefficients(
            battle_df,
            [feature],
            config["statistical_analysis"]["bootstrap_samples"],
            config["statistical_analysis"]["random_seed"],
        )

        if feature_result:
            individual_results[feature] = feature_result
            coeff = feature_result["coefficients"][feature]
            ci = feature_result["confidence_intervals"][feature]
            significant = not (ci["lower"] <= 0 <= ci["upper"])
            sig_marker = " *" if significant else ""

            print(
                f"  Coefficient: {coeff:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]{sig_marker}"
            )
            print(f"  Log-likelihood: {feature_result['log_likelihood']:.2f}")
            print(
                f"  Bootstrap success rate: {feature_result['bootstrap_success_rate']:.1%}"
            )
        else:
            print(f"  ERROR: Failed to analyze {feature}")

    return individual_results


def print_individual_summary(individual_results):
    """Print summary of individual feature effects."""
    print("\n" + "=" * 60)
    print("INDIVIDUAL FEATURE EFFECTS SUMMARY")
    print("=" * 60)

    if not individual_results:
        print("No individual feature effects computed.")
        return

    # Collect all effects for ranking
    effects = []
    for feature, results in individual_results.items():
        coeff = results["coefficients"][feature]
        ci = results["confidence_intervals"][feature]
        significant = not (ci["lower"] <= 0 <= ci["upper"])
        effects.append((feature, coeff, ci, significant))

    # Sort by effect size (absolute value)
    effects.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nFeature Effects (ranked by magnitude):")
    print("  (* = statistically significant)")

    for feature, coeff, ci, significant in effects:
        sig_marker = " *" if significant else ""
        direction = "positive" if coeff > 0 else "negative" if coeff < 0 else "neutral"
        print(
            f"  {feature}: {coeff:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}] ({direction}){sig_marker}"
        )

    # Count significant effects
    significant_count = sum(1 for _, _, _, sig in effects if sig)
    print(f"\nSignificant effects: {significant_count}/{len(effects)} features")

    print("\n" + "=" * 60)


def main():
    """Analyze individual feature effects."""
    # Configuration
    config = {
        "filtering": {
            "exclude_ties": True,
            "valid_winners": ["model_a", "model_b"],
            "min_battles_per_analysis": 50,
        },
        "statistical_analysis": {
            "bootstrap_samples": 500,  # Fewer samples for individual analysis
            "random_seed": 42,
        },
        "response_signals": {
            "primary_features": [
                "response_length",
                "response_word_count",
                "num_citations",
                "proportion_news",
                "proportion_academic",
                "proportion_social_media",
                "proportion_unclassified",
                "proportion_wiki",
                "proportion_gov_edu",
                "proportion_tech",
                "proportion_search_engine",
                "proportion_community_blog",
                "proportion_left_leaning",
                "proportion_right_leaning",
                "proportion_unknown_leaning",
                "proportion_high_quality",
                "proportion_low_quality",
                "proportion_unknown_quality",
                "news_proportion_high_quality",
                "news_proportion_low_quality",
                "news_proportion_unknown_quality",
                "news_proportion_left_leaning",
                "news_proportion_right_leaning",
                "news_proportion_unknown_leaning",
            ]
        },
    }

    try:
        # Get input/output paths from Snakemake or command line
        if "snakemake" in globals():
            input_battle_data = snakemake.input[0]
            output_results = snakemake.output.results
            output_coefficients = snakemake.output.coefficients
            # Override config with Snakemake parameters if available
            if hasattr(snakemake, "params"):
                if hasattr(snakemake.params, "bootstrap_samples"):
                    config["statistical_analysis"]["bootstrap_samples"] = (
                        snakemake.params.bootstrap_samples
                    )
                if hasattr(snakemake.params, "random_seed"):
                    config["statistical_analysis"]["random_seed"] = (
                        snakemake.params.random_seed
                    )
        else:
            # For standalone execution
            if len(sys.argv) < 4:
                print(
                    "Usage: python individual_effects.py <battle_data.parquet> <output_results.json> <output_coefficients.csv>"
                )
                return 1
            input_battle_data = sys.argv[1]
            output_results = sys.argv[2]
            output_coefficients = sys.argv[3]

        print("=" * 60)
        print("INDIVIDUAL FEATURE EFFECTS ANALYSIS")
        print("=" * 60)

        # Load battle data
        battle_df = load_battle_data(input_battle_data)

        # Filter data
        filtered_battle_df = filter_battle_data(battle_df, config)

        # Analyze individual features
        features = config["response_signals"]["primary_features"]
        individual_results = analyze_individual_effects(
            filtered_battle_df, features, config
        )

        if not individual_results:
            print("ERROR: No individual feature effects computed")
            return 1

        # Organize results
        results = {"individual_feature_effects": individual_results}

        # Save results
        save_results(results, output_results, output_coefficients)

        # Print summary
        print_individual_summary(individual_results)

        print("\nIndividual feature effects analysis completed successfully!")
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
