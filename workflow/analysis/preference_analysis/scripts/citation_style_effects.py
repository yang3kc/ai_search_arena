#!/usr/bin/env python3
"""
Citation Style Effects Analysis with Flexible Model Specification.

This script examines citation style effects with the flexibility to specify
different models with different features, allowing observation of changes
in effects as features are added or modified.
"""

import pandas as pd
import sys
from pathlib import Path
import json

# Import utility functions
from bt_utils import (
    load_battle_data,
    filter_battle_data,
    compute_style_coefficients,
)


class FeatureModelSpecification:
    """Class to handle flexible feature model specifications."""

    def __init__(self):
        self.models = {}

    def add_model(self, name, features, description=""):
        """Add a model specification with given features."""
        self.models[name] = {"features": features, "description": description}

    def get_model(self, name):
        """Get model specification by name."""
        return self.models.get(name)

    def list_models(self):
        """List all available model specifications."""
        return list(self.models.keys())

    def print_models(self):
        """Print all model specifications."""
        print("Available Model Specifications:")
        for name, spec in self.models.items():
            features_str = ", ".join(spec["features"])
            print(f"  {name}: {len(spec['features'])} features")
            if spec["description"]:
                print(f"    Description: {spec['description']}")
            print(f"    Features: {features_str}")
            print()


def create_default_model_specifications():
    """Create default model specifications for common analyses."""
    spec = FeatureModelSpecification()

    # Basic response characteristics
    spec.add_model(
        "basic_response",
        ["response_word_count", "num_citations"],
        "Basic response characteristics: length and citation count. All models should include these two features as control variables.",
    )

    # Citation source types
    spec.add_model(
        "proportion_news",
        ["response_word_count", "num_citations", "proportion_news"],
        "Proportion of news sources in the citation pool.",
    )

    # Bias + quality among all citations
    # The results are not very interesting, so we're not including it in the analysis.
    # spec.add_model(
    #     "bias_and_quality_all_citations",
    #     [
    #         "response_word_count",
    #         "num_citations",
    #         "proportion_news",
    #         "proportion_left_leaning",
    #         "proportion_right_leaning",
    #         "proportion_high_quality",
    #         "proportion_low_quality",
    #     ],
    #     "Political and quality among all citations.",
    # )

    # Bias + quality among news citations
    spec.add_model(
        "bias_and_quality_news_citations",
        [
            "response_word_count",
            "num_citations",
            "proportion_news",
            "news_proportion_left_leaning",
            "news_proportion_right_leaning",
            "news_proportion_unknown_leaning",
            "news_proportion_high_quality",
            "news_proportion_low_quality",
        ],
        "Political and quality among news citations.",
    )

    return spec


def analyze_citation_style_effects(battle_df, model_specs, config):
    """Analyze citation style effects for multiple model specifications."""
    print("Analyzing citation style effects with flexible model specifications...")

    results = {}

    # Use all models if none specified
    selected_models = model_specs.list_models()

    print(f"Running analysis for {len(selected_models)} model specifications")

    for model_name in selected_models:
        model_spec = model_specs.get_model(model_name)
        if not model_spec:
            print(f"WARNING: Model '{model_name}' not found, skipping")
            continue

        features = model_spec["features"]
        print(f"\n--- Analyzing Model: {model_name} ---")
        print(f"Description: {model_spec['description']}")
        print(f"Features ({len(features)}): {', '.join(features)}")

        # Check if all features are available
        available_features = []
        missing_features = []
        for feature in features:
            diff_col = f"{feature}_diff"
            if diff_col in battle_df.columns:
                available_features.append(feature)
            else:
                missing_features.append(feature)

        if missing_features:
            print(f"  WARNING: Missing features: {missing_features}")

        if not available_features:
            print(f"  ERROR: No available features for model {model_name}, skipping")
            continue

        print(f"  Using features: {available_features}")

        # Compute style coefficients
        style_result = compute_style_coefficients(
            battle_df,
            available_features,
            config["statistical_analysis"]["bootstrap_samples"],
            config["statistical_analysis"]["random_seed"],
        )

        if style_result:
            # Add model metadata
            style_result["model_name"] = model_name
            style_result["model_description"] = model_spec["description"]
            style_result["requested_features"] = features
            style_result["missing_features"] = missing_features

            results[model_name] = style_result

            # Print results
            print(f"  Log-likelihood: {style_result['log_likelihood']:.2f}")
            print(
                f"  Bootstrap success rate: {style_result['bootstrap_success_rate']:.1%}"
            )
            print("  Feature coefficients:")

            for feature in style_result["features"]:
                coeff = style_result["coefficients"][feature]
                ci = style_result["confidence_intervals"][feature]
                significant = not (ci["lower"] <= 0 <= ci["upper"])
                sig_marker = " *" if significant else ""
                print(
                    f"    {feature}: {coeff:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]{sig_marker}"
                )
        else:
            print(f"  ERROR: Failed to analyze model {model_name}")

    return results


def compare_models(results):
    """Compare results across different model specifications."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    if not results:
        print("No results to compare.")
        return

    # Compare log-likelihoods
    print("\nModel Performance (Log-likelihood):")
    ll_comparison = []
    for model_name, result in results.items():
        ll = result["log_likelihood"]
        n_features = len(result["features"])
        ll_comparison.append((model_name, ll, n_features))

    # Sort by log-likelihood (higher is better)
    ll_comparison.sort(key=lambda x: x[1], reverse=True)

    for i, (model_name, ll, n_features) in enumerate(ll_comparison):
        print(f"  {i + 1}. {model_name}: {ll:.2f} ({n_features} features)")

    # Find common features across models
    all_features = set()
    for result in results.values():
        all_features.update(result["features"])

    print(f"\nFeature Effects Across Models:")
    print("  (Showing features that appear in multiple models)")

    for feature in sorted(all_features):
        print(f"\n  {feature}:")
        feature_effects = []

        for model_name, result in results.items():
            if feature in result["features"]:
                coeff = result["coefficients"][feature]
                ci = result["confidence_intervals"][feature]
                significant = not (ci["lower"] <= 0 <= ci["upper"])
                feature_effects.append((model_name, coeff, ci, significant))

        if len(feature_effects) > 1:  # Only show if in multiple models
            for model_name, coeff, ci, significant in feature_effects:
                sig_marker = " *" if significant else ""
                print(
                    f"    {model_name}: {coeff:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]{sig_marker}"
                )

    print("\n" + "=" * 80)


def save_flexible_results(results, output_results_path, output_coefficients_path):
    """Save results with model comparison information."""
    print("Saving flexible model results...")

    # Ensure output directories exist
    Path(output_results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_coefficients_path).parent.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    with open(output_results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Results saved to: {output_results_path}")

    # Create detailed coefficients CSV
    coefficients_data = []

    for model_name, model_results in results.items():
        for feature in model_results["features"]:
            coeff = model_results["coefficients"][feature]
            ci = model_results["confidence_intervals"][feature]
            significant = not (ci["lower"] <= 0 <= ci["upper"])

            coefficients_data.append(
                {
                    "model_specification": model_name,
                    "model_description": model_results.get("model_description", ""),
                    "feature": feature,
                    "coefficient": coeff,
                    "ci_lower": ci["lower"],
                    "ci_upper": ci["upper"],
                    "significant": significant,
                    "log_likelihood": model_results["log_likelihood"],
                    "n_features": len(model_results["features"]),
                    "n_battles": model_results["n_battles"],
                    "bootstrap_success_rate": model_results["bootstrap_success_rate"],
                }
            )

    # Save coefficients CSV
    coefficients_df = pd.DataFrame(coefficients_data)
    coefficients_df.to_csv(output_coefficients_path, index=False)

    print(f"  Coefficients saved to: {output_coefficients_path}")

    return coefficients_df


def main():
    """Analyze citation style effects with flexible model specifications."""
    # Configuration
    config = {
        "filtering": {
            "exclude_ties": True,
            "valid_winners": ["model_a", "model_b"],
            "min_battles_per_analysis": 50,
        },
        "statistical_analysis": {
            "bootstrap_samples": 1000,
            "random_seed": 42,
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
                    "Usage: python citation_style_effects.py <battle_data.parquet> <output_results.json> <output_coefficients.csv> [model1,model2,...]"
                )
                return 1
            input_battle_data = sys.argv[1]
            output_results = sys.argv[2]
            output_coefficients = sys.argv[3]

        print("=" * 80)
        print("CITATION STYLE EFFECTS ANALYSIS - FLEXIBLE MODEL SPECIFICATION")
        print("=" * 80)

        # Create model specifications
        model_specs = create_default_model_specifications()
        print("\n")
        model_specs.print_models()

        # Load battle data
        battle_df = load_battle_data(input_battle_data)

        # Filter data
        filtered_battle_df = filter_battle_data(battle_df, config)

        # Run analysis for specified models
        results = analyze_citation_style_effects(
            filtered_battle_df, model_specs, config
        )

        if not results:
            print("ERROR: No citation style effects computed")
            return 1

        # Compare models
        compare_models(results)

        # Save results
        save_flexible_results(results, output_results, output_coefficients)

        print(f"\nCitation style effects analysis completed successfully!")
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
