#!/usr/bin/env python3
"""
Preference Analysis Script for News Citation Effects.

This script performs Bradley-Terry model analysis with bootstrap confidence intervals
to understand how different citation patterns affect user preferences in AI responses.
"""

import pandas as pd
import numpy as np
import scipy.optimize as opt
from pathlib import Path
import json
import sys


def load_battle_data(battle_data_path):
    """Load the battle data from create_battles phase."""
    print("Loading battle data...")

    battle_df = pd.read_parquet(battle_data_path)

    print(f"Loaded {len(battle_df):,} battles")
    print(
        f"Unique models: {len(set(battle_df['model_a']) | set(battle_df['model_b']))}"
    )
    print(f"Winner distribution: {dict(battle_df['winner'].value_counts())}")

    return battle_df


def filter_battle_data(battle_df, config):
    """Filter battle data according to configuration settings."""
    print("Filtering battle data...")

    initial_count = len(battle_df)

    # Filter out ties if configured
    if config["filtering"]["exclude_ties"]:
        valid_winners = config["filtering"]["valid_winners"]
        battle_df = battle_df[battle_df["winner"].isin(valid_winners)].copy()
        print(f"  Excluded ties: {initial_count - len(battle_df):,} battles removed")

    # Check minimum battles threshold
    min_battles = config["filtering"]["min_battles_per_analysis"]
    if len(battle_df) < min_battles:
        raise ValueError(
            f"Insufficient battles for analysis: {len(battle_df)} < {min_battles}"
        )

    print(f"Final battle count: {len(battle_df):,}")
    return battle_df


def compute_bradley_terry_ratings(battle_df, anchor_model, anchor_rating=1000.0):
    """
    Compute Bradley-Terry model ratings using maximum likelihood estimation.

    Args:
        battle_df: DataFrame with model_a, model_b, winner columns
        anchor_model: Model to use as anchor (fixed rating)
        anchor_rating: Rating to assign to anchor model

    Returns:
        Dictionary with model ratings and log-likelihood
    """
    print("Computing Bradley-Terry ratings...")

    # Get unique models
    models = sorted(set(battle_df["model_a"]) | set(battle_df["model_b"]))
    model_to_idx = {model: i for i, model in enumerate(models)}
    n_models = len(models)

    print(f"  Models in analysis: {models}")

    # Find anchor model index
    if anchor_model not in model_to_idx:
        print(f"  WARNING: Anchor model '{anchor_model}' not found, using first model")
        anchor_idx = 0
        anchor_model = models[0]
    else:
        anchor_idx = model_to_idx[anchor_model]

    print(f"  Anchor model: {anchor_model} (index {anchor_idx})")

    # Prepare battle data
    battles = []
    outcomes = []

    for _, row in battle_df.iterrows():
        model_a_idx = model_to_idx[row["model_a"]]
        model_b_idx = model_to_idx[row["model_b"]]

        battles.append((model_a_idx, model_b_idx))
        # Winner is 1 if model_a wins, 0 if model_b wins
        outcomes.append(1 if row["winner"] == "model_a" else 0)

    battles = np.array(battles)
    outcomes = np.array(outcomes)

    # Objective function for MLE
    def neg_log_likelihood(ratings):
        # Compute win probabilities using Bradley-Terry model
        rating_diffs = ratings[battles[:, 0]] - ratings[battles[:, 1]]
        probs = 1 / (1 + np.exp(-rating_diffs))
        probs = np.clip(probs, 1e-10, 1 - 1e-10)  # Numerical stability

        # Compute log-likelihood
        ll = np.sum(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))
        return -ll

    # Initial ratings (all zero except anchor)
    initial_ratings = np.zeros(n_models)

    # Optimize ratings with anchor constraint
    def objective_with_anchor(free_ratings):
        # Reconstruct full ratings with anchor fixed
        ratings = np.zeros(n_models)
        free_idx = 0
        for i in range(n_models):
            if i == anchor_idx:
                ratings[i] = anchor_rating
            else:
                ratings[i] = free_ratings[free_idx]
                free_idx += 1
        return neg_log_likelihood(ratings)

    # Optimize free parameters (all except anchor)
    initial_free = np.zeros(n_models - 1)

    try:
        result = opt.minimize(objective_with_anchor, initial_free, method="BFGS")

        if not result.success:
            print("  WARNING: Optimization may not have converged")

        # Reconstruct final ratings
        final_ratings = np.zeros(n_models)
        free_idx = 0
        for i in range(n_models):
            if i == anchor_idx:
                final_ratings[i] = anchor_rating
            else:
                final_ratings[i] = result.x[free_idx]
                free_idx += 1

        # Create results dictionary
        model_ratings = {models[i]: final_ratings[i] for i in range(n_models)}
        log_likelihood = -result.fun

        print(f"  Optimization successful: log-likelihood = {log_likelihood:.2f}")

        return {
            "model_ratings": model_ratings,
            "log_likelihood": log_likelihood,
            "n_battles": len(battle_df),
            "n_models": n_models,
            "anchor_model": anchor_model,
            "anchor_rating": anchor_rating,
        }

    except Exception as e:
        print(f"  ERROR: Bradley-Terry optimization failed: {e}")
        return None


def compute_style_coefficients(battle_df, features, num_bootstrap=1000, random_seed=42):
    """
    Compute style coefficients for citation features using Bradley-Terry with controls.

    Args:
        battle_df: Battle data with feature differences
        features: List of features to analyze (without _diff suffix)
        num_bootstrap: Number of bootstrap samples
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with coefficients and confidence intervals
    """
    print(f"Computing style coefficients for {len(features)} features...")

    np.random.seed(random_seed)

    # Find available difference columns
    available_features = []
    for feature in features:
        diff_col = f"{feature}_diff"
        if diff_col in battle_df.columns:
            available_features.append(feature)
        else:
            print(f"  WARNING: {diff_col} not found in battle data")

    if not available_features:
        raise ValueError("No valid feature difference columns found")

    print(f"  Analyzing features: {available_features}")

    # Get models for Bradley-Terry component
    models = sorted(set(battle_df["model_a"]) | set(battle_df["model_b"]))
    model_to_idx = {model: i for i, model in enumerate(models)}
    n_models = len(models)
    n_features = len(available_features)

    # Prepare data matrices
    battles = []
    outcomes = []
    feature_diffs = []

    for _, row in battle_df.iterrows():
        model_a_idx = model_to_idx[row["model_a"]]
        model_b_idx = model_to_idx[row["model_b"]]

        # Get feature differences and check for NaN values
        feature_diff = []
        valid_row = True

        for feature in available_features:
            diff_val = row[f"{feature}_diff"]
            if pd.isna(diff_val):
                # Replace NaN with 0 or skip this battle
                diff_val = 0.0
            feature_diff.append(diff_val)

        # Only include battles with valid feature data
        if valid_row and not any(pd.isna(f) for f in feature_diff):
            battles.append((model_a_idx, model_b_idx))
            outcomes.append(1 if row["winner"] == "model_a" else 0)
            feature_diffs.append(feature_diff)

    battles = np.array(battles)
    outcomes = np.array(outcomes)
    feature_diffs = np.array(feature_diffs)

    if len(battles) == 0:
        print(f"  ERROR: No valid battles found after filtering NaN values")
        return None

    print(
        f"  Using {len(battles):,} battles for analysis (filtered from {len(battle_df):,})"
    )

    # Fit main model
    def fit_model(battle_indices):
        """Fit Bradley-Terry model with style controls for given battle subset."""
        sub_battles = battles[battle_indices]
        sub_outcomes = outcomes[battle_indices]
        sub_features = feature_diffs[battle_indices]

        def neg_log_likelihood(params):
            # Split parameters: model ratings + feature coefficients
            ratings = params[:n_models]
            coefficients = params[n_models:]

            # Compute win probabilities
            rating_diffs = ratings[sub_battles[:, 0]] - ratings[sub_battles[:, 1]]
            feature_effects = np.sum(sub_features * coefficients, axis=1)
            logits = rating_diffs + feature_effects

            # Clip logits to prevent overflow
            logits = np.clip(logits, -20, 20)

            probs = 1 / (1 + np.exp(-logits))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)

            ll = np.sum(
                sub_outcomes * np.log(probs) + (1 - sub_outcomes) * np.log(1 - probs)
            )

            # Check for NaN or infinite values
            if not np.isfinite(ll):
                return 1e10  # Return large positive value if invalid

            return -ll

        # Initial parameters (models=0, features=0)
        initial_params = np.zeros(n_models + n_features)

        try:
            # Try bounded optimization first
            bounds = [(-5, 5)] * n_models + [(-2, 2)] * n_features  # Reasonable bounds
            result = opt.minimize(
                neg_log_likelihood, initial_params, method="L-BFGS-B", bounds=bounds
            )

            if result.success and np.isfinite(result.fun):
                return result.x[n_models:], -result.fun

            # If bounded fails, try unbounded
            result = opt.minimize(neg_log_likelihood, initial_params, method="BFGS")
            if result.success and np.isfinite(result.fun):
                return result.x[n_models:], -result.fun

            return None, None
        except Exception as e:
            print(f"    Optimization failed: {e}")
            return None, None

    # Fit main model
    main_coeffs, main_ll = fit_model(np.arange(len(battles)))

    if main_coeffs is None:
        print("  ERROR: Main model fitting failed")
        return None

    print(f"  Main model fitted: log-likelihood = {main_ll:.2f}")

    # Bootstrap for confidence intervals
    print(f"  Running {num_bootstrap} bootstrap samples...")
    bootstrap_coefficients = []

    for i in range(num_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"    Bootstrap sample {i + 1}/{num_bootstrap}")

        # Sample battles with replacement
        bootstrap_indices = np.random.choice(
            len(battles), size=len(battles), replace=True
        )

        # Fit model on bootstrap sample
        boot_coeffs, _ = fit_model(bootstrap_indices)

        if boot_coeffs is not None:
            bootstrap_coefficients.append(boot_coeffs)

    if not bootstrap_coefficients:
        print("  ERROR: All bootstrap samples failed")
        return None

    bootstrap_coefficients = np.array(bootstrap_coefficients)
    print(f"  Successful bootstrap samples: {len(bootstrap_coefficients)}")

    # Compute confidence intervals
    ci_lower = np.percentile(bootstrap_coefficients, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_coefficients, 97.5, axis=0)

    # Organize results
    results = {
        "features": available_features,
        "coefficients": {
            feature: main_coeffs[i] for i, feature in enumerate(available_features)
        },
        "confidence_intervals": {
            feature: {"lower": ci_lower[i], "upper": ci_upper[i]}
            for i, feature in enumerate(available_features)
        },
        "log_likelihood": main_ll,
        "n_battles": len(battle_df),
        "n_bootstrap_samples": len(bootstrap_coefficients),
        "bootstrap_success_rate": len(bootstrap_coefficients) / num_bootstrap,
    }

    return results


def analyze_preferences(battle_df, config):
    """Run complete preference analysis with multiple feature sets."""
    print("Running preference analysis...")

    results = {}

    # 1. Basic Bradley-Terry ratings (no style controls)
    print("\n=== Basic Bradley-Terry Ratings ===")
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

    # 2. Style coefficient analysis for primary features
    print("\n=== Citation Style Effects ===")
    primary_features = config["response_signals"]["primary_features"]

    style_results = compute_style_coefficients(
        battle_df,
        primary_features,
        config["statistical_analysis"]["bootstrap_samples"],
        config["statistical_analysis"]["random_seed"],
    )

    if style_results:
        results["citation_style_effects"] = style_results
        print("Style coefficients (positive = helps model A win):")
        for feature in style_results["features"]:
            coeff = style_results["coefficients"][feature]
            ci = style_results["confidence_intervals"][feature]
            significant = not (ci["lower"] <= 0 <= ci["upper"])
            sig_marker = " *" if significant else ""
            print(
                f"  {feature}: {coeff:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]{sig_marker}"
            )

    # 3. Individual feature analysis
    print("\n=== Individual Feature Effects ===")
    individual_results = {}

    for feature in primary_features:
        print(f"\nAnalyzing {feature}...")
        feature_result = compute_style_coefficients(
            battle_df,
            [feature],
            config["statistical_analysis"]["bootstrap_samples"]
            // 2,  # Fewer samples for individual
            config["statistical_analysis"]["random_seed"],
        )

        if feature_result:
            individual_results[feature] = feature_result
            coeff = feature_result["coefficients"][feature]
            ci = feature_result["confidence_intervals"][feature]
            print(f"  Coefficient: {coeff:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    results["individual_feature_effects"] = individual_results

    return results


def save_results(results, output_results_path, output_coefficients_path):
    """Save analysis results to JSON and CSV files."""
    print("Saving results...")

    # Ensure output directories exist
    Path(output_results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_coefficients_path).parent.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    with open(output_results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Results saved to: {output_results_path}")

    # Create coefficients CSV for easy inspection
    coefficients_data = []

    # Add Bradley-Terry ratings
    if "bradley_terry_ratings" in results:
        bt_results = results["bradley_terry_ratings"]
        for model, rating in bt_results["model_ratings"].items():
            coefficients_data.append(
                {
                    "analysis_type": "bradley_terry_rating",
                    "feature": model,
                    "coefficient": rating,
                    "ci_lower": None,
                    "ci_upper": None,
                    "significant": None,
                    "n_battles": bt_results["n_battles"],
                }
            )

    # Add style coefficients
    if "citation_style_effects" in results:
        style_results = results["citation_style_effects"]
        for feature in style_results["features"]:
            coeff = style_results["coefficients"][feature]
            ci = style_results["confidence_intervals"][feature]
            significant = not (ci["lower"] <= 0 <= ci["upper"])

            coefficients_data.append(
                {
                    "analysis_type": "citation_style_effect",
                    "feature": feature,
                    "coefficient": coeff,
                    "ci_lower": ci["lower"],
                    "ci_upper": ci["upper"],
                    "significant": significant,
                    "n_battles": style_results["n_battles"],
                }
            )

    # Add individual feature effects
    if "individual_feature_effects" in results:
        for feature, feature_results in results["individual_feature_effects"].items():
            coeff = feature_results["coefficients"][feature]
            ci = feature_results["confidence_intervals"][feature]
            significant = not (ci["lower"] <= 0 <= ci["upper"])

            coefficients_data.append(
                {
                    "analysis_type": "individual_feature_effect",
                    "feature": feature,
                    "coefficient": coeff,
                    "ci_lower": ci["lower"],
                    "ci_upper": ci["upper"],
                    "significant": significant,
                    "n_battles": feature_results["n_battles"],
                }
            )

    # Save coefficients CSV
    coefficients_df = pd.DataFrame(coefficients_data)
    coefficients_df.to_csv(output_coefficients_path, index=False)

    print(f"  Coefficients saved to: {output_coefficients_path}")

    return coefficients_df


def print_summary(results):
    """Print a summary of key findings."""
    print("\n" + "=" * 60)
    print("PREFERENCE ANALYSIS SUMMARY")
    print("=" * 60)

    # Bradley-Terry ratings summary
    if "bradley_terry_ratings" in results:
        bt_results = results["bradley_terry_ratings"]
        print(f"\nModel Performance ({bt_results['n_battles']:,} battles):")
        sorted_models = sorted(
            bt_results["model_ratings"].items(), key=lambda x: x[1], reverse=True
        )
        for i, (model, rating) in enumerate(sorted_models):
            print(f"  {i + 1}. {model}: {rating:.1f}")

    # Citation style effects summary
    if "citation_style_effects" in results:
        style_results = results["citation_style_effects"]
        print(f"\nCitation Style Effects ({style_results['n_battles']:,} battles):")

        significant_effects = []
        for feature in style_results["features"]:
            coeff = style_results["coefficients"][feature]
            ci = style_results["confidence_intervals"][feature]
            if not (ci["lower"] <= 0 <= ci["upper"]):  # Significant
                direction = "positive" if coeff > 0 else "negative"
                significant_effects.append((feature, coeff, direction))

        if significant_effects:
            print("  Significant effects:")
            for feature, coeff, direction in sorted(
                significant_effects, key=lambda x: abs(x[1]), reverse=True
            ):
                print(f"    {feature}: {coeff:.4f} ({direction})")
        else:
            print("  No statistically significant citation effects found")

    print("\n" + "=" * 60)


def main():
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
                "response_length",
                "num_citations",
                "proportion_low_quality",
                "proportion_right_leaning",
            ]
        },
    }

    try:
        # Load battle data
        battle_df = load_battle_data(input_battle_data)

        # Filter data
        filtered_battle_df = filter_battle_data(battle_df, config)

        # Run analysis
        results = analyze_preferences(filtered_battle_df, config)

        # Save results
        coefficients_df = save_results(results, output_results, output_coefficients)

        # Print summary
        print_summary(results)

        print(f"\nAnalysis completed successfully!")
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
