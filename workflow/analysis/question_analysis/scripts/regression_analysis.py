#!/usr/bin/env python3
"""
Regression Analysis Script for Question Analysis Pipeline.

This script analyzes relationships between question features and citation patterns using:
- Linear regression for continuous outcomes
- Model diagnostics and validation
- Feature importance analysis
- Comprehensive statistical reporting
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_cleaned_data(input_path):
    """Load the cleaned features dataset."""
    logger.info(f"Loading cleaned data from {input_path}")
    data = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(data):,} rows with {len(data.columns)} columns")
    return data


def identify_variable_groups(data):
    """Identify different groups of variables in the dataset."""
    logger.info("Identifying variable groups...")

    # Embedding dimensions (original and PCA) - keep pattern matching for these
    embedding_cols = [col for col in data.columns if col.startswith("embedding_dim_")]
    pca_cols = [col for col in data.columns if col.startswith("embedding_pc_")]

    # Citation pattern outcomes (dependent variables) - explicitly listed
    citation_outcomes = [
        "proportion_left_leaning",
        "proportion_right_leaning",
        "proportion_center_leaning",
        "proportion_high_quality",
        "proportion_low_quality",
        "news_proportion_left_leaning",
        "news_proportion_right_leaning",
        "news_proportion_center_leaning",
        "news_proportion_high_quality",
        "news_proportion_low_quality",
        "proportion_news",
        "num_citations",
    ]

    # Question features - explicitly listed
    question_features = [
        "question_length_chars_log",
        "question_length_words_log",
        "turn_number",
        "total_turns",
    ]

    # Response features - explicitly listed
    response_features = [
        "response_length_log",
        "response_word_count_log",
    ]

    # Categorical dummy variables - explicitly listed by group
    dummy_vars = []
    
    # Client country dummies (based on actual data)
    client_country_dummies = [
        "client_country_US", "client_country_GB", "client_country_CA", 
        "client_country_DE", "client_country_CN", "client_country_HK",
        "client_country_IN", "client_country_MX", "client_country_RU",
        "client_country_Other", "client_country_nan"
    ]
    dummy_vars.extend([col for col in client_country_dummies if col in data.columns])
    
    # Model family dummies (based on actual data)
    model_family_dummies = [
        "model_family_openai", "model_family_perplexity", "model_family_nan"
    ]
    dummy_vars.extend([col for col in model_family_dummies if col in data.columns])
    
    # Primary intent dummies (based on actual data)
    primary_intent_dummies = [
        "primary_intent_Creative Generation", "primary_intent_Explanation", 
        "primary_intent_Factual Lookup", "primary_intent_Guidance",
        "primary_intent_Info Synthesis", "primary_intent_Other",
        "primary_intent_Recommendation", "primary_intent_Text Processing", 
        "primary_intent_nan"
    ]
    dummy_vars.extend([col for col in primary_intent_dummies if col in data.columns])

    # Source composition variables - explicitly listed
    source_composition = [
        "proportion_academic",
        "proportion_community_blog", 
        "proportion_gov_edu",
        "proportion_other",
        "proportion_search_engine",
        "proportion_social_media",
        "proportion_tech",
        "proportion_unclassified",
        "proportion_unknown_leaning",
        "proportion_unknown_quality",
        "proportion_wiki",
    ]

    # Validate which variables actually exist in the data
    def filter_existing_vars(var_list, var_type):
        existing = [var for var in var_list if var in data.columns]
        missing = [var for var in var_list if var not in data.columns]
        if missing:
            logger.warning(f"Missing {var_type} variables: {missing}")
        return existing

    # Filter to only existing variables
    citation_outcomes = filter_existing_vars(citation_outcomes, "citation outcome")
    question_features = filter_existing_vars(question_features, "question feature")
    response_features = filter_existing_vars(response_features, "response feature")
    dummy_vars = filter_existing_vars(dummy_vars, "dummy variable")
    source_composition = filter_existing_vars(source_composition, "source composition")

    logger.info(f"Original embedding dimensions: {len(embedding_cols)}")
    logger.info(f"PCA embedding dimensions: {len(pca_cols)}")
    logger.info(f"Citation outcomes: {len(citation_outcomes)}")
    logger.info(f"Question features: {len(question_features)}")
    logger.info(f"Response features: {len(response_features)}")
    logger.info(f"Dummy variables: {len(dummy_vars)}")
    logger.info(f"Source composition: {len(source_composition)}")

    return {
        "embeddings": embedding_cols,
        "pca_embeddings": pca_cols,
        "outcomes": citation_outcomes,
        "question_features": question_features,
        "response_features": response_features,
        "dummy_vars": dummy_vars,
        "source_composition": source_composition,
    }


def prepare_features_for_regression(data, variable_groups, use_pca=True):
    """Prepare feature sets for regression analysis."""
    logger.info("Preparing features for regression...")

    # Choose embedding features: PCA or original
    if use_pca and variable_groups["pca_embeddings"]:
        logger.info(
            f"Using {len(variable_groups['pca_embeddings'])} PCA embedding components"
        )
        embedding_features = variable_groups["pca_embeddings"]
    else:
        logger.info(
            f"Using {len(variable_groups['embeddings'])} original embedding dimensions"
        )
        embedding_features = variable_groups["embeddings"]

    # Handle dummy variables - exclude one from each group as reference
    dummy_vars_filtered = []
    
    # Define dummy variable groups with explicit reference categories
    dummy_groups = {
        "client_country_": "client_country_BR",  # Reference: Brazil (excluded from our list)
        "model_family_": "model_family_google",   # Reference: Google (excluded from our list)
        "primary_intent_": "primary_intent_Analysis"  # Reference: Analysis (excluded from our list)
    }

    for group_prefix, reference_var in dummy_groups.items():
        group_vars = [
            col for col in variable_groups["dummy_vars"] if col.startswith(group_prefix)
        ]
        if group_vars:
            # Sort to ensure consistent ordering
            group_vars.sort()
            
            # If the intended reference is in the data, exclude it; otherwise exclude the first
            if reference_var in group_vars:
                excluded_var = reference_var
                included_vars = [var for var in group_vars if var != reference_var]
            else:
                excluded_var = group_vars[0]
                included_vars = group_vars[1:]

            logger.info(
                f"Dummy group '{group_prefix}': excluding '{excluded_var}' as reference, including {len(included_vars)} variables"
            )
            dummy_vars_filtered.extend(included_vars)

    # Combine all predictor variables
    all_predictors = (
        embedding_features
        + variable_groups["question_features"]
        + variable_groups["response_features"]
        + dummy_vars_filtered
        + variable_groups["source_composition"]
    )

    # Create final feature matrix
    feature_data = data[all_predictors].copy()

    # Ensure all data is numeric
    feature_data = feature_data.astype(float)

    logger.info(f"Final feature matrix: {feature_data.shape}")
    logger.info(f"Using {'PCA' if use_pca else 'original'} embedding features")
    logger.info(
        f"Excluded {len(variable_groups['dummy_vars']) - len(dummy_vars_filtered)} dummy variables as reference categories"
    )

    return feature_data, all_predictors


def calculate_vif(X):
    """Calculate Variance Inflation Factor for multicollinearity diagnosis."""
    try:
        # Add constant for VIF calculation
        X_with_const = sm.add_constant(X)
        
        # Calculate VIF for each feature
        vif_data = []
        for i in range(1, X_with_const.shape[1]):  # Skip constant term
            try:
                vif_value = variance_inflation_factor(X_with_const.values, i)
                vif_data.append({
                    'feature': X_with_const.columns[i],
                    'vif': float(vif_value)
                })
            except:
                # Handle cases where VIF calculation fails
                vif_data.append({
                    'feature': X_with_const.columns[i],
                    'vif': np.nan
                })
        
        return vif_data
    except Exception as e:
        logger.warning(f"VIF calculation failed: {e}")
        return []


def run_regression_analysis(X, y, outcome_name):
    """Run OLS regression analysis using statsmodels for a single outcome."""
    logger.info(f"Running OLS regression for {outcome_name}")

    # Check for missing values
    if X.isnull().any().any() or y.isnull().any():
        logger.warning(f"Missing values detected in {outcome_name}, dropping rows")
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
    else:
        X_clean = X
        y_clean = y

    if len(X_clean) == 0:
        logger.error(f"No valid data for {outcome_name}")
        return None

    # Calculate multicollinearity diagnostics
    logger.info(f"Calculating multicollinearity diagnostics for {outcome_name}")
    vif_data = calculate_vif(X_clean)
    
    # Add intercept term
    X_with_const = sm.add_constant(X_clean)

    try:
        # Fit OLS model using statsmodels
        model = sm.OLS(y_clean, X_with_const)
        results_sm = model.fit()

        # Extract key statistics
        r2 = results_sm.rsquared
        adj_r2 = results_sm.rsquared_adj
        f_stat = results_sm.fvalue
        f_pvalue = results_sm.f_pvalue
        aic = results_sm.aic
        bic = results_sm.bic

        # Extract coefficient information
        coefficients = results_sm.params
        std_errors = results_sm.bse
        t_values = results_sm.tvalues
        p_values = results_sm.pvalues
        conf_int = results_sm.conf_int()

        # Create results dictionary
        results = {
            "outcome": outcome_name,
            "n_samples": len(y_clean),
            "n_features": X_clean.shape[1],
            "model_performance": {
                "r2": float(r2),
                "adj_r2": float(adj_r2),
                "f_statistic": float(f_stat),
                "f_pvalue": float(f_pvalue),
                "aic": float(aic),
                "bic": float(bic),
                "rmse": float(np.sqrt(results_sm.mse_resid)),
            },
            "coefficients": {
                "intercept": {
                    "coefficient": float(coefficients["const"]),
                    "std_error": float(std_errors["const"]),
                    "t_value": float(t_values["const"]),
                    "p_value": float(p_values["const"]),
                    "conf_int_lower": float(conf_int.loc["const", 0]),
                    "conf_int_upper": float(conf_int.loc["const", 1]),
                },
                "features": [
                    {
                        "feature": feature_name,
                        "coefficient": float(coefficients[feature_name]),
                        "std_error": float(std_errors[feature_name]),
                        "t_value": float(t_values[feature_name]),
                        "p_value": float(p_values[feature_name]),
                        "conf_int_lower": float(conf_int.loc[feature_name, 0]),
                        "conf_int_upper": float(conf_int.loc[feature_name, 1]),
                        "significant": bool(p_values[feature_name] < 0.05),
                    }
                    for feature_name in X_clean.columns
                ],
            },
        }

        # Sort features by absolute t-value (statistical significance)
        results["coefficients"]["features"].sort(
            key=lambda x: abs(x["t_value"]), reverse=True
        )

        # Add model diagnostics
        try:
            jb_stat, jb_pvalue = sm.stats.jarque_bera(results_sm.resid)
            dw_stat = sm.stats.durbin_watson(results_sm.resid)
        except:
            jb_stat, jb_pvalue = np.nan, np.nan
            dw_stat = np.nan

        # Multicollinearity analysis
        high_vif_features = [vif for vif in vif_data if not np.isnan(vif["vif"]) and vif["vif"] > 10]
        moderate_vif_features = [vif for vif in vif_data if not np.isnan(vif["vif"]) and 5 <= vif["vif"] <= 10]
        
        results["diagnostics"] = {
            "condition_number": float(np.linalg.cond(X_with_const)),
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pvalue": float(jb_pvalue),
            "durbin_watson": float(dw_stat),
            "n_significant_features": sum(
                1 for feat in results["coefficients"]["features"] if feat["significant"]
            ),
            "multicollinearity": {
                "vif_data": vif_data,
                "high_vif_count": len(high_vif_features),
                "moderate_vif_count": len(moderate_vif_features),
                "max_vif": max([vif["vif"] for vif in vif_data if not np.isnan(vif["vif"])]) if vif_data else np.nan,
                "mean_vif": np.mean([vif["vif"] for vif in vif_data if not np.isnan(vif["vif"])]) if vif_data else np.nan,
            },
        }

        return results

    except Exception as e:
        logger.error(f"Regression failed for {outcome_name}: {e}")
        return None


def analyze_model_differences(data, features, outcomes):
    """Analyze differences between model families."""
    logger.info("Analyzing model family differences...")

    model_analysis = {}

    # Get model family columns
    model_family_cols = [col for col in data.columns if col.startswith("model_family_")]

    for outcome in outcomes[:5]:  # Analyze top 5 outcomes
        if outcome not in data.columns:
            continue

        model_results = {}

        for model_col in model_family_cols:
            if model_col.endswith("_nan"):
                continue

            model_name = model_col.replace("model_family_", "")
            model_mask = data[model_col] == 1

            if model_mask.sum() < 100:  # Skip if too few samples
                continue

            model_data = data[model_mask]
            outcome_mean = model_data[outcome].mean()
            outcome_std = model_data[outcome].std()
            sample_size = len(model_data)

            model_results[model_name] = {
                "mean": float(outcome_mean),
                "std": float(outcome_std),
                "n": int(sample_size),
            }

        model_analysis[outcome] = model_results

    return model_analysis


def create_summary_statistics(data, variable_groups):
    """Create comprehensive summary statistics."""
    logger.info("Creating summary statistics...")

    summary = {}

    # Overall dataset stats
    summary["dataset"] = {
        "n_observations": len(data),
        "n_features": len(data.columns),
        "missing_values": int(data.isnull().sum().sum()),
    }

    # Outcome variable statistics
    summary["outcomes"] = {}
    for outcome in variable_groups["outcomes"]:
        if outcome in data.columns:
            series = data[outcome]
            summary["outcomes"][outcome] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "skewness": float(series.skew()),
                "missing": int(series.isnull().sum()),
            }

    # Categorical variable distributions
    summary["categorical"] = {}
    for cat_prefix in ["client_country_", "model_family_", "primary_intent_"]:
        cat_cols = [col for col in data.columns if col.startswith(cat_prefix)]
        if cat_cols:
            cat_summary = {}
            for col in cat_cols:
                if not col.endswith("_nan"):
                    cat_summary[col] = int(data[col].sum())
            summary["categorical"][cat_prefix.rstrip("_")] = cat_summary

    return summary


def main():
    """Main function for regression analysis."""
    try:
        # Get input/output paths from Snakemake
        input_path = snakemake.input[0]  # type: ignore
        output_path = snakemake.output[0]  # type: ignore

    except NameError:
        # Fallback for running outside Snakemake
        logger.info("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent

        input_path = (
            base_dir / "data/intermediate/question_analysis/cleaned_features.parquet"
        )
        output_path = base_dir / "data/output/question_analysis/regression_results.json"

    # Load data
    data = load_cleaned_data(input_path)

    # Identify variable groups
    variable_groups = identify_variable_groups(data)

    # Prepare features
    X, feature_names = prepare_features_for_regression(data, variable_groups)

    # Run regression analyses for key outcomes only (for performance)
    logger.info("Running regression analyses...")
    regression_results = []

    # Focus on most important outcomes
    key_outcomes = [
        "proportion_left_leaning",
        "proportion_right_leaning",
        "proportion_center_leaning",
        "proportion_high_quality",
        "proportion_low_quality",
        "news_proportion_left_leaning",
        "news_proportion_right_leaning",
        "news_proportion_center_leaning",
        "news_proportion_high_quality",
        "news_proportion_low_quality",
    ]

    for outcome in key_outcomes:
        if outcome not in data.columns:
            logger.warning(f"Outcome {outcome} not found in data")
            continue

        y = data[outcome].astype(float)

        # Run regression
        result = run_regression_analysis(X, y, outcome)
        if result:
            regression_results.append(result)

    # Model family analysis
    model_analysis = analyze_model_differences(data, X, variable_groups["outcomes"])

    # Summary statistics
    summary_stats = create_summary_statistics(data, variable_groups)

    # Compile final results
    final_results = {
        "metadata": {
            "analysis_type": "question_citation_regression",
            "timestamp": pd.Timestamp.now().isoformat(),
            "n_outcomes_analyzed": len(regression_results),
            "feature_engineering": {
                "pca_precomputed": True,
                "pca_used": True,
                "total_features": len(feature_names),
            },
            "regression_method": "OLS_statsmodels",
        },
        "summary_statistics": summary_stats,
        "regression_results": regression_results,
        "model_comparisons": model_analysis,
        "variable_groups": {
            k: len(v) if isinstance(v, list) else v for k, v in variable_groups.items()
        },
    }

    # Create output directory and save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Saved regression results to {output_path}")

    # Log key findings
    logger.info("=== REGRESSION ANALYSIS SUMMARY ===")
    logger.info(f"Analyzed {len(regression_results)} outcomes")

    # Best performing models
    if regression_results:
        best_r2 = max(regression_results, key=lambda x: x["model_performance"]["r2"])
        logger.info(
            f"Best R² achieved: {best_r2['model_performance']['r2']:.3f} for {best_r2['outcome']}"
        )
        logger.info(f"Adjusted R²: {best_r2['model_performance']['adj_r2']:.3f}")

        # Top significant features across all models
        all_significant_features = {}
        for result in regression_results:
            significant_features = [
                feat
                for feat in result["coefficients"]["features"]
                if feat["significant"]
            ][:10]  # Top 10 significant features per model

            for feat in significant_features:
                feat_name = feat["feature"]
                if feat_name not in all_significant_features:
                    all_significant_features[feat_name] = []
                all_significant_features[feat_name].append(abs(feat["t_value"]))

        # Average t-statistic across models
        avg_t_stats = {
            feat: np.mean(t_stats) for feat, t_stats in all_significant_features.items()
        }
        top_features = sorted(avg_t_stats.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        logger.info(
            "Top 10 most significant features (average |t-statistic| across models):"
        )
        for feat, avg_t in top_features:
            logger.info(f"  {feat}: {avg_t:.3f}")

        # Overall significance summary
        total_significant = sum(
            result["diagnostics"]["n_significant_features"]
            for result in regression_results
        )
        logger.info(
            f"Total significant features across all models: {total_significant}"
        )

    logger.info("✅ Regression analysis completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
