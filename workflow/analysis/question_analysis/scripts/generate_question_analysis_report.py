#!/usr/bin/env python3
"""
Generate comprehensive question analysis report with visualizations.

This script creates an HTML report analyzing the relationship between
question features and citation patterns, including:
- Summary statistics
- Correlation matrices
- Regression coefficient plots
- Feature importance visualizations
- Subgroup analyses
"""

import base64
import json
import logging
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set style for static plots
plt.style.use("default")
sns.set_palette("husl")


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def load_data(data_path, results_path):
    """Load the analysis data and regression results."""
    logger.info(f"Loading data from {data_path}")
    data = pd.read_parquet(data_path)

    logger.info(f"Loading regression results from {results_path}")
    with open(results_path, "r") as f:
        results = json.load(f)

    logger.info(f"Loaded {len(data):,} rows with {len(data.columns)} columns")
    return data, results


def create_summary_statistics_table(data):
    """Create comprehensive summary statistics."""
    logger.info("Creating summary statistics...")

    # Define variable groups
    embedding_cols = [col for col in data.columns if col.startswith("embedding_")]
    outcome_cols = [
        col
        for col in data.columns
        if any(
            pattern in col
            for pattern in [
                "proportion_left_leaning",
                "proportion_right_leaning",
                "proportion_high_quality",
                "proportion_news",
                "num_citations",
            ]
        )
    ]
    feature_cols = [
        col
        for col in data.columns
        if col.endswith("_log") or col in ["turn_number", "total_turns"]
    ]

    summary_stats = []

    # Outcome variables
    for col in outcome_cols:
        if col in data.columns:
            stats = {
                "Variable": col,
                "Type": "Citation Outcome",
                "Mean": f"{data[col].mean():.4f}",
                "Std": f"{data[col].std():.4f}",
                "Min": f"{data[col].min():.4f}",
                "Max": f"{data[col].max():.4f}",
                "N": f"{data[col].count():,}",
                "Missing": f"{data[col].isnull().sum():,}",
            }
            summary_stats.append(stats)

    # Feature variables
    for col in feature_cols:
        if col in data.columns:
            stats = {
                "Variable": col,
                "Type": "Question/Response Feature",
                "Mean": f"{data[col].mean():.4f}",
                "Std": f"{data[col].std():.4f}",
                "Min": f"{data[col].min():.4f}",
                "Max": f"{data[col].max():.4f}",
                "N": f"{data[col].count():,}",
                "Missing": f"{data[col].isnull().sum():,}",
            }
            summary_stats.append(stats)

    # Model family distribution
    model_cols = [col for col in data.columns if col.startswith("model_family_")]
    for col in model_cols:
        if col in data.columns and not col.endswith("_nan"):
            count = data[col].sum()
            if count > 0:
                stats = {
                    "Variable": col,
                    "Type": "Model Family",
                    "Mean": f"{count:,} observations",
                    "Std": f"{count / len(data) * 100:.1f}%",
                    "Min": "-",
                    "Max": "-",
                    "N": f"{len(data):,}",
                    "Missing": "0",
                }
                summary_stats.append(stats)

    return pd.DataFrame(summary_stats)


def create_correlation_matrix(data, max_vars=20):
    """Create correlation matrix for key variables."""
    logger.info("Creating correlation matrix...")

    # Select key variables for correlation analysis
    key_cols = []

    # Outcome variables
    outcome_cols = [
        "proportion_left_leaning",
        "proportion_right_leaning",
        "proportion_high_quality",
        "news_proportion_left_leaning",
        "proportion_news",
        "num_citations",
    ]
    key_cols.extend([col for col in outcome_cols if col in data.columns])

    # Feature variables
    feature_cols = [
        col
        for col in data.columns
        if col.endswith("_log") or col in ["turn_number", "total_turns"]
    ]
    key_cols.extend(feature_cols[:8])  # Top 8 feature variables

    # First few PCA components
    pca_cols = [col for col in data.columns if col.startswith("embedding_pc_")][:6]
    key_cols.extend(pca_cols)

    # Limit to max_vars total
    key_cols = key_cols[:max_vars]

    # Calculate correlation matrix
    corr_data = data[key_cols].select_dtypes(include=[np.number])
    corr_matrix = corr_data.corr()

    # Create matplotlib heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.columns)

    # Add text annotations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(
                j,
                i,
                f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation", rotation=270, labelpad=20)

    ax.set_title("Correlation Matrix: Key Variables", fontsize=14, pad=20)
    plt.tight_layout()

    return fig


def create_regression_coefficient_plots(results):
    """Create plots showing regression coefficients."""
    logger.info("Creating regression coefficient plots...")

    plots = []

    for result in results["regression_results"]:
        outcome = result["outcome"]

        # Get all significant features except embedding features
        features = result["coefficients"]["features"]
        features_to_plot = [
            f
            for f in features
            if not f["feature"].startswith("embedding_dim_")
            and not f["feature"].startswith("embedding_pc_")
        ]

        # Add intercept to the beginning of the list
        intercept = result["coefficients"]["intercept"]
        intercept_dict = {
            "feature": "Intercept",
            "coefficient": intercept["coefficient"],
            "conf_int_lower": intercept["conf_int_lower"],
            "conf_int_upper": intercept["conf_int_upper"],
            "p_value": intercept["p_value"]
        }
        features_to_plot = [intercept_dict] + features_to_plot

        # Sort features by name for consistent ordering (but keep intercept first)
        intercept_item = features_to_plot[0]
        other_features = sorted(features_to_plot[1:], key=lambda x: x["feature"])
        features_to_plot = [intercept_item] + other_features

        if len(features_to_plot) <= 1:  # Only intercept
            continue

        # Prepare data for plotting
        feature_names = [f["feature"] for f in features_to_plot]
        coefficients = [f["coefficient"] for f in features_to_plot]
        conf_lower = [f["conf_int_lower"] for f in features_to_plot]
        conf_upper = [f["conf_int_upper"] for f in features_to_plot]
        p_values = [f["p_value"] for f in features_to_plot]

        # Create coefficient plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create error bars (confidence intervals)
        y_pos = range(len(feature_names))
        errors = [
            [c - l for c, l in zip(coefficients, conf_lower)],
            [u - c for c, u in zip(coefficients, conf_upper)],
        ]

        # Create horizontal error bar plot
        ax.errorbar(
            coefficients,
            y_pos,
            xerr=errors,
            fmt="o",
            capsize=5,
            capthick=2,
            elinewidth=2,
            markersize=8,
        )

        # Color the markers by significance and highlight intercept
        for i, (coef, p_val, feature_name) in enumerate(zip(coefficients, p_values, feature_names)):
            # Special styling for intercept
            if feature_name == "Intercept":
                marker_style = "s"  # square marker
                marker_size = 120
                edge_color = "black"
                edge_width = 2
            else:
                marker_style = "o"  # circle marker
                marker_size = 100
                edge_color = None
                edge_width = 0
            
            # Color by significance
            if p_val < 0.001:
                color = "red"
            elif p_val < 0.01:
                color = "orange"
            elif p_val < 0.05:
                color = "blue"
            else:
                color = "gray"
            
            ax.scatter(coef, i, c=color, s=marker_size, zorder=5, 
                      marker=marker_style, edgecolors=edge_color, linewidth=edge_width)

        # Add vertical line at zero
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.7)

        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Coefficient Value")
        ax.set_ylabel("Features")
        ax.set_title(f"Regression Coefficients: {outcome}")
        ax.grid(True, alpha=0.3)

        # Add legend for significance levels and intercept
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [
            Patch(facecolor="red", label="p < 0.001"),
            Patch(facecolor="orange", label="p < 0.01"),
            Patch(facecolor="blue", label="p < 0.05"),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                   markeredgecolor='black', markersize=10, label="Intercept"),
        ]
        ax.legend(handles=legend_elements, loc="best")

        plt.tight_layout()

        plots.append(
            {
                "outcome": outcome,
                "figure": fig,
                "r2": result["model_performance"]["r2"],
                "n_features": len(features_to_plot),
            }
        )

    return plots


def create_feature_importance_plot(results):
    """Create overall feature importance plot across all models."""
    logger.info("Creating feature importance plot...")

    # Aggregate feature importance across models
    feature_importance = {}

    for result in results["regression_results"]:
        for feature in result["coefficients"]["features"]:
            if feature["significant"]:
                feat_name = feature["feature"]
                abs_t_value = abs(feature["t_value"])

                if feat_name not in feature_importance:
                    feature_importance[feat_name] = []
                feature_importance[feat_name].append(abs_t_value)

    # Calculate average importance
    avg_importance = {
        feat: np.mean(t_vals) for feat, t_vals in feature_importance.items()
    }

    # Get top 20 features
    top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:20]

    if not top_features:
        return None

    feature_names, importance_scores = zip(*top_features)

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = range(len(feature_names))
    bars = ax.barh(
        y_pos,
        importance_scores,
        color=plt.cm.viridis(np.linspace(0, 1, len(importance_scores))),
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Average |t-statistic|")
    ax.set_ylabel("Features")
    ax.set_title("Top 20 Most Important Features (Average |t-statistic| Across Models)")
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, importance_scores)):
        ax.text(
            score + max(importance_scores) * 0.01,
            i,
            f"{score:.2f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    return fig


def create_model_comparison_plots(data, results):
    """Create plots comparing outcomes across model families."""
    logger.info("Creating model comparison plots...")

    plots = []

    # Get model family columns
    model_cols = [
        col
        for col in data.columns
        if col.startswith("model_family_") and not col.endswith("_nan")
    ]

    # Key outcomes to analyze
    key_outcomes = [
        "proportion_left_leaning",
        "proportion_right_leaning",
        "proportion_high_quality",
        "proportion_news",
        "num_citations",
    ]

    for outcome in key_outcomes:
        if outcome not in data.columns:
            continue

        # Prepare data for plotting
        model_names = []
        outcome_values = []

        for model_col in model_cols:
            model_name = model_col.replace("model_family_", "")
            model_mask = data[model_col] == 1

            if model_mask.sum() < 50:  # Skip if too few samples
                continue

            values = data[model_mask][outcome].values
            model_names.extend([model_name] * len(values))
            outcome_values.extend(values)

        if not model_names:
            continue

        # Create box plot using matplotlib/seaborn
        fig, ax = plt.subplots(figsize=(10, 6))

        # Convert to DataFrame for seaborn
        plot_data = pd.DataFrame({"Model": model_names, "Value": outcome_values})

        # Create box plot
        sns.boxplot(data=plot_data, x="Model", y="Value", ax=ax)

        ax.set_title(f"{outcome.replace('_', ' ').title()} by Model Family")
        ax.set_xlabel("Model Family")
        ax.set_ylabel(outcome.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        plots.append({"outcome": outcome, "figure": fig})

    return plots


def create_multicollinearity_plots(results):
    """Create multicollinearity diagnostic plots."""
    logger.info("Creating multicollinearity diagnostic plots...")

    plots = []

    for result in results["regression_results"]:
        outcome = result["outcome"]
        multicollinearity = result["diagnostics"].get("multicollinearity", {})
        vif_data = multicollinearity.get("vif_data", [])

        if not vif_data:
            continue

        # Filter out NaN values and sort by VIF
        valid_vif = [vif for vif in vif_data if not np.isnan(vif["vif"])]
        if not valid_vif:
            continue

        # Sort by VIF value and take top 20
        valid_vif.sort(key=lambda x: x["vif"], reverse=True)
        top_vif = valid_vif[:20]

        # Create VIF plot
        feature_names = [vif["feature"] for vif in top_vif]
        vif_values = [vif["vif"] for vif in top_vif]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Color bars by VIF level
        colors = [
            "red" if vif >= 10 else "orange" if vif >= 5 else "blue"
            for vif in vif_values
        ]

        y_pos = range(len(feature_names))
        bars = ax.barh(y_pos, vif_values, color=colors)

        # Add reference lines
        ax.axvline(
            x=5, color="orange", linestyle="--", alpha=0.7, label="VIF = 5 (moderate)"
        )
        ax.axvline(
            x=10, color="red", linestyle="--", alpha=0.7, label="VIF = 10 (high)"
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Variance Inflation Factor (VIF)")
        ax.set_ylabel("Features")
        ax.set_title(f"Multicollinearity Diagnosis (VIF): {outcome}")
        ax.legend()
        ax.grid(True, axis="x", alpha=0.3)

        # Add value labels on bars
        for i, (bar, vif) in enumerate(zip(bars, vif_values)):
            ax.text(
                vif + max(vif_values) * 0.01, i, f"{vif:.1f}", va="center", fontsize=8
            )

        plt.tight_layout()

        plots.append(
            {
                "outcome": outcome,
                "figure": fig,
                "high_vif_count": multicollinearity.get("high_vif_count", 0),
                "mean_vif": multicollinearity.get("mean_vif", 0),
            }
        )

    return plots


def create_performance_summary_table(results):
    """Create model performance summary table."""
    logger.info("Creating performance summary table...")

    performance_data = []

    for result in results["regression_results"]:
        perf = result["model_performance"]
        diag = result["diagnostics"]
        multicollinearity = diag.get("multicollinearity", {})

        # Get intercept information
        intercept = result["coefficients"]["intercept"]
        intercept_str = f"{intercept['coefficient']:.4f}"
        if intercept['p_value'] < 0.001:
            intercept_str += "***"
        elif intercept['p_value'] < 0.01:
            intercept_str += "**"
        elif intercept['p_value'] < 0.05:
            intercept_str += "*"

        performance_data.append(
            {
                "Outcome": result["outcome"],
                "Intercept": intercept_str,
                "R²": f"{perf['r2']:.4f}",
                "Adj. R²": f"{perf['adj_r2']:.4f}",
                "F-statistic": f"{perf['f_statistic']:.2f}",
                "F p-value": f"{perf['f_pvalue']:.4f}",
                "AIC": f"{perf['aic']:.1f}",
                "BIC": f"{perf['bic']:.1f}",
                "RMSE": f"{perf['rmse']:.4f}",
                "N Significant Features": diag["n_significant_features"],
                "High VIF Features": multicollinearity.get("high_vif_count", 0),
                "Mean VIF": f"{multicollinearity.get('mean_vif', 0):.2f}",
                "Max VIF": f"{multicollinearity.get('max_vif', 0):.2f}",
                "Sample Size": f"{result['n_samples']:,}",
            }
        )

    return pd.DataFrame(performance_data)


def generate_html_report(data, results, output_path):
    """Generate comprehensive HTML report."""
    logger.info("Generating HTML report...")

    # Create visualizations
    summary_table = create_summary_statistics_table(data)
    correlation_fig = create_correlation_matrix(data)
    coefficient_plots = create_regression_coefficient_plots(results)
    importance_fig = create_feature_importance_plot(results)
    model_comparison_plots = create_model_comparison_plots(data, results)
    multicollinearity_plots = create_multicollinearity_plots(results)
    performance_table = create_performance_summary_table(results)

    # Start HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Question Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2E4057; border-bottom: 3px solid #048A81; }}
            h2 {{ color: #048A81; border-bottom: 1px solid #048A81; }}
            h3 {{ color: #54C6EB; }}
            .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #048A81; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .plot-container {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Question Analysis Report</h1>
        <p><strong>Generated:</strong> {results["metadata"]["timestamp"]}</p>

        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metric">
                <strong>Dataset Size:</strong><br>
                {results["summary_statistics"]["dataset"]["n_observations"]:,} observations
            </div>
            <div class="metric">
                <strong>Features:</strong><br>
                {results["summary_statistics"]["dataset"]["n_features"]} total
            </div>
            <div class="metric">
                <strong>Models Analyzed:</strong><br>
                {results["metadata"]["n_outcomes_analyzed"]} outcomes
            </div>
            <div class="metric">
                <strong>Best R²:</strong><br>
                {max(r["model_performance"]["r2"] for r in results["regression_results"]):.3f}
            </div>
        </div>
    """

    # Add performance summary table
    html_content += f"""
        <h2>Model Performance Summary</h2>
        <div class="plot-container">
            {performance_table.to_html(index=False, table_id="performance-table")}
        </div>
    """

    # Add correlation matrix
    corr_img = fig_to_base64(correlation_fig)
    html_content += f"""
        <h2>Correlation Matrix</h2>
        <div class="plot-container">
            <img src="{corr_img}" style="max-width: 100%; height: auto;">
        </div>
    """

    # Add feature importance plot
    if importance_fig:
        importance_img = fig_to_base64(importance_fig)
        html_content += f"""
            <h2>Feature Importance</h2>
            <div class="plot-container">
                <img src="{importance_img}" style="max-width: 100%; height: auto;">
            </div>
        """

    # Add regression coefficient plots
    html_content += "<h2>Regression Coefficients by Outcome</h2>"
    for i, plot_data in enumerate(coefficient_plots):
        coeff_img = fig_to_base64(plot_data["figure"])
        html_content += f"""
            <h3>{plot_data["outcome"]} (R² = {plot_data["r2"]:.3f}, {plot_data["n_features"]} features)</h3>
            <div class="plot-container">
                <img src="{coeff_img}" style="max-width: 100%; height: auto;">
            </div>
        """

    # Add model comparison plots
    if model_comparison_plots:
        html_content += "<h2>Model Family Comparisons</h2>"
        for i, plot_data in enumerate(model_comparison_plots):
            model_img = fig_to_base64(plot_data["figure"])
            html_content += f"""
                <h3>{plot_data["outcome"]}</h3>
                <div class="plot-container">
                    <img src="{model_img}" style="max-width: 100%; height: auto;">
                </div>
            """

    # Add multicollinearity diagnostic plots
    if multicollinearity_plots:
        html_content += "<h2>Multicollinearity Diagnostics</h2>"
        html_content += """
        <div class="summary">
            <p><strong>Interpretation:</strong> Variance Inflation Factor (VIF) measures multicollinearity.</p>
            <ul>
                <li><strong>VIF < 5:</strong> Low multicollinearity (acceptable)</li>
                <li><strong>5 ≤ VIF < 10:</strong> Moderate multicollinearity (concerning)</li>
                <li><strong>VIF ≥ 10:</strong> High multicollinearity (problematic)</li>
            </ul>
        </div>
        """
        for i, plot_data in enumerate(multicollinearity_plots):
            vif_img = fig_to_base64(plot_data["figure"])
            html_content += f"""
                <h3>{plot_data["outcome"]} (High VIF: {plot_data["high_vif_count"]}, Mean VIF: {plot_data["mean_vif"]:.2f})</h3>
                <div class="plot-container">
                    <img src="{vif_img}" style="max-width: 100%; height: auto;">
                </div>
            """

    # Add summary statistics table
    html_content += f"""
        <h2>Summary Statistics</h2>
        <div class="plot-container">
            {summary_table.to_html(index=False, table_id="summary-table")}
        </div>

        <h2>Technical Details</h2>
        <div class="summary">
            <p><strong>Regression Method:</strong> {results["metadata"]["regression_method"]}</p>
            <p><strong>PCA Precomputed:</strong> {results["metadata"]["feature_engineering"]["pca_precomputed"]}</p>
            <p><strong>PCA Used:</strong> {results["metadata"]["feature_engineering"]["pca_used"]}</p>
            <p><strong>Total Features:</strong> {results["metadata"]["feature_engineering"]["total_features"]}</p>
        </div>

    </body>
    </html>
    """

    # Save HTML report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML report saved to {output_path}")


def main():
    """Main function for report generation."""
    try:
        # Get input/output paths from Snakemake
        data_path = snakemake.input.data  # type: ignore
        results_path = snakemake.input.results  # type: ignore
        output_path = snakemake.output[0]  # type: ignore

    except NameError:
        # Fallback for running outside Snakemake
        logger.info("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent

        data_path = (
            base_dir / "data/intermediate/question_analysis/cleaned_features.parquet"
        )
        results_path = (
            base_dir / "data/output/question_analysis/regression_results.json"
        )
        output_path = (
            base_dir / "data/output/question_analysis/question_analysis_report.html"
        )

    # Load data
    data, results = load_data(data_path, results_path)

    # Generate report
    generate_html_report(data, results, output_path)

    logger.info("✅ Report generation completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
