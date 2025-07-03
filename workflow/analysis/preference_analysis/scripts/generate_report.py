#!/usr/bin/env python3
"""
Comprehensive Report Generator for Preference Analysis.

This script generates HTML reports and visualizations from all analysis components:
- Bradley-Terry model ratings
- Individual feature effects
- Citation style effects with flexible models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
from pathlib import Path
from datetime import datetime
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Set style for professional plots
plt.style.use("default")
sns.set_palette("husl")


def load_all_results(input_files):
    """Load all analysis results from JSON files."""
    print("Loading analysis results...")

    results = {}

    # Load Bradley-Terry ratings
    if input_files["bt_ratings_results"].exists():
        with open(input_files["bt_ratings_results"], "r") as f:
            results["bt_ratings"] = json.load(f)
        print("  Loaded Bradley-Terry ratings")

    # Load individual effects
    if input_files["individual_effects_results"].exists():
        with open(input_files["individual_effects_results"], "r") as f:
            results["individual_effects"] = json.load(f)
        print("  Loaded individual effects")

    # Load citation style effects
    if input_files["citation_style_effects_results"].exists():
        with open(input_files["citation_style_effects_results"], "r") as f:
            results["citation_style_effects"] = json.load(f)
        print("  Loaded citation style effects")

    return results


def load_coefficients_data(input_files):
    """Load all coefficient CSV files for comprehensive analysis."""
    print("Loading coefficient data...")

    coefficients = {}

    # Load Bradley-Terry coefficients
    if input_files["bt_ratings_coefficients"].exists():
        coefficients["bt_ratings"] = pd.read_csv(input_files["bt_ratings_coefficients"])
        print(f"  Loaded BT coefficients: {len(coefficients['bt_ratings'])} rows")

    # Load individual effects coefficients
    if input_files["individual_effects_coefficients"].exists():
        coefficients["individual_effects"] = pd.read_csv(
            input_files["individual_effects_coefficients"]
        )
        print(
            f"  Loaded individual coefficients: {len(coefficients['individual_effects'])} rows"
        )

    # Load citation style effects coefficients
    if input_files["citation_style_effects_coefficients"].exists():
        coefficients["citation_style_effects"] = pd.read_csv(
            input_files["citation_style_effects_coefficients"]
        )
        print(
            f"  Loaded citation style coefficients: {len(coefficients['citation_style_effects'])} rows"
        )

    return coefficients


def create_individual_effects_plot(coefficients, output_path):
    """Create individual effects confidence intervals visualization."""
    print("Creating individual effects confidence intervals plot...")

    plt.figure(figsize=(12, 10))

    # Get individual effects data only
    plot_data = []

    if "individual_effects" in coefficients:
        for _, row in coefficients["individual_effects"].iterrows():
            if row["analysis_type"] == "individual_feature_effect":
                plot_data.append(
                    {
                        "feature": row["feature"],
                        "coefficient": row["coefficient"],
                        "ci_lower": row["ci_lower"],
                        "ci_upper": row["ci_upper"],
                        "significant": row["significant"],
                    }
                )

    if not plot_data:
        print("  No data available for individual effects plot")
        plt.text(
            0.5,
            0.5,
            "No individual effects data available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=14,
        )
        plt.title("Individual Feature Effects - No Data Available")
    else:
        df = pd.DataFrame(plot_data)

        # Sort by coefficient value
        df = df.sort_values("coefficient", ascending=True)

        # Create plot
        y_pos = np.arange(len(df))

        # Plot confidence intervals as error bars
        plt.errorbar(
            df["coefficient"],
            y_pos,
            xerr=[
                df["coefficient"] - df["ci_lower"],
                df["ci_upper"] - df["coefficient"],
            ],
            fmt="o",
            capsize=5,
            capthick=2,
            color="black",
            alpha=0.7,
        )

        # Color code the points
        for i, (coeff, sig) in enumerate(zip(df["coefficient"], df["significant"])):
            color = "red" if sig else "blue"
            plt.scatter(coeff, i, color=color, s=100, alpha=0.8, zorder=3)

        plt.yticks(y_pos, df["feature"], fontsize=10)
        plt.xlabel("Effect Size (Coefficient)", fontsize=12)
        plt.title(
            "Individual Feature Effects - 95% Confidence Intervals\n(Red = Significant, Blue = Not Significant)",
            fontsize=14,
        )
        plt.grid(axis="x", alpha=0.3)

        # Add vertical line at zero
        plt.axvline(x=0, color="black", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Individual effects confidence intervals plot saved to: {output_path}")


def create_citation_style_effects_plot(coefficients, output_path):
    """Create citation style effects confidence intervals visualization."""
    print("Creating citation style effects confidence intervals plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Get citation style effects data
    if "citation_style_effects" in coefficients:
        # Group by model specification
        model_groups = coefficients["citation_style_effects"].groupby(
            "model_specification"
        )

        plot_idx = 0
        for model_name, model_data in model_groups:
            if plot_idx >= 4:  # Limit to 4 subplots
                break

            ax = axes[plot_idx]

            # Prepare data for this model
            plot_data = []
            for _, row in model_data.iterrows():
                plot_data.append(
                    {
                        "feature": row["feature"],
                        "coefficient": row["coefficient"],
                        "ci_lower": row["ci_lower"],
                        "ci_upper": row["ci_upper"],
                        "significant": row["significant"],
                    }
                )

            if plot_data:
                df = pd.DataFrame(plot_data)
                df = df.iloc[::-1]

                y_pos = np.arange(len(df))

                # Plot confidence intervals as error bars
                ax.errorbar(
                    df["coefficient"],
                    y_pos,
                    xerr=[
                        df["coefficient"] - df["ci_lower"],
                        df["ci_upper"] - df["coefficient"],
                    ],
                    fmt="o",
                    capsize=4,
                    capthick=1.5,
                    color="black",
                    alpha=0.7,
                )

                # Color code the points
                for i, (coeff, sig) in enumerate(
                    zip(df["coefficient"], df["significant"])
                ):
                    color = "red" if sig else "blue"
                    ax.scatter(coeff, i, color=color, s=80, alpha=0.8, zorder=3)

                ax.set_yticks(y_pos)
                ax.set_yticklabels(df["feature"], fontsize=9)
                ax.set_xlabel("Effect Size (Coefficient)", fontsize=10)
                ax.set_title(
                    f"{model_name.replace('_', ' ').title()}\nModel", fontsize=12
                )
                ax.grid(axis="x", alpha=0.3)
                ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_title(
                    f"{model_name.replace('_', ' ').title()}\nModel - No Data",
                    fontsize=12,
                )

            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)

    else:
        print("  No citation style effects data available")
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No citation style effects data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title("Citation Style Effects - No Data Available")

    plt.suptitle(
        "Citation Style Effects - 95% Confidence Intervals by Model\n(Red = Significant, Blue = Not Significant)",
        fontsize=16,
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Citation style effects confidence intervals plot saved to: {output_path}")


def create_model_comparison_plot(coefficients, output_path):
    """Create model comparison visualization."""
    print("Creating model comparison plot...")

    plt.figure(figsize=(12, 8))

    if "bt_ratings" in coefficients:
        bt_data = coefficients["bt_ratings"]
        if len(bt_data) > 0:
            # Sort by rating
            bt_data = bt_data.sort_values("coefficient", ascending=False)

            # Create horizontal bar plot
            y_pos = np.arange(len(bt_data))
            plt.barh(y_pos, bt_data["coefficient"], color="skyblue", alpha=0.7)
            plt.yticks(y_pos, bt_data["feature"], fontsize=10)
            plt.xlabel("Bradley-Terry Rating", fontsize=12)
            plt.title(
                "Model Performance Rankings\n(Bradley-Terry Ratings)", fontsize=14
            )
            plt.grid(axis="x", alpha=0.3)

            # Add value labels
            for i, rating in enumerate(bt_data["coefficient"]):
                plt.text(
                    rating + max(bt_data["coefficient"]) * 0.01,
                    i,
                    f"{rating:.0f}",
                    va="center",
                    fontsize=10,
                )
        else:
            plt.text(
                0.5,
                0.5,
                "No Bradley-Terry rating data available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                fontsize=14,
            )
            plt.title("Model Performance Rankings - No Data Available")
    else:
        plt.text(
            0.5,
            0.5,
            "No Bradley-Terry rating data available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=14,
        )
        plt.title("Model Performance Rankings - No Data Available")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Model comparison plot saved to: {output_path}")


def generate_html_report(results, coefficients, battle_data, output_path):
    """Generate comprehensive HTML report."""
    print("Generating HTML report...")

    # Load battle data for context
    battle_df = pd.read_parquet(battle_data) if battle_data.exists() else None

    # Generate report timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Preference Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            min-width: 150px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .significant {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .not-significant {{
            color: #95a5a6;
        }}
        .visualization {{
            text-align: center;
            margin: 20px 0;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .summary-box {{
            background-color: #e8f6f3;
            border-left: 4px solid #1abc9c;
            padding: 15px;
            margin: 15px 0;
        }}
        .error-box {{
            background-color: #fdf2e9;
            border-left: 4px solid #e67e22;
            padding: 15px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>News Citation Preference Analysis Report</h1>
        <p>Comprehensive analysis of how citation patterns affect user preferences in AI search responses</p>
        <p><strong>Generated:</strong> {timestamp}</p>
    </div>
"""

    # Add overview section
    html_content += generate_overview_section(results, battle_df)

    # Add Bradley-Terry ratings section
    html_content += generate_bt_ratings_section(results, coefficients)

    # Add individual effects section
    html_content += generate_individual_effects_section(results, coefficients)

    # Add citation style effects section
    html_content += generate_citation_style_effects_section(results, coefficients)

    # Add visualizations section
    html_content += generate_visualizations_section()

    # Add methodology section
    html_content += generate_methodology_section()

    # Close HTML
    html_content += """
</body>
</html>
"""

    # Save HTML report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  HTML report saved to: {output_path}")


def generate_overview_section(results, battle_df):
    """Generate overview section of the report."""
    n_battles = len(battle_df) if battle_df is not None else 0
    n_models = (
        len(set(battle_df["model_a"]) | set(battle_df["model_b"]))
        if battle_df is not None
        else 0
    )

    return f"""
    <div class="section">
        <h2>Analysis Overview</h2>
        <div class="metric">
            <div class="metric-value">{n_battles:,}</div>
            <div class="metric-label">Total Battles</div>
        </div>
        <div class="metric">
            <div class="metric-value">{n_models}</div>
            <div class="metric-label">AI Models</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(results)}</div>
            <div class="metric-label">Analysis Types</div>
        </div>

        <div class="summary-box">
            <h3>Key Findings Summary</h3>
            <p>This analysis examines how different citation patterns in AI search responses influence user preferences using Bradley-Terry models and contextual statistical analysis.</p>
        </div>
    </div>
"""


def generate_bt_ratings_section(results, coefficients):
    """Generate Bradley-Terry ratings section."""
    if "bt_ratings" not in results:
        return """
    <div class="section">
        <h2>Model Performance Rankings</h2>
        <div class="error-box">
            <p>Bradley-Terry ratings data not available.</p>
        </div>
    </div>
"""

    bt_results = results["bt_ratings"]["bradley_terry_ratings"]
    sorted_models = sorted(
        bt_results["model_ratings"].items(), key=lambda x: x[1], reverse=True
    )

    table_rows = ""
    for i, (model, rating) in enumerate(sorted_models):
        table_rows += f"""
        <tr>
            <td>{i + 1}</td>
            <td>{model.replace("_", " ").title()}</td>
            <td>{rating:.1f}</td>
        </tr>
"""

    return f"""
    <div class="section">
        <h2>Model Performance Rankings</h2>
        <p>Bradley-Terry ratings based on user preferences in news citation competitions.</p>

        <div class="metric">
            <div class="metric-value">{bt_results["n_battles"]:,}</div>
            <div class="metric-label">Battles Analyzed</div>
        </div>
        <div class="metric">
            <div class="metric-value">{bt_results["n_models"]}</div>
            <div class="metric-label">Models Compared</div>
        </div>
        <div class="metric">
            <div class="metric-value">{bt_results["log_likelihood"]:.1f}</div>
            <div class="metric-label">Log-Likelihood</div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Rating</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
"""


def generate_individual_effects_section(results, coefficients):
    """Generate individual effects section."""
    if "individual_effects" not in results:
        return """
    <div class="section">
        <h2>Individual Feature Effects</h2>
        <div class="error-box">
            <p>Individual feature effects data not available.</p>
        </div>
    </div>
"""

    individual_results = results["individual_effects"]["individual_feature_effects"]

    # Count significant effects
    significant_count = 0
    total_count = len(individual_results)

    table_rows = ""
    effects_list = []

    for feature, feature_results in individual_results.items():
        coeff = feature_results["coefficients"][feature]
        ci = feature_results["confidence_intervals"][feature]
        significant = not (ci["lower"] <= 0 <= ci["upper"])

        if significant:
            significant_count += 1

        effects_list.append((feature, coeff, ci, significant))

    # Sort by absolute effect size
    effects_list.sort(key=lambda x: abs(x[1]), reverse=True)

    for feature, coeff, ci, significant in effects_list:
        sig_class = "significant" if significant else "not-significant"
        sig_text = "Yes" if significant else "No"
        direction = "Positive" if coeff > 0 else "Negative" if coeff < 0 else "Neutral"

        table_rows += f"""
        <tr>
            <td>{feature.replace("_", " ").title()}</td>
            <td class="{sig_class}">{coeff:.4f}</td>
            <td>[{ci["lower"]:.4f}, {ci["upper"]:.4f}]</td>
            <td class="{sig_class}">{sig_text}</td>
            <td>{direction}</td>
        </tr>
"""

    return f"""
    <div class="section">
        <h2>Individual Feature Effects</h2>
        <p>Analysis of how individual citation features affect user preferences when considered in isolation.</p>

        <div class="metric">
            <div class="metric-value">{significant_count}</div>
            <div class="metric-label">Significant Features</div>
        </div>
        <div class="metric">
            <div class="metric-value">{total_count}</div>
            <div class="metric-label">Total Features</div>
        </div>
        <div class="metric">
            <div class="metric-value">{significant_count / total_count:.1%}</div>
            <div class="metric-label">Significance Rate</div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Coefficient</th>
                    <th>95% CI</th>
                    <th>Significant</th>
                    <th>Direction</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
"""


def generate_citation_style_effects_section(results, coefficients):
    """Generate citation style effects section."""
    if "citation_style_effects" not in results:
        return """
    <div class="section">
        <h2>Citation Style Effects</h2>
        <div class="error-box">
            <p>Citation style effects data not available.</p>
        </div>
    </div>
"""

    citation_results = results["citation_style_effects"]

    # If there are multiple models, create a section for each
    table_sections = ""

    for model_name, model_results in citation_results.items():
        if isinstance(model_results, dict) and "features" in model_results:
            significant_count = 0
            table_rows = ""

            for feature in model_results["features"]:
                coeff = model_results["coefficients"][feature]
                ci = model_results["confidence_intervals"][feature]
                significant = not (ci["lower"] <= 0 <= ci["upper"])

                if significant:
                    significant_count += 1

                sig_class = "significant" if significant else "not-significant"
                sig_text = "Yes" if significant else "No"
                direction = (
                    "Positive" if coeff > 0 else "Negative" if coeff < 0 else "Neutral"
                )

                table_rows += f"""
                <tr>
                    <td>{feature.replace("_", " ").title()}</td>
                    <td class="{sig_class}">{coeff:.4f}</td>
                    <td>[{ci["lower"]:.4f}, {ci["upper"]:.4f}]</td>
                    <td class="{sig_class}">{sig_text}</td>
                    <td>{direction}</td>
                </tr>
"""

            table_sections += f"""
            <h3>{model_name.replace("_", " ").title()} Model</h3>
            <p>Description: {model_results.get("model_description", "Multi-feature contextual model")}</p>

            <div class="metric">
                <div class="metric-value">{significant_count}</div>
                <div class="metric-label">Significant Features</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(model_results["features"])}</div>
                <div class="metric-label">Total Features</div>
            </div>
            <div class="metric">
                <div class="metric-value">{model_results["log_likelihood"]:.1f}</div>
                <div class="metric-label">Log-Likelihood</div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Coefficient</th>
                        <th>95% CI</th>
                        <th>Significant</th>
                        <th>Direction</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
"""

    return f"""
    <div class="section">
        <h2>Citation Style Effects</h2>
        <p>Analysis of citation features in multi-feature contextual models to understand combined effects.</p>

        {table_sections}
    </div>
"""


def generate_visualizations_section():
    """Generate visualizations section."""
    return """
    <div class="section">
        <h2>Visualizations</h2>
        <p>Visual representations of the analysis results showing effect sizes, confidence intervals, statistical significance, and model comparisons.</p>

        <div class="visualization">
            <h3>Individual Feature Effects</h3>
            <img src="visualizations/individual_effects.png" alt="Individual Effects Confidence Intervals">
            <p>95% confidence intervals for each citation feature when analyzed in isolation. Red points indicate statistically significant effects.</p>
        </div>

        <div class="visualization">
            <h3>Citation Style Effects</h3>
            <img src="visualizations/citation_style_effects.png" alt="Citation Style Effects Confidence Intervals">
            <p>95% confidence intervals showing how features perform across different multi-feature model specifications. Red points indicate statistically significant effects.</p>
        </div>


        <div class="visualization">
            <h3>Model Comparison</h3>
            <img src="visualizations/model_comparison.png" alt="Model Comparison Plot">
        </div>
    </div>
"""


def generate_methodology_section():
    """Generate methodology section."""
    return """
    <div class="section">
        <h2>Methodology</h2>
        <h3>Bradley-Terry Model</h3>
        <p>Used to rank AI models based on user preferences in head-to-head comparisons. The model estimates the probability that model A beats model B based on their relative strengths.</p>

        <h3>Contextual Bradley-Terry Model</h3>
        <p>Extension that incorporates citation features to understand how different citation patterns influence user preferences beyond base model performance.</p>

        <h3>Bootstrap Confidence Intervals</h3>
        <p>Generated using 1000 bootstrap samples to provide robust uncertainty estimates for all effect sizes and model comparisons.</p>

        <h3>Statistical Significance</h3>
        <p>Effects are considered significant at the 95% confidence level if their confidence interval does not include zero.</p>

        <h3>Data Processing</h3>
        <p>Analysis focuses on battles where both responses cite news sources, ensuring fair comparison of citation strategies.</p>
    </div>
"""


def main():
    """Generate comprehensive preference analysis report."""
    try:
        # Get input/output paths from Snakemake or command line
        if "snakemake" in globals():
            input_files = {
                "bt_ratings_results": Path(snakemake.input.bt_ratings_results),
                "individual_effects_results": Path(
                    snakemake.input.individual_effects_results
                ),
                "citation_style_effects_results": Path(
                    snakemake.input.citation_style_effects_results
                ),
                "bt_ratings_coefficients": Path(
                    snakemake.input.bt_ratings_coefficients
                ),
                "individual_effects_coefficients": Path(
                    snakemake.input.individual_effects_coefficients
                ),
                "citation_style_effects_coefficients": Path(
                    snakemake.input.citation_style_effects_coefficients
                ),
                "battle_data": Path(snakemake.input.battle_data),
            }
            output_files = {
                "report": Path(snakemake.output.report),
                "individual_effects": Path(snakemake.output.individual_effects),
                "citation_style_effects": Path(snakemake.output.citation_style_effects),
                "model_comparison": Path(snakemake.output.model_comparison),
            }
        else:
            print("Usage: Run via Snakemake workflow")
            return 1

        print("=" * 80)
        print("GENERATING PREFERENCE ANALYSIS REPORT")
        print("=" * 80)

        # Ensure output directories exist
        output_files["report"].parent.mkdir(parents=True, exist_ok=True)
        output_files["individual_effects"].parent.mkdir(parents=True, exist_ok=True)

        # Load all analysis results
        results = load_all_results(input_files)
        coefficients = load_coefficients_data(input_files)

        # Generate visualizations
        create_individual_effects_plot(coefficients, output_files["individual_effects"])
        create_citation_style_effects_plot(
            coefficients, output_files["citation_style_effects"]
        )
        create_model_comparison_plot(coefficients, output_files["model_comparison"])

        # Generate comprehensive HTML report
        generate_html_report(
            results, coefficients, input_files["battle_data"], output_files["report"]
        )

        print("\nReport generation completed successfully!")
        print(f"HTML report: {output_files['report']}")
        print(f"Visualizations: {output_files['individual_effects'].parent}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
