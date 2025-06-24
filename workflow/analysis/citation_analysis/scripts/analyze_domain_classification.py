#!/usr/bin/env python3
"""
Domain Classification Analysis Script.

This script analyzes citation patterns across different domain classifications,
examining how AI models cite different source types and identifying specialization patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_integrated_data(data_path):
    """Load the integrated citation dataset."""
    print(f"Loading integrated citations from {data_path}")
    data = pd.read_parquet(data_path)
    print(f"Loaded {len(data):,} citations with {len(data.columns)} columns")
    return data


def analyze_overall_domain_distribution(data):
    """Analyze overall distribution of citations across domain types."""
    print("\n=== OVERALL DOMAIN DISTRIBUTION ANALYSIS ===")

    # Overall distribution
    domain_counts = data["domain_classification"].value_counts()
    domain_pcts = data["domain_classification"].value_counts(normalize=True) * 100

    print("Citation distribution by domain type:")
    for domain_type in domain_counts.index:
        count = domain_counts[domain_type]
        pct = domain_pcts[domain_type]
        print(f"  {domain_type}: {count:,} citations ({pct:.1f}%)")

    # Top domains within each classification
    print("\nTop 3 domains within each classification:")
    for domain_type in domain_counts.index[:8]:  # Top 8 domain types
        top_domains = (
            data[data["domain_classification"] == domain_type]
            .groupby("domain")["citation_id"]
            .count()
            .sort_values(ascending=False)
            .head(3)
        )

        print(f"\n  {domain_type}:")
        for domain, count in top_domains.items():
            print(f"    {domain}: {count:,} citations")

    return {"domain_counts": domain_counts, "domain_percentages": domain_pcts}


def analyze_model_domain_preferences(data):
    """Analyze domain citation preferences by AI model."""
    print("\n=== MODEL DOMAIN PREFERENCES ANALYSIS ===")

    if "model_name_llm" not in data.columns:
        print("Model information not available")
        return {}

    # Model-domain cross-tabulation
    model_domain_crosstab = pd.crosstab(
        data["model_name_llm"], data["domain_classification"]
    )

    # Convert to percentages within each model
    model_domain_pcts = (
        model_domain_crosstab.div(model_domain_crosstab.sum(axis=1), axis=0) * 100
    )

    print("Domain preferences by model (% of citations):")
    print(model_domain_pcts.round(1))

    # Identify models with highest preference for each domain type
    print("\nModels with highest preference for each domain type:")
    for domain_type in model_domain_pcts.columns:
        best_model = model_domain_pcts[domain_type].idxmax()
        best_pct = model_domain_pcts.loc[best_model, domain_type]
        print(f"  {domain_type}: {best_model} ({best_pct:.1f}%)")

    # Statistical significance tests could be added here

    return {
        "model_domain_crosstab": model_domain_crosstab,
        "model_domain_percentages": model_domain_pcts,
    }


def analyze_intent_domain_patterns(data):
    """Analyze domain citation patterns by query intent."""
    print("\n=== INTENT-DOMAIN PATTERNS ANALYSIS ===")

    if "primary_intent" not in data.columns:
        print("Intent information not available")
        return {}

    # Intent-domain cross-tabulation
    intent_domain_crosstab = pd.crosstab(
        data["primary_intent"], data["domain_classification"]
    )

    # Convert to percentages within each intent
    intent_domain_pcts = (
        intent_domain_crosstab.div(intent_domain_crosstab.sum(axis=1), axis=0) * 100
    )

    print("Domain preferences by query intent (% of citations):")
    print(intent_domain_pcts.round(1))

    # Identify specialization patterns
    print("\nSpecialization patterns (intent -> domain preferences):")
    for intent in intent_domain_pcts.index:
        top_domains = intent_domain_pcts.loc[intent].nlargest(3)
        print(f"  {intent}:")
        for domain_type, pct in top_domains.items():
            print(f"    {domain_type}: {pct:.1f}%")

    return {
        "intent_domain_crosstab": intent_domain_crosstab,
        "intent_domain_percentages": intent_domain_pcts,
    }


def analyze_model_comparison(data):
    """Compare domain citation patterns between model sides A and B."""
    print("\n=== MODEL A vs B DOMAIN COMPARISON ===")

    if "model_side" not in data.columns:
        print("Model side information not available")
        return {}

    # Compare domain distributions between model sides
    side_domain_crosstab = pd.crosstab(
        data["model_side"], data["domain_classification"]
    )

    side_domain_pcts = (
        side_domain_crosstab.div(side_domain_crosstab.sum(axis=1), axis=0) * 100
    )

    print("Domain distribution by model side:")
    print(side_domain_pcts.round(1))

    # Calculate differences
    if len(side_domain_pcts) >= 2:
        side_a_pcts = (
            side_domain_pcts.loc["a"] if "a" in side_domain_pcts.index else pd.Series()
        )
        side_b_pcts = (
            side_domain_pcts.loc["b"] if "b" in side_domain_pcts.index else pd.Series()
        )

        if not side_a_pcts.empty and not side_b_pcts.empty:
            differences = side_b_pcts - side_a_pcts

            print(f"\nDifferences (Model B - Model A):")
            for domain_type, diff in differences.items():
                direction = "↑" if diff > 0 else "↓" if diff < 0 else "="
                print(f"  {domain_type}: {diff:+.1f}pp {direction}")

    return {
        "side_domain_crosstab": side_domain_crosstab,
        "side_domain_percentages": side_domain_pcts,
    }


def analyze_winner_domain_patterns(data):
    """Analyze domain citation patterns for winning vs losing models."""
    print("\n=== WINNER vs LOSER DOMAIN PATTERNS ===")

    if "model_won" not in data.columns or "model_lost" not in data.columns:
        print("Winner information not available")
        return {}

    # Filter for clear wins/losses (exclude ties)
    winners = data[data["model_won"] == True]
    losers = data[data["model_lost"] == True]

    print(f"Analyzing {len(winners):,} citations from winning models")
    print(f"Analyzing {len(losers):,} citations from losing models")

    if len(winners) == 0 or len(losers) == 0:
        print("Insufficient data for winner/loser comparison")
        return {}

    # Domain distributions
    winner_domain_pcts = (
        winners["domain_classification"].value_counts(normalize=True) * 100
    )
    loser_domain_pcts = (
        losers["domain_classification"].value_counts(normalize=True) * 100
    )

    print("\nDomain preferences:")
    print("Winners vs Losers (% of citations)")

    comparison_df = pd.DataFrame(
        {"Winners": winner_domain_pcts, "Losers": loser_domain_pcts}
    ).fillna(0)

    comparison_df["Difference"] = comparison_df["Winners"] - comparison_df["Losers"]
    comparison_df = comparison_df.sort_values("Difference", ascending=False)

    print(comparison_df.round(1))

    return {
        "winner_domain_percentages": winner_domain_pcts,
        "loser_domain_percentages": loser_domain_pcts,
        "comparison_df": comparison_df,
    }


def create_visualizations(analysis_results, output_dir):
    """Create visualizations for domain classification analysis."""
    print("\n=== CREATING VISUALIZATIONS ===")

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("Set2")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Domain Classification Analysis Dashboard", fontsize=16)

    # 1. Overall domain distribution (pie chart)
    if "domain_counts" in analysis_results:
        domain_counts = analysis_results["domain_counts"].head(
            8
        )  # Top 8 for readability
        axes[0, 0].pie(
            domain_counts.values, labels=domain_counts.index, autopct="%1.1f%%"
        )
        axes[0, 0].set_title("Overall Domain Distribution")

    # 2. Model domain preferences (bar chart)
    if "model_domain_percentages" in analysis_results:
        model_domain_pcts = analysis_results["model_domain_percentages"]
        # Show top 5 models and top 6 domain types
        top_models = model_domain_pcts.sum(axis=1).nlargest(5).index
        top_domains = model_domain_pcts.sum(axis=0).nlargest(6).index

        plot_data = model_domain_pcts.loc[top_models, top_domains]
        plot_data.plot(kind="bar", stacked=True, ax=axes[0, 1])
        axes[0, 1].set_title("Model Domain Preferences (Top 5 Models)")
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[0, 1].tick_params(axis="x", rotation=45)

    # 3. Intent-domain heatmap
    if "intent_domain_percentages" in analysis_results:
        intent_domain_pcts = analysis_results["intent_domain_percentages"]
        top_domains = intent_domain_pcts.sum(axis=0).nlargest(6).index

        sns.heatmap(
            intent_domain_pcts[top_domains],
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            ax=axes[0, 2],
        )
        axes[0, 2].set_title("Intent-Domain Patterns")
        axes[0, 2].tick_params(axis="x", rotation=45)
        axes[0, 2].tick_params(axis="y", rotation=0)

    # 4. Model A vs B comparison
    if "side_domain_percentages" in analysis_results:
        side_domain_pcts = analysis_results["side_domain_percentages"]

        if len(side_domain_pcts) >= 2:
            side_domain_pcts.plot(kind="bar", ax=axes[1, 0])
            axes[1, 0].set_title("Model A vs B Domain Distribution")
            axes[1, 0].tick_params(axis="x", rotation=45)
            axes[1, 0].legend()

    # 5. Winner vs loser comparison
    if "comparison_df" in analysis_results:
        comparison_df = analysis_results["comparison_df"].head(8)

        colors = ["green" if x > 0 else "red" for x in comparison_df["Difference"]]
        axes[1, 1].bar(
            range(len(comparison_df)), comparison_df["Difference"], color=colors
        )
        axes[1, 1].set_title("Winner vs Loser Domain Preferences")
        axes[1, 1].set_xticks(range(len(comparison_df)))
        axes[1, 1].set_xticklabels(comparison_df.index, rotation=45)
        axes[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # 6. Summary statistics
    if "domain_counts" in analysis_results:
        domain_counts = analysis_results["domain_counts"]

        axes[1, 2].bar(range(len(domain_counts.head(8))), domain_counts.head(8).values)
        axes[1, 2].set_title("Top Domain Types by Citation Count")
        axes[1, 2].set_xticks(range(len(domain_counts.head(8))))
        axes[1, 2].set_xticklabels(domain_counts.head(8).index, rotation=45)
        axes[1, 2].set_ylabel("Citation Count")

    # Adjust layout and save
    plt.tight_layout()
    output_path = Path(output_dir) / "domain_classification_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization dashboard to {output_path}")

    return str(output_path)


def generate_analysis_report(analysis_results, data, output_path):
    """Generate comprehensive analysis results."""
    print("\n=== GENERATING ANALYSIS REPORT ===")

    # Combine all analysis results into a comprehensive dataset
    report_data = {}

    # Overall statistics
    report_data["total_citations"] = len(data)
    report_data["unique_domains"] = data["domain"].nunique()
    report_data["unique_models"] = (
        data["model_name_llm"].nunique() if "model_name_llm" in data.columns else 0
    )

    # Domain distribution
    report_data["domain_distribution"] = analysis_results.get("domain_counts", {})

    # Model preferences
    if "model_domain_percentages" in analysis_results:
        report_data["model_preferences"] = analysis_results[
            "model_domain_percentages"
        ].to_dict()

    # Intent patterns
    if "intent_domain_percentages" in analysis_results:
        report_data["intent_patterns"] = analysis_results[
            "intent_domain_percentages"
        ].to_dict()

    # Winner patterns
    if "comparison_df" in analysis_results:
        report_data["winner_advantage"] = analysis_results["comparison_df"][
            "Difference"
        ].to_dict()

    # Create summary DataFrame for easy analysis
    summary_df = pd.DataFrame(
        {
            "metric": list(report_data.keys()),
            "value": [str(v) for v in report_data.values()],
        }
    )

    # Save analysis results
    analysis_df = pd.DataFrame(analysis_results.get("domain_counts", {}), index=[0]).T
    analysis_df.columns = ["citation_count"]
    analysis_df["percentage"] = analysis_results.get("domain_percentages", {})

    print(f"Saving analysis results to {output_path}")
    analysis_df.to_parquet(output_path, index=True)

    return report_data


def main():
    """Main function for domain classification analysis."""
    # Get paths from Snakemake
    input_path = snakemake.input.integrated_citations
    analysis_output_path = snakemake.output.analysis_results
    report_output_path = snakemake.output.report

    # Create output directory
    output_dir = Path(analysis_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load integrated data
    data = load_integrated_data(input_path)

    # Perform all analyses
    analysis_results = {}

    # 1. Overall domain distribution
    overall_results = analyze_overall_domain_distribution(data)
    analysis_results.update(overall_results)

    # 2. Model domain preferences
    model_results = analyze_model_domain_preferences(data)
    analysis_results.update(model_results)

    # 3. Intent-domain patterns
    intent_results = analyze_intent_domain_patterns(data)
    analysis_results.update(intent_results)

    # 4. Model A vs B comparison
    comparison_results = analyze_model_comparison(data)
    analysis_results.update(comparison_results)

    # 5. Winner vs loser patterns
    winner_results = analyze_winner_domain_patterns(data)
    analysis_results.update(winner_results)

    # Create visualizations
    dashboard_path = create_visualizations(analysis_results, output_dir)

    # Generate comprehensive report
    report_data = generate_analysis_report(analysis_results, data, analysis_output_path)

    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Domain Classification Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ margin: 10px 0; }}
            .number {{ font-weight: bold; color: #2c3e50; }}
        </style>
    </head>
    <body>
        <h1>Domain Classification Analysis Report</h1>

        <h2>Summary Statistics</h2>
        <div class="metric">Total Citations: <span class="number">{report_data.get("total_citations", "N/A"):,}</span></div>
        <div class="metric">Unique Domains: <span class="number">{report_data.get("unique_domains", "N/A"):,}</span></div>
        <div class="metric">Unique Models: <span class="number">{report_data.get("unique_models", "N/A"):,}</span></div>

        <h2>Interactive Dashboard</h2>
        <p><a href="{Path(dashboard_path).name}">View Interactive Dashboard</a></p>

        <h2>Key Findings</h2>
        <ul>
            <li>Analysis of domain classification patterns across AI models</li>
            <li>Model specialization in different source types</li>
            <li>Query intent impact on citation behavior</li>
            <li>Performance correlation with citation patterns</li>
        </ul>

        <p><em>Generated by Citation Analysis Pipeline</em></p>
    </body>
    </html>
    """

    with open(report_output_path, "w") as f:
        f.write(html_content)

    print(f"\n✅ Domain classification analysis completed!")
    print(f"Analysis results: {analysis_output_path}")
    print(f"HTML report: {report_output_path}")
    print(f"Interactive dashboard: {dashboard_path}")


if __name__ == "__main__":
    main()
