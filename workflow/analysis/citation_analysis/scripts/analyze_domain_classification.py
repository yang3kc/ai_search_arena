#!/usr/bin/env python3
"""
Domain Classification Analysis Script.

This script analyzes citation patterns across different domain classifications,
examining how AI models cite different source types and identifying specialization patterns.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

    if "model_name_raw" not in data.columns:
        print("Model information not available")
        return {}

    # Model-domain cross-tabulation
    model_domain_crosstab = pd.crosstab(
        data["model_name_raw"], data["domain_classification"]
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

            print("\nDifferences (Model B - Model A):")
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
    """Create separate visualizations for each domain classification analysis."""
    print("\n=== CREATING SEPARATE VISUALIZATIONS ===")

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("Set2")

    output_files = []

    # 1. Overall domain distribution (bar chart)
    if "domain_counts" in analysis_results:
        fig, ax = plt.subplots(figsize=(12, 8))
        domain_counts = analysis_results["domain_counts"]
        classified_counts = domain_counts

        bars = ax.bar(
            range(len(classified_counts)), classified_counts.values, color="skyblue"
        )
        ax.set_title(
            "Overall Domain Distribution (Classified Domains Only)", fontsize=16, pad=20
        )
        ax.set_xticks(range(len(classified_counts)))
        ax.set_xticklabels(classified_counts.index, rotation=45, ha="right")
        ax.set_ylabel("Citation Count", fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        output_path = Path(output_dir) / "01_overall_domain_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_files.append(output_path)
        print(f"Saved overall domain distribution to {output_path}")

    # 2. Model domain preferences (grouped bar chart)
    if "model_domain_percentages" in analysis_results:
        fig, ax = plt.subplots(figsize=(16, 10))
        model_domain_pcts = analysis_results["model_domain_percentages"]
        # Show all domain types except unclassified
        classified_domains = model_domain_pcts.columns[
            model_domain_pcts.columns != "unclassified"
        ]

        plot_data = model_domain_pcts[classified_domains]
        plot_data.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(
            "Model Domain Preferences (Classified Domains Only)", fontsize=16, pad=20
        )
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylabel("% of Citations", fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = Path(output_dir) / "02_model_domain_preferences.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_files.append(output_path)
        print(f"Saved model preferences to {output_path}")

    # 3. Intent-domain heatmap
    if "intent_domain_percentages" in analysis_results:
        fig, ax = plt.subplots(figsize=(12, 8))
        intent_domain_pcts = analysis_results["intent_domain_percentages"]
        # Exclude unclassified from the visualization
        classified_domains = intent_domain_pcts.columns[
            intent_domain_pcts.columns != "unclassified"
        ]

        sns.heatmap(
            intent_domain_pcts[classified_domains],
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "% of Citations"},
        )
        ax.set_title(
            "Intent-Domain Citation Patterns (Classified Domains Only)",
            fontsize=16,
            pad=20,
        )
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=10)
        ax.set_xlabel("Domain Classification", fontsize=12)
        ax.set_ylabel("Query Intent", fontsize=12)

        plt.tight_layout()
        output_path = Path(output_dir) / "03_intent_domain_patterns.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_files.append(output_path)
        print(f"Saved intent patterns to {output_path}")

    # 4. Model A vs B comparison
    if "side_domain_percentages" in analysis_results:
        fig, ax = plt.subplots(figsize=(12, 8))
        side_domain_pcts = analysis_results["side_domain_percentages"]

        if len(side_domain_pcts) >= 2:
            side_domain_pcts.plot(kind="bar", ax=ax, width=0.7)
            ax.set_title("Model A vs B Domain Distribution", fontsize=16, pad=20)
            ax.tick_params(axis="x", rotation=45, labelsize=10)
            ax.set_ylabel("% of Citations", fontsize=12)
            ax.legend(title="Model Side", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            output_path = Path(output_dir) / "04_model_a_vs_b_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            output_files.append(output_path)
            print(f"Saved A vs B comparison to {output_path}")

    # 5. Winner vs loser comparison
    if "comparison_df" in analysis_results:
        fig, ax = plt.subplots(figsize=(12, 8))
        comparison_df = analysis_results["comparison_df"].head(10)

        colors = ["green" if x > 0 else "red" for x in comparison_df["Difference"]]
        bars = ax.bar(
            range(len(comparison_df)),
            comparison_df["Difference"],
            color=colors,
            alpha=0.7,
        )
        ax.set_title("Winner vs Loser Domain Preferences", fontsize=16, pad=20)
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels(comparison_df.index, rotation=45, ha="right")
        ax.set_ylabel("Difference (Winners - Losers) %", fontsize=12)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:+.1f}",
                ha="center",
                va="bottom" if height > 0 else "top",
            )

        plt.tight_layout()
        output_path = Path(output_dir) / "05_winner_vs_loser_patterns.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        output_files.append(output_path)
        print(f"Saved winner vs loser patterns to {output_path}")

    print(f"\n✅ Created {len(output_files)} separate visualization files")
    return [str(path) for path in output_files]


def generate_analysis_report(analysis_results, data, output_path):
    """Generate comprehensive analysis results."""
    print("\n=== GENERATING ANALYSIS REPORT ===")

    # Combine all analysis results into a comprehensive dataset
    report_data = {}

    # Overall statistics
    report_data["total_citations"] = len(data)
    report_data["unique_domains"] = data["domain"].nunique()
    report_data["unique_models"] = (
        data["model_name_raw"].nunique() if "model_name_raw" in data.columns else 0
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
    visualization_paths = create_visualizations(analysis_results, output_dir)

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

        <h2>Analysis Visualizations</h2>
        <ul>
            {"".join(f'<li><a href="{Path(path).name}">{Path(path).stem.replace("_", " ").title()}</a></li>' for path in visualization_paths)}
        </ul>

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

    print("\n✅ Domain classification analysis completed!")
    print(f"Analysis results: {analysis_output_path}")
    print(f"HTML report: {report_output_path}")
    print(f"Generated {len(visualization_paths)} visualization files:")
    for path in visualization_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
