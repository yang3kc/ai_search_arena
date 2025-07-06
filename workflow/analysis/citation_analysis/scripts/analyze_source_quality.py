#!/usr/bin/env python3
"""
Source Quality Analysis for AI Search Arena Citations

This script analyzes the quality and credibility patterns of sources cited by AI models,
using domain credibility ratings and quality metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
import warnings

warnings.filterwarnings("ignore")


def load_news_citations(file_path):
    """Load news citations data."""
    return pd.read_parquet(file_path)


def load_quality_ratings(file_path):
    """Load domain quality/credibility ratings."""
    df = pd.read_csv(file_path, compression="gzip")
    return df


def integrate_quality_data(citations_df, quality_df):
    """Integrate news citations with quality ratings."""
    # Check if quality data is already integrated
    if "domain_quality_score" in citations_df.columns:
        print("Quality data already integrated in news citations.")
        print("Using existing 'domain_quality_score' column as reliability metric.")

        # Add reliability column for consistency with analysis functions
        citations_df["reliability"] = citations_df["domain_quality_score"]
        return citations_df

    # If not integrated, perform integration
    print("Integrating external quality data...")
    quality_clean = quality_df.copy()
    quality_clean["domain"] = quality_clean["domain"].str.lower().str.strip()

    # Use 'domain' column (which is base domain in our data)
    citations_df["domain_lower"] = citations_df["domain"].str.lower()

    # Select primary reliability metric from quality data (pc1 appears to be the main score)
    quality_clean["reliability"] = quality_clean["pc1"]

    # Merge quality ratings
    quality_citations = citations_df.merge(
        quality_clean[["domain", "reliability"]],
        left_on="domain_lower",
        right_on="domain",
        how="left",
    )

    # Clean up columns
    quality_citations = quality_citations.drop(
        ["domain_y", "domain_lower"], axis=1, errors="ignore"
    )
    if "domain_x" in quality_citations.columns:
        quality_citations = quality_citations.rename(columns={"domain_x": "domain"})

    return quality_citations


def analyze_model_quality_patterns(data):
    """Analyze source quality patterns by individual AI models."""
    print("=== AI MODEL QUALITY ANALYSIS ===\n")

    # Focus on citations with quality scores
    quality_data = data[data["reliability"].notna()].copy()

    print("Quality Analysis Coverage:")
    print(f"- Total news citations: {len(data):,}")
    print(f"- Citations with quality scores: {len(quality_data):,}")
    print(f"- Coverage rate: {len(quality_data) / len(data) * 100:.1f}%\n")

    # Model quality statistics
    model_quality_stats = (
        quality_data.groupby("model_name_raw")["reliability"]
        .agg(["mean", "median", "std", "count", "min", "max"])
        .round(3)
    )
    model_quality_stats.columns = [
        "Mean_Quality",
        "Median_Quality",
        "Std_Quality",
        "Citations",
        "Min_Quality",
        "Max_Quality",
    ]
    model_quality_stats = model_quality_stats.sort_values(
        "Mean_Quality", ascending=False
    )

    print("Model Quality Rankings (by Mean Reliability Score):")
    print(model_quality_stats.to_string())
    print()

    # Quality score ranges by model
    print("Quality Score Ranges by Model:")
    for model in model_quality_stats.index:
        model_data = quality_data[quality_data["model_name_raw"] == model][
            "reliability"
        ]
        q25, q75 = np.percentile(model_data, [25, 75])
        print(
            f"{model:35} | Range: {model_data.min():.3f} - {model_data.max():.3f} | IQR: {q25:.3f} - {q75:.3f}"
        )
    print()

    # Statistical significance tests between models
    print("Statistical Significance Tests (Quality Differences):")
    models = quality_data["model_name_raw"].unique()
    significant_pairs = []

    for i, model1 in enumerate(models):
        for model2 in models[i + 1 :]:
            scores1 = quality_data[quality_data["model_name_raw"] == model1][
                "reliability"
            ]
            scores2 = quality_data[quality_data["model_name_raw"] == model2][
                "reliability"
            ]

            if len(scores1) > 10 and len(scores2) > 10:
                statistic, p_value = mannwhitneyu(
                    scores1, scores2, alternative="two-sided"
                )
                mean_diff = scores1.mean() - scores2.mean()
                effect_size = abs(mean_diff) / ((scores1.std() + scores2.std()) / 2)

                if p_value < 0.05:
                    significant_pairs.append(
                        {
                            "Model1": model1,
                            "Model2": model2,
                            "Mean_Diff": mean_diff,
                            "P_Value": p_value,
                            "Effect_Size": effect_size,
                        }
                    )

    if significant_pairs:
        sig_df = pd.DataFrame(significant_pairs)
        sig_df = sig_df.sort_values("P_Value")
        print(f"Found {len(sig_df)} significant differences (p < 0.05):")
        for _, row in sig_df.head(10).iterrows():
            print(
                f"{row['Model1']:25} vs {row['Model2']:25} | "
                f"Diff: {row['Mean_Diff']:+.3f} | p: {row['P_Value']:.2e} | "
                f"Effect: {row['Effect_Size']:.3f}"
            )
    else:
        print("No significant quality differences found between models.")
    print()

    return model_quality_stats


def analyze_model_family_quality_patterns(data):
    """Analyze source quality patterns by AI model family with detailed statistical analysis."""
    print("=== AI MODEL FAMILY QUALITY ANALYSIS ===\n")

    # Filter for citations with quality scores
    quality_data = data[data["reliability"].notna()].copy()

    print("Quality Analysis Coverage by Family:")
    family_coverage = (
        data.groupby("model_family")
        .agg({"reliability": ["count", lambda x: x.notna().sum()]})
        .round(1)
    )
    family_coverage.columns = ["Total_Citations", "With_Quality"]
    family_coverage["Coverage_Rate"] = (
        family_coverage["With_Quality"] / family_coverage["Total_Citations"] * 100
    ).round(1)
    print(family_coverage.to_string())
    print()

    # Model family quality statistics with min/max
    family_quality_stats = (
        quality_data.groupby("model_family")["reliability"]
        .agg(["mean", "median", "std", "count", "min", "max"])
        .round(3)
    )
    family_quality_stats.columns = [
        "Mean_Quality",
        "Median_Quality",
        "Std_Quality",
        "Citations",
        "Min_Quality",
        "Max_Quality",
    ]
    family_quality_stats = family_quality_stats.sort_values(
        "Mean_Quality", ascending=False
    )

    print("Model Family Quality Rankings:")
    print(family_quality_stats.to_string())
    print()

    # Quantile analysis by family
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    family_quantiles = (
        quality_data.groupby("model_family")["reliability"]
        .quantile(quantiles)
        .unstack()
    )
    family_quantiles.columns = [f"Q{int(q * 100)}" for q in quantiles]
    family_quantiles = family_quantiles.round(3)

    print("Quality Distribution Quantiles by Model Family:")
    print(family_quantiles.to_string())
    print()

    # Quality consistency rankings
    quality_consistency = family_quality_stats[["Std_Quality", "Mean_Quality"]].copy()
    quality_consistency["Consistency_Rank"] = quality_consistency["Std_Quality"].rank()
    quality_consistency = quality_consistency.sort_values("Std_Quality")

    print("Quality Consistency Rankings (Lower Std = More Consistent):")
    for family, row in quality_consistency.iterrows():
        print(
            f"{family:12} | Std: {row['Std_Quality']:.3f} | Mean: {row['Mean_Quality']:.3f} | Rank: {int(row['Consistency_Rank'])}"
        )
    print()

    # Statistical significance tests with effect sizes
    families = quality_data["model_family"].unique()
    print("Statistical Significance Tests Between Model Families:")

    for i, family1 in enumerate(families):
        for family2 in families[i + 1 :]:
            scores1 = quality_data[quality_data["model_family"] == family1][
                "reliability"
            ]
            scores2 = quality_data[quality_data["model_family"] == family2][
                "reliability"
            ]

            if len(scores1) > 30 and len(scores2) > 30:
                statistic, p_value = mannwhitneyu(
                    scores1, scores2, alternative="two-sided"
                )
                mean_diff = scores1.mean() - scores2.mean()
                effect_size = abs(mean_diff) / ((scores1.std() + scores2.std()) / 2)

                significance = (
                    "***"
                    if p_value < 0.001
                    else "**"
                    if p_value < 0.01
                    else "*"
                    if p_value < 0.05
                    else "ns"
                )
                print(
                    f"{family1:12} vs {family2:12} | "
                    f"Diff: {mean_diff:+.3f} | p: {p_value:.2e} | "
                    f"Effect: {effect_size:.3f} | {significance}"
                )
    print()

    # Domain preference patterns by quality-aware families (if domain classification available)
    if "domain_classification" in quality_data.columns:
        print("High-Quality Domain Preferences by Model Family:")
        high_quality_threshold = quality_data["reliability"].quantile(0.75)
        high_quality_citations = quality_data[
            quality_data["reliability"] >= high_quality_threshold
        ]

        family_domain_prefs = (
            high_quality_citations.groupby(["model_family", "domain_classification"])
            .size()
            .unstack(fill_value=0)
        )
        family_domain_prefs_pct = (
            family_domain_prefs.div(family_domain_prefs.sum(axis=1), axis=0) * 100
        )

        # Show top domain preferences for high-quality sources
        available_domains = family_domain_prefs_pct.columns
        print(f"\nAvailable domain types: {list(available_domains)}")

        # Show preferences for each available domain type
        for domain in available_domains:
            print(f"\n{domain.title()} Domain Preference (High-Quality Sources):")
            domain_prefs = family_domain_prefs_pct[domain].sort_values(ascending=False)
            for family, pct in domain_prefs.items():
                print(f"  {family:12}: {pct:5.1f}%")
    else:
        print(
            "Domain classification data not available for detailed preference analysis."
        )

    return family_quality_stats, family_quantiles, quality_consistency


def analyze_winner_loser_quality(data):
    """Analyze quality differences between winners and losers."""
    print("=== WINNER vs LOSER QUALITY ANALYSIS ===\n")

    # Filter for citations with quality scores and clear win/loss status
    quality_data = data[data["reliability"].notna()].copy()

    # Use 'model_won' column instead of 'is_winner'
    winner_loser_data = quality_data[quality_data["model_won"].notna()].copy()

    print("Winner/Loser Quality Analysis Coverage:")
    print(f"- Citations with quality scores: {len(quality_data):,}")
    print(f"- Citations with win/loss status: {len(winner_loser_data):,}")
    print(
        f"- Analysis coverage: {len(winner_loser_data) / len(quality_data) * 100:.1f}%\n"
    )

    # Quality statistics by winner status
    winner_quality_stats = (
        winner_loser_data.groupby("model_won")["reliability"]
        .agg(["mean", "median", "std", "count", "min", "max"])
        .round(3)
    )

    print("Quality Statistics by Winner Status:")
    print(winner_quality_stats.to_string())
    print()

    # Statistical test
    winners = winner_loser_data[winner_loser_data["model_won"] == True]["reliability"]
    losers = winner_loser_data[winner_loser_data["model_won"] == False]["reliability"]

    if len(winners) > 0 and len(losers) > 0:
        statistic, p_value = mannwhitneyu(winners, losers, alternative="two-sided")
        mean_diff = winners.mean() - losers.mean()
        effect_size = abs(mean_diff) / ((winners.std() + losers.std()) / 2)

        print("Statistical Test Results:")
        print(f"- Winners mean quality: {winners.mean():.3f}")
        print(f"- Losers mean quality: {losers.mean():.3f}")
        print(f"- Mean difference: {mean_diff:+.3f}")
        print(f"- Mann-Whitney U p-value: {p_value:.2e}")
        print(f"- Effect size (Cohen's d): {effect_size:.3f}")
        print(f"- Statistical significance: {'Yes' if p_value < 0.05 else 'No'}")
        print()

    return winner_quality_stats


def create_quality_visualizations(data, output_dir):
    """Create comprehensive quality analysis visualizations."""
    # Set style for better looking plots
    plt.style.use("default")
    sns.set_palette("husl")

    quality_data = data[data["reliability"].notna()].copy()

    # 1. Model Quality Comparison (Box Plot)
    plt.figure(figsize=(15, 8))
    models_order = (
        quality_data.groupby("model_name_raw")["reliability"]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    box_plot = plt.boxplot(
        [
            quality_data[quality_data["model_name_raw"] == model]["reliability"].values
            for model in models_order
        ],
        positions=range(len(models_order)),
        patch_artist=True,
    )

    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(models_order)))
    for patch, color in zip(box_plot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.xticks(range(len(models_order)), models_order, rotation=45, ha="right")
    plt.ylabel("Reliability Score")
    plt.title("Source Quality Scores by AI Model")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "04_model_quality_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Model Family Quality Distribution
    plt.figure(figsize=(12, 8))
    families_order = (
        quality_data.groupby("model_family")["reliability"]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    family_box_plot = plt.boxplot(
        [
            quality_data[quality_data["model_family"] == family]["reliability"].values
            for family in families_order
        ],
        positions=range(len(families_order)),
        patch_artist=True,
    )

    colors = plt.cm.Set3(np.linspace(0, 1, len(families_order)))
    for patch, color in zip(family_box_plot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    plt.xticks(range(len(families_order)), families_order, rotation=45, ha="right")
    plt.ylabel("Reliability Score")
    plt.title("Source Quality Scores by AI Model Family")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "04_family_quality_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Quality Heatmap by Model
    plt.figure(figsize=(10, 8))

    # Create quality bins
    quality_data["quality_category"] = pd.cut(
        quality_data["reliability"],
        bins=3,
        labels=["Low_Quality", "Medium_Quality", "High_Quality"],
    )

    # Create heatmap data
    quality_dist = (
        quality_data.groupby(["model_name_raw", "quality_category"])
        .size()
        .unstack(fill_value=0)
    )
    quality_dist_pct = quality_dist.div(quality_dist.sum(axis=1), axis=0) * 100

    # Sort by high quality percentage
    quality_dist_pct = quality_dist_pct.sort_values("High_Quality", ascending=False)

    # Create heatmap
    sns.heatmap(
        quality_dist_pct,
        annot=True,
        fmt=".1f",
        cmap="RdYlBu_r",
        cbar_kws={"label": "% of Citations"},
    )
    plt.title("Source Quality Distribution by AI Model")
    plt.xlabel("Quality Category")
    plt.ylabel("AI Model")
    plt.tight_layout()
    plt.savefig(
        output_dir / "04_model_quality_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 4. Winner vs Loser Quality Analysis
    winner_loser_data = quality_data[quality_data["model_won"].notna()].copy()

    if len(winner_loser_data) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Box plot comparison
        winner_loser_data["Status"] = winner_loser_data["model_won"].map(
            {True: "Winners", False: "Losers"}
        )

        box_plot = ax1.boxplot(
            [
                winner_loser_data[winner_loser_data["Status"] == status][
                    "reliability"
                ].values
                for status in ["Winners", "Losers"]
            ],
            labels=["Winners", "Losers"],
            patch_artist=True,
        )

        box_plot["boxes"][0].set_facecolor("green")
        box_plot["boxes"][1].set_facecolor("red")
        box_plot["boxes"][0].set_alpha(0.7)
        box_plot["boxes"][1].set_alpha(0.7)

        ax1.set_ylabel("Reliability Score")
        ax1.set_title("Source Quality: Winners vs Losers")
        ax1.grid(True, alpha=0.3)

        # Mean comparison with error bars
        status_stats = (
            winner_loser_data.groupby("Status")["reliability"]
            .agg(["mean", "std"])
            .reset_index()
        )

        bars = ax2.bar(
            status_stats["Status"],
            status_stats["mean"],
            yerr=status_stats["std"],
            capsize=5,
            color=["green", "red"],
            alpha=0.7,
        )

        ax2.set_ylabel("Average Reliability Score")
        ax2.set_title("Average Source Quality with Standard Deviation")

        # Add statistical test result
        winners = winner_loser_data[winner_loser_data["Status"] == "Winners"][
            "reliability"
        ]
        losers = winner_loser_data[winner_loser_data["Status"] == "Losers"][
            "reliability"
        ]

        if len(winners) > 0 and len(losers) > 0:
            _, p_value = mannwhitneyu(winners, losers, alternative="two-sided")
            significance_text = f"p = {p_value:.2e}"
            if p_value < 0.001:
                significance_text += "\n***"
            elif p_value < 0.01:
                significance_text += "\n**"
            elif p_value < 0.05:
                significance_text += "\n*"
            else:
                significance_text += "\nns"

            ax2.text(
                0.5,
                max(status_stats["mean"]) * 0.9,
                significance_text,
                ha="center",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(
            output_dir / "05_winner_loser_quality.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    print(f"Quality visualizations saved to {output_dir}")


def generate_quality_report(data, model_stats, family_stats, winner_stats, output_dir):
    """Generate comprehensive HTML report for quality analysis."""
    quality_data = data[data["reliability"].notna()].copy()

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Search Arena - Source Quality Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }}
            h3 {{ color: #7f8c8d; }}
            .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px 20px; padding: 10px; background-color: #3498db; color: white; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #bdc3c7; padding: 12px; text-align: left; }}
            th {{ background-color: #34495e; color: white; }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            .visualization {{ text-align: center; margin: 30px 0; }}
            .visualization img {{ max-width: 100%; height: auto; border: 1px solid #bdc3c7; border-radius: 5px; }}
            .key-finding {{ background-color: #d5e8d4; padding: 15px; border-left: 5px solid #82b366; margin: 15px 0; }}
        </style>
    </head>
    <body>
        <h1>AI Search Arena - Source Quality Analysis Report</h1>

        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metric">Total News Citations: {len(data):,}</div>
            <div class="metric">Citations with Quality Scores: {len(quality_data):,}</div>
            <div class="metric">Coverage Rate: {len(quality_data) / len(data) * 100:.1f}%</div>
            <div class="metric">Unique Models Analyzed: {quality_data["model_name_raw"].nunique()}</div>
            <div class="metric">Unique Model Families: {quality_data["model_family"].nunique()}</div>
        </div>

        <div class="key-finding">
            <h3>🎯 Key Finding</h3>
            <p><strong>Quality Leadership:</strong> {model_stats.index[0]} leads in source quality with an average reliability score of {model_stats.iloc[0]["Mean_Quality"]:.3f},
            while {model_stats.index[-1]} has the lowest average score of {model_stats.iloc[-1]["Mean_Quality"]:.3f}.</p>

            <p><strong>Family Performance:</strong> {family_stats.index[0]} family shows the highest average quality ({family_stats.iloc[0]["Mean_Quality"]:.3f}),
            while {family_stats.index[-1]} family has the most variable quality (std: {family_stats.iloc[-1]["Std_Quality"]:.3f}).</p>
        </div>

        <h2>Source Quality by Individual Models</h2>
        <div class="visualization">
            <img src="04_model_quality_comparison.png" alt="Model Quality Comparison">
        </div>

        <table>
            <tr>
                <th>Model</th>
                <th>Mean Quality</th>
                <th>Median Quality</th>
                <th>Std Quality</th>
                <th>Citations</th>
                <th>Quality Range</th>
            </tr>
    """

    for model, stats in model_stats.iterrows():
        html_content += f"""
            <tr>
                <td>{model}</td>
                <td>{stats["Mean_Quality"]:.3f}</td>
                <td>{stats["Median_Quality"]:.3f}</td>
                <td>{stats["Std_Quality"]:.3f}</td>
                <td>{stats["Citations"]:,}</td>
                <td>{stats["Min_Quality"]:.3f} - {stats["Max_Quality"]:.3f}</td>
            </tr>
        """

    html_content += f"""
        </table>

        <h2>Source Quality by Model Family</h2>
        <div class="visualization">
            <img src="04_family_quality_comparison.png" alt="Family Quality Comparison">
        </div>

        <div class="visualization">
            <img src="04_model_quality_heatmap.png" alt="Quality Distribution Heatmap">
        </div>

        <h2>Winner vs Loser Quality Analysis</h2>
        <div class="visualization">
            <img src="05_winner_loser_quality.png" alt="Winner vs Loser Quality">
        </div>

        <h2>Technical Notes</h2>
        <ul>
            <li><strong>Quality Scores:</strong> Based on domain credibility ratings from lin_domain_ratings.csv.gz</li>
            <li><strong>Statistical Tests:</strong> Mann-Whitney U tests used for non-parametric comparisons</li>
            <li><strong>Effect Sizes:</strong> Cohen's d calculated for practical significance assessment</li>
            <li><strong>Coverage:</strong> Analysis includes {len(quality_data) / len(data) * 100:.1f}% of news citations with available quality scores</li>
        </ul>

        <p><em>Report generated on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
    </body>
    </html>
    """

    # Save HTML report
    report_path = output_dir / "source_quality_analysis_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Quality analysis report saved to: {report_path}")


def main():
    """Main analysis pipeline for source quality analysis."""
    print("Starting AI Search Arena - Source Quality Analysis")
    print("=" * 60)

    # Use Snakemake inputs/outputs if available, otherwise fallback to file paths
    try:
        # Snakemake mode - use input/output parameters
        news_citations_path = snakemake.input.news_citations
        quality_integrated_output = snakemake.output.quality_integrated_citations
        quality_report_output = snakemake.output.quality_report

        # Derive paths from Snakemake outputs
        output_dir = Path(quality_integrated_output).parent

        # Setup data paths relative to Snakemake working directory
        quality_ratings_path = Path("../../../data/raw_data/lin_domain_ratings.csv.gz")

    except NameError:
        # Standalone mode - construct paths manually
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent.parent
        data_dir = project_root / "data"

        # Input paths
        news_citations_path = (
            data_dir / "intermediate" / "citation_analysis" / "news_citations.parquet"
        )
        quality_ratings_path = data_dir / "raw_data" / "lin_domain_ratings.csv.gz"

        # Output paths
        output_dir = data_dir / "intermediate" / "citation_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        quality_integrated_output = output_dir / "quality_integrated_citations.parquet"
        quality_report_output = output_dir / "source_quality_analysis_report.html"

    # Load data
    print("Loading data...")
    news_citations = load_news_citations(news_citations_path)
    quality_ratings = load_quality_ratings(quality_ratings_path)

    # Integrate quality data with citations
    print("Integrating quality data with citations...")
    quality_citations = integrate_quality_data(news_citations, quality_ratings)

    # Save integrated dataset
    quality_citations.to_parquet(quality_integrated_output, index=False)
    print(f"Quality-integrated citations saved to: {quality_integrated_output}")

    # Perform analyses
    print("\nPerforming quality analyses...")
    model_stats = analyze_model_quality_patterns(quality_citations)
    family_stats, family_quantiles, quality_consistency = (
        analyze_model_family_quality_patterns(quality_citations)
    )
    winner_stats = analyze_winner_loser_quality(quality_citations)

    # Create visualizations
    print("\nGenerating visualizations...")
    create_quality_visualizations(quality_citations, output_dir)

    # Generate comprehensive report
    print("\nGenerating analysis report...")

    # For standalone mode, use output_dir path, for Snakemake mode use direct path
    try:
        report_path = quality_report_output
        output_path_for_report = output_dir
    except NameError:
        report_path = output_dir / "source_quality_analysis_report.html"
        output_path_for_report = output_dir

    generate_quality_report(
        quality_citations,
        model_stats,
        family_stats,
        winner_stats,
        output_path_for_report,
    )

    print("\n✅ Source Quality Analysis Complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 View the complete report: {report_path}")


if __name__ == "__main__":
    main()
