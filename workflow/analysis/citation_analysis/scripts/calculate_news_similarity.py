#!/usr/bin/env python3
"""
Calculate pairwise Jaccard similarity between models based on their cited news sources.

This script analyzes how similar different AI models are in terms of which news
domains they cite, providing insights into citation diversity and overlap patterns.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_news_citations(input_path):
    """Load news citations data."""
    logger.info(f"Loading news citations from {input_path}")
    citations_df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(citations_df):,} news citations")
    return citations_df


def get_model_domain_sets(citations_df):
    """Get set of domains cited by each model."""
    logger.info("Extracting domain sets for each model...")

    model_domains = {}
    models = citations_df["model_name_raw"].unique()

    for model in models:
        model_citations = citations_df[citations_df["model_name_raw"] == model]
        domains = set(model_citations["domain"].unique())
        model_domains[model] = domains
        logger.info(f"{model}: {len(domains):,} unique news domains")

    return model_domains


def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0  # Both empty sets are considered identical

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union > 0 else 0.0


def calculate_pairwise_similarities(model_domains):
    """Calculate pairwise Jaccard similarities between all models."""
    logger.info("Calculating pairwise Jaccard similarities...")

    models = list(model_domains.keys())
    similarities = []

    # Calculate similarities for all pairs
    for model1, model2 in combinations(models, 2):
        domains1 = model_domains[model1]
        domains2 = model_domains[model2]

        jaccard_sim = calculate_jaccard_similarity(domains1, domains2)

        # Calculate additional metrics
        intersection_size = len(domains1.intersection(domains2))
        union_size = len(domains1.union(domains2))

        similarities.append(
            {
                "model1": model1,
                "model2": model2,
                "jaccard_similarity": jaccard_sim,
                "intersection_size": intersection_size,
                "union_size": union_size,
                "model1_domains": len(domains1),
                "model2_domains": len(domains2),
                "model1_unique": len(domains1 - domains2),
                "model2_unique": len(domains2 - domains1),
            }
        )

    return pd.DataFrame(similarities)


def get_model_provider(model_name):
    """Extract provider from model name."""
    if model_name.startswith("gpt-"):
        return "OpenAI"
    elif model_name.startswith("sonar"):
        return "Perplexity"
    elif model_name.startswith("gemini"):
        return "Google"
    else:
        return "Unknown"


def analyze_provider_similarities(similarities_df):
    """Analyze similarities grouped by provider families."""
    logger.info("Analyzing similarities by provider families...")

    # Add provider information
    similarities_df["provider1"] = similarities_df["model1"].apply(get_model_provider)
    similarities_df["provider2"] = similarities_df["model2"].apply(get_model_provider)

    # Categorize comparisons
    def get_comparison_type(row):
        if row["provider1"] == row["provider2"]:
            return f"Within {row['provider1']}"
        else:
            providers = sorted([row["provider1"], row["provider2"]])
            return f"{providers[0]} vs {providers[1]}"

    similarities_df["comparison_type"] = similarities_df.apply(
        get_comparison_type, axis=1
    )

    # Calculate summary statistics by comparison type
    provider_analysis = (
        similarities_df.groupby("comparison_type")
        .agg(
            {
                "jaccard_similarity": ["mean", "std", "min", "max", "count"],
                "intersection_size": ["mean", "std"],
                "union_size": ["mean", "std"],
            }
        )
        .round(4)
    )

    return similarities_df, provider_analysis


def create_similarity_matrix(similarities_df):
    """Create a symmetric similarity matrix."""
    logger.info("Creating similarity matrix...")

    # Get all unique models
    models = sorted(
        set(similarities_df["model1"].tolist() + similarities_df["model2"].tolist())
    )

    # Initialize matrix
    matrix = pd.DataFrame(index=models, columns=models, dtype=float)

    # Fill diagonal with 1.0 (self-similarity)
    for model in models:
        matrix.loc[model, model] = 1.0

    # Fill matrix with similarities
    for _, row in similarities_df.iterrows():
        model1, model2 = row["model1"], row["model2"]
        similarity = row["jaccard_similarity"]
        matrix.loc[model1, model2] = similarity
        matrix.loc[model2, model1] = similarity  # Symmetric

    return matrix


def generate_analysis_report(
    similarities_df, provider_analysis, model_domains, output_dir
):
    """Generate a comprehensive analysis report."""
    logger.info("Generating analysis report...")

    report_lines = [
        "# News Citation Similarity Analysis Report",
        "",
        f"*Generated on: {pd.Timestamp.now().isoformat()}*",
        "",
        "## Executive Summary",
        "",
        f"- **Total Model Pairs Analyzed**: {len(similarities_df):,}",
        f"- **Average Jaccard Similarity**: {similarities_df['jaccard_similarity'].mean():.3f}",
        f"- **Highest Similarity**: {similarities_df['jaccard_similarity'].max():.3f}",
        f"- **Lowest Similarity**: {similarities_df['jaccard_similarity'].min():.3f}",
        "",
        "## Model Domain Coverage",
        "",
    ]

    # Add model domain counts
    for model, domains in sorted(
        model_domains.items(), key=lambda x: len(x[1]), reverse=True
    ):
        provider = get_model_provider(model)
        report_lines.append(
            f"- **{model}** ({provider}): {len(domains):,} unique news domains"
        )

    report_lines.extend(
        [
            "",
            "## Top 10 Most Similar Model Pairs",
            "",
        ]
    )

    # Top similarities
    top_similarities = similarities_df.nlargest(10, "jaccard_similarity")
    for _, row in top_similarities.iterrows():
        report_lines.append(
            f"- **{row['model1']}** vs **{row['model2']}**: "
            f"{row['jaccard_similarity']:.3f} "
            f"({row['intersection_size']:,}/{row['union_size']:,} domains)"
        )

    report_lines.extend(
        [
            "",
            "## Top 10 Most Different Model Pairs",
            "",
        ]
    )

    # Bottom similarities
    bottom_similarities = similarities_df.nsmallest(10, "jaccard_similarity")
    for _, row in bottom_similarities.iterrows():
        report_lines.append(
            f"- **{row['model1']}** vs **{row['model2']}**: "
            f"{row['jaccard_similarity']:.3f} "
            f"({row['intersection_size']:,}/{row['union_size']:,} domains)"
        )

    report_lines.extend(
        [
            "",
            "## Provider Family Analysis",
            "",
        ]
    )

    # Provider analysis
    for comparison_type in provider_analysis.index:
        stats = provider_analysis.loc[comparison_type]
        mean_sim = stats[("jaccard_similarity", "mean")]
        std_sim = stats[("jaccard_similarity", "std")]
        count = int(stats[("jaccard_similarity", "count")])

        report_lines.append(f"### {comparison_type}")
        report_lines.append(f"- **Pairs**: {count}")
        report_lines.append(f"- **Average Similarity**: {mean_sim:.3f} ± {std_sim:.3f}")
        report_lines.append(
            f"- **Range**: {stats[('jaccard_similarity', 'min')]:.3f} - {stats[('jaccard_similarity', 'max')]:.3f}"
        )
        report_lines.append("")

    # Save report
    report_path = Path(output_dir) / "news_similarity_analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Analysis report saved to: {report_path}")


def main():
    """Main function for Snakemake execution."""

    # Get paths from snakemake object if available, otherwise use defaults
    try:
        # When run by Snakemake
        input_path = snakemake.input.news_citations
        output_similarities = snakemake.output.similarities
        output_matrix = snakemake.output.similarity_matrix
        output_summary = snakemake.output.analysis_summary
        output_dir = Path(snakemake.output.similarities).parent
    except NameError:
        # For standalone execution
        input_path = (
            "../../../data/intermediate/citation_analysis/news_citations.parquet"
        )
        output_dir = Path("../../../data/intermediate/citation_analysis")
        output_similarities = output_dir / "news_similarities.parquet"
        output_matrix = output_dir / "news_similarity_matrix.parquet"
        output_summary = output_dir / "news_similarity_summary.json"
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    citations_df = load_news_citations(input_path)

    # Extract domain sets for each model
    model_domains = get_model_domain_sets(citations_df)

    # Calculate pairwise similarities
    similarities_df = calculate_pairwise_similarities(model_domains)

    # Analyze by provider families
    similarities_df, provider_analysis = analyze_provider_similarities(similarities_df)

    # Create similarity matrix
    similarity_matrix = create_similarity_matrix(similarities_df)

    # Save results
    similarities_df.to_parquet(output_similarities, index=False)
    similarity_matrix.to_parquet(output_matrix)

    # Generate summary statistics
    summary_stats = {
        "total_models": len(model_domains),
        "total_pairs": len(similarities_df),
        "avg_jaccard_similarity": float(similarities_df["jaccard_similarity"].mean()),
        "std_jaccard_similarity": float(similarities_df["jaccard_similarity"].std()),
        "min_jaccard_similarity": float(similarities_df["jaccard_similarity"].min()),
        "max_jaccard_similarity": float(similarities_df["jaccard_similarity"].max()),
        "model_domain_counts": {
            model: len(domains) for model, domains in model_domains.items()
        },
        # Convert provider_analysis to JSON-serializable format
        "provider_analysis": {
            str(idx): {
                str(col): float(val) if pd.notna(val) else None 
                for col, val in row.items()
            }
            for idx, row in provider_analysis.iterrows()
        },
    }

    with open(output_summary, "w") as f:
        json.dump(summary_stats, f, indent=2)

    # Generate comprehensive report
    generate_analysis_report(
        similarities_df, provider_analysis, model_domains, output_dir
    )

    logger.info("News similarity analysis completed successfully!")

    # Print key findings
    logger.info("=== KEY FINDINGS ===")
    logger.info(
        f"Average Jaccard Similarity: {similarities_df['jaccard_similarity'].mean():.3f}"
    )

    most_similar = similarities_df.loc[similarities_df["jaccard_similarity"].idxmax()]
    logger.info(
        f"Most Similar: {most_similar['model1']} vs {most_similar['model2']} ({most_similar['jaccard_similarity']:.3f})"
    )

    least_similar = similarities_df.loc[similarities_df["jaccard_similarity"].idxmin()]
    logger.info(
        f"Least Similar: {least_similar['model1']} vs {least_similar['model2']} ({least_similar['jaccard_similarity']:.3f})"
    )


if __name__ == "__main__":
    main()
