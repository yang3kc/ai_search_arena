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
import logging
from sklearn.metrics.pairwise import cosine_similarity

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


def create_domain_vectors(model_domains, citations_df):
    """Create one-hot encoded vectors for each model based on news domain citations."""
    logger.info("Creating domain vectors for cosine similarity...")

    # Get all unique domains across all models
    all_domains = set()
    for domains in model_domains.values():
        all_domains.update(domains)
    all_domains = sorted(list(all_domains))

    logger.info(f"Total unique domains across all models: {len(all_domains):,}")

    # Create domain to index mapping
    domain_to_idx = {domain: idx for idx, domain in enumerate(all_domains)}

    # Create vectors for each model
    model_vectors = {}
    models = list(model_domains.keys())

    for model in models:
        # Get citation counts for this model (weighted by frequency)
        model_citations = citations_df[citations_df["model_name_raw"] == model]
        domain_counts = model_citations["domain"].value_counts()

        # Create vector with citation counts (not just binary)
        vector = np.zeros(len(all_domains))
        for domain, count in domain_counts.items():
            if domain in domain_to_idx:
                vector[domain_to_idx[domain]] = count

        model_vectors[model] = vector
        logger.info(
            f"{model}: {np.count_nonzero(vector):,} domains cited, "
            f"total citations: {int(vector.sum()):,}"
        )

    return model_vectors, all_domains


def calculate_cosine_similarities(model_vectors):
    """Calculate pairwise cosine similarities between model vectors."""
    logger.info("Calculating pairwise cosine similarities...")

    models = list(model_vectors.keys())

    # Create matrix of all model vectors
    vectors_matrix = np.array([model_vectors[model] for model in models])

    # Calculate cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(vectors_matrix)

    # Extract pairwise similarities
    cosine_similarities = []
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:  # Only calculate upper triangle to avoid duplicates
                cosine_sim = cosine_sim_matrix[i][j]

                # Additional vector statistics
                vector1 = model_vectors[model1]
                vector2 = model_vectors[model2]

                dot_product = np.dot(vector1, vector2)
                norm1 = np.linalg.norm(vector1)
                norm2 = np.linalg.norm(vector2)

                cosine_similarities.append(
                    {
                        "model1": model1,
                        "model2": model2,
                        "cosine_similarity": cosine_sim,
                        "dot_product": dot_product,
                        "norm1": norm1,
                        "norm2": norm2,
                        "vector1_citations": int(vector1.sum()),
                        "vector2_citations": int(vector2.sum()),
                        "shared_citation_weight": dot_product,  # Sum of min counts for shared domains
                    }
                )

    return pd.DataFrame(cosine_similarities)


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


def analyze_provider_similarities(jaccard_df, cosine_df=None):
    """Analyze similarities grouped by provider families."""
    logger.info("Analyzing similarities by provider families...")

    # Add provider information for Jaccard similarities
    jaccard_df["provider1"] = jaccard_df["model1"].apply(get_model_provider)
    jaccard_df["provider2"] = jaccard_df["model2"].apply(get_model_provider)

    # Categorize comparisons
    def get_comparison_type(row):
        if row["provider1"] == row["provider2"]:
            return f"Within {row['provider1']}"
        else:
            providers = sorted([row["provider1"], row["provider2"]])
            return f"{providers[0]} vs {providers[1]}"

    jaccard_df["comparison_type"] = jaccard_df.apply(get_comparison_type, axis=1)

    # Calculate summary statistics by comparison type for Jaccard
    jaccard_analysis = (
        jaccard_df.groupby("comparison_type")
        .agg(
            {
                "jaccard_similarity": ["mean", "std", "min", "max", "count"],
                "intersection_size": ["mean", "std"],
                "union_size": ["mean", "std"],
            }
        )
        .round(4)
    )

    # If cosine similarities are provided, analyze them too
    cosine_analysis = None
    if cosine_df is not None:
        cosine_df["provider1"] = cosine_df["model1"].apply(get_model_provider)
        cosine_df["provider2"] = cosine_df["model2"].apply(get_model_provider)
        cosine_df["comparison_type"] = cosine_df.apply(get_comparison_type, axis=1)

        cosine_analysis = (
            cosine_df.groupby("comparison_type")
            .agg(
                {
                    "cosine_similarity": ["mean", "std", "min", "max", "count"],
                    "dot_product": ["mean", "std"],
                    "shared_citation_weight": ["mean", "std"],
                }
            )
            .round(4)
        )

    return jaccard_df, cosine_df, jaccard_analysis, cosine_analysis


def create_similarity_matrix(similarities_df, similarity_column="jaccard_similarity"):
    """Create a symmetric similarity matrix."""
    logger.info(f"Creating similarity matrix for {similarity_column}...")

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
        similarity = row[similarity_column]
        matrix.loc[model1, model2] = similarity
        matrix.loc[model2, model1] = similarity  # Symmetric

    return matrix


def generate_analysis_report(
    jaccard_df, cosine_df, jaccard_analysis, cosine_analysis, model_domains, output_dir
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
        f"- **Total Model Pairs Analyzed**: {len(jaccard_df):,}",
        "",
        "### Jaccard Similarity (Set Overlap)",
        f"- **Average Jaccard Similarity**: {jaccard_df['jaccard_similarity'].mean():.3f}",
        f"- **Highest Jaccard Similarity**: {jaccard_df['jaccard_similarity'].max():.3f}",
        f"- **Lowest Jaccard Similarity**: {jaccard_df['jaccard_similarity'].min():.3f}",
        "",
        "### Cosine Similarity (Vector Similarity)",
        f"- **Average Cosine Similarity**: {cosine_df['cosine_similarity'].mean():.3f}",
        f"- **Highest Cosine Similarity**: {cosine_df['cosine_similarity'].max():.3f}",
        f"- **Lowest Cosine Similarity**: {cosine_df['cosine_similarity'].min():.3f}",
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
            "## Top 10 Most Similar Model Pairs (Jaccard)",
            "",
        ]
    )

    # Top Jaccard similarities
    top_jaccard = jaccard_df.nlargest(10, "jaccard_similarity")
    for _, row in top_jaccard.iterrows():
        report_lines.append(
            f"- **{row['model1']}** vs **{row['model2']}**: "
            f"{row['jaccard_similarity']:.3f} "
            f"({row['intersection_size']:,}/{row['union_size']:,} domains)"
        )

    report_lines.extend(
        [
            "",
            "## Top 10 Most Similar Model Pairs (Cosine)",
            "",
        ]
    )

    # Top Cosine similarities
    top_cosine = cosine_df.nlargest(10, "cosine_similarity")
    for _, row in top_cosine.iterrows():
        report_lines.append(
            f"- **{row['model1']}** vs **{row['model2']}**: "
            f"{row['cosine_similarity']:.3f} "
            f"(citations: {row['vector1_citations']:,} vs {row['vector2_citations']:,})"
        )

    report_lines.extend(
        [
            "",
            "## Most Different Model Pairs (Jaccard)",
            "",
        ]
    )

    # Bottom Jaccard similarities
    bottom_jaccard = jaccard_df.nsmallest(10, "jaccard_similarity")
    for _, row in bottom_jaccard.iterrows():
        report_lines.append(
            f"- **{row['model1']}** vs **{row['model2']}**: "
            f"{row['jaccard_similarity']:.3f} "
            f"({row['intersection_size']:,}/{row['union_size']:,} domains)"
        )

    report_lines.extend(
        [
            "",
            "## Most Different Model Pairs (Cosine)",
            "",
        ]
    )

    # Bottom Cosine similarities
    bottom_cosine = cosine_df.nsmallest(10, "cosine_similarity")
    for _, row in bottom_cosine.iterrows():
        report_lines.append(
            f"- **{row['model1']}** vs **{row['model2']}**: "
            f"{row['cosine_similarity']:.3f} "
            f"(citations: {row['vector1_citations']:,} vs {row['vector2_citations']:,})"
        )

    report_lines.extend(
        [
            "",
            "## Provider Family Analysis",
            "",
            "### Jaccard Similarity by Provider Family",
            "",
        ]
    )

    # Jaccard provider analysis
    for comparison_type in jaccard_analysis.index:
        stats = jaccard_analysis.loc[comparison_type]
        mean_sim = stats[("jaccard_similarity", "mean")]
        std_sim = stats[("jaccard_similarity", "std")]
        count = int(stats[("jaccard_similarity", "count")])

        report_lines.append(f"#### {comparison_type}")
        report_lines.append(f"- **Pairs**: {count}")
        report_lines.append(f"- **Average Jaccard**: {mean_sim:.3f} ± {std_sim:.3f}")
        report_lines.append(
            f"- **Range**: {stats[('jaccard_similarity', 'min')]:.3f} - {stats[('jaccard_similarity', 'max')]:.3f}"
        )
        report_lines.append("")

    report_lines.extend(
        [
            "",
            "### Cosine Similarity by Provider Family",
            "",
        ]
    )

    # Cosine provider analysis
    for comparison_type in cosine_analysis.index:
        stats = cosine_analysis.loc[comparison_type]
        mean_sim = stats[("cosine_similarity", "mean")]
        std_sim = stats[("cosine_similarity", "std")]
        count = int(stats[("cosine_similarity", "count")])

        report_lines.append(f"#### {comparison_type}")
        report_lines.append(f"- **Pairs**: {count}")
        report_lines.append(f"- **Average Cosine**: {mean_sim:.3f} ± {std_sim:.3f}")
        report_lines.append(
            f"- **Range**: {stats[('cosine_similarity', 'min')]:.3f} - {stats[('cosine_similarity', 'max')]:.3f}"
        )
        report_lines.append("")

    # Add methodology explanation
    report_lines.extend(
        [
            "",
            "## Methodology",
            "",
            "### Jaccard Similarity",
            "- Measures overlap between sets of cited domains",
            "- Formula: |A ∩ B| / |A ∪ B|",
            "- Range: 0 (no overlap) to 1 (identical sets)",
            "- Focuses on domain diversity, not citation frequency",
            "",
            "### Cosine Similarity",
            "- Measures similarity between citation frequency vectors",
            "- Uses citation counts for each domain across all models",
            "- Range: 0 (orthogonal) to 1 (identical patterns)",
            "- Considers both domain overlap and citation intensity",
            "",
        ]
    )

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
        output_jaccard = snakemake.output.similarities_jaccard
        output_cosine = snakemake.output.similarities_cosine
        output_matrix_jaccard = snakemake.output.similarity_matrix_jaccard
        output_matrix_cosine = snakemake.output.similarity_matrix_cosine
        output_summary = snakemake.output.analysis_summary
        output_dir = Path(snakemake.output.similarities_jaccard).parent
    except NameError:
        # For standalone execution
        input_path = (
            "../../../data/intermediate/citation_analysis/news_citations.parquet"
        )
        output_dir = Path("../../../data/intermediate/citation_analysis")
        output_jaccard = output_dir / "news_similarities_jaccard.parquet"
        output_cosine = output_dir / "news_similarities_cosine.parquet"
        output_matrix_jaccard = output_dir / "news_similarity_matrix_jaccard.parquet"
        output_matrix_cosine = output_dir / "news_similarity_matrix_cosine.parquet"
        output_summary = output_dir / "news_similarity_summary.json"
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    citations_df = load_news_citations(input_path)

    # Extract domain sets for each model
    model_domains = get_model_domain_sets(citations_df)

    # Calculate Jaccard similarities
    jaccard_df = calculate_pairwise_similarities(model_domains)

    # Create domain vectors and calculate cosine similarities
    model_vectors, all_domains = create_domain_vectors(model_domains, citations_df)
    cosine_df = calculate_cosine_similarities(model_vectors)

    # Analyze by provider families
    jaccard_df, cosine_df, jaccard_analysis, cosine_analysis = (
        analyze_provider_similarities(jaccard_df, cosine_df)
    )

    # Create similarity matrices
    jaccard_matrix = create_similarity_matrix(jaccard_df, "jaccard_similarity")
    cosine_matrix = create_similarity_matrix(cosine_df, "cosine_similarity")

    # Save results
    jaccard_df.to_parquet(output_jaccard, index=False)
    cosine_df.to_parquet(output_cosine, index=False)
    jaccard_matrix.to_parquet(output_matrix_jaccard)
    cosine_matrix.to_parquet(output_matrix_cosine)

    # Generate summary statistics
    summary_stats = {
        "total_models": len(model_domains),
        "total_pairs": len(jaccard_df),
        "total_unique_domains": len(all_domains),
        "jaccard_similarity": {
            "avg": float(jaccard_df["jaccard_similarity"].mean()),
            "std": float(jaccard_df["jaccard_similarity"].std()),
            "min": float(jaccard_df["jaccard_similarity"].min()),
            "max": float(jaccard_df["jaccard_similarity"].max()),
        },
        "cosine_similarity": {
            "avg": float(cosine_df["cosine_similarity"].mean()),
            "std": float(cosine_df["cosine_similarity"].std()),
            "min": float(cosine_df["cosine_similarity"].min()),
            "max": float(cosine_df["cosine_similarity"].max()),
        },
        "model_domain_counts": {
            model: len(domains) for model, domains in model_domains.items()
        },
        "model_citation_counts": {
            model: int(model_vectors[model].sum()) for model in model_vectors.keys()
        },
        # Convert provider analyses to JSON-serializable format
        "jaccard_provider_analysis": {
            str(idx): {
                str(col): float(val) if pd.notna(val) else None
                for col, val in row.items()
            }
            for idx, row in jaccard_analysis.iterrows()
        },
        "cosine_provider_analysis": {
            str(idx): {
                str(col): float(val) if pd.notna(val) else None
                for col, val in row.items()
            }
            for idx, row in cosine_analysis.iterrows()
        },
    }

    with open(output_summary, "w") as f:
        json.dump(summary_stats, f, indent=2)

    # Generate comprehensive report
    generate_analysis_report(
        jaccard_df,
        cosine_df,
        jaccard_analysis,
        cosine_analysis,
        model_domains,
        output_dir,
    )

    logger.info("News similarity analysis completed successfully!")

    # Print key findings
    logger.info("=== KEY FINDINGS ===")
    logger.info(
        f"Average Jaccard Similarity: {jaccard_df['jaccard_similarity'].mean():.3f}"
    )
    logger.info(
        f"Average Cosine Similarity: {cosine_df['cosine_similarity'].mean():.3f}"
    )

    most_similar_jaccard = jaccard_df.loc[jaccard_df["jaccard_similarity"].idxmax()]
    logger.info(
        f"Most Similar (Jaccard): {most_similar_jaccard['model1']} vs {most_similar_jaccard['model2']} ({most_similar_jaccard['jaccard_similarity']:.3f})"
    )

    most_similar_cosine = cosine_df.loc[cosine_df["cosine_similarity"].idxmax()]
    logger.info(
        f"Most Similar (Cosine): {most_similar_cosine['model1']} vs {most_similar_cosine['model2']} ({most_similar_cosine['cosine_similarity']:.3f})"
    )


if __name__ == "__main__":
    main()
