#!/usr/bin/env python3
"""
Generate top 20 most frequent news sources for each model family.

This script analyzes news citations to identify the most frequently cited
news sources by each model family and outputs the results as a LaTeX table
for inclusion in research papers.
"""

import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_news_citations(input_path):
    """Load news citations data."""
    logger.info(f"Loading news citations from {input_path}")
    news_citations = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(news_citations):,} news citations")
    return news_citations


def calculate_top_sources_by_family(df, model_family_col, top_n=20):
    """Calculate top N most frequent news sources for each model family."""
    logger.info(f"Calculating top {top_n} sources by model family")

    # Get all model families
    families = df[model_family_col].unique()
    logger.info(f"Found {len(families)} model families: {list(families)}")

    all_results = []

    for family in families:
        if pd.isna(family):
            continue

        family_data = df[df[model_family_col] == family]
        logger.info(f"Processing {family}: {len(family_data):,} citations")

        # Count frequency by domain
        domain_counts = family_data["domain"].value_counts(normalize=True)
        top_domains = domain_counts.head(top_n)

        # Calculate cumulative frequency
        total_citations = len(family_data)
        cumulative_freq = 0

        for rank, (domain, frequency) in enumerate(top_domains.items(), 1):
            cumulative_freq += frequency

            all_results.append(
                {
                    "model_family": family,
                    "rank": rank,
                    "domain": domain,
                    "frequency": frequency,
                    "cumulative_frequency": cumulative_freq,
                }
            )

    results_df = pd.DataFrame(all_results)
    logger.info(f"Generated results for {len(results_df)} entries")

    return results_df


def format_latex_table(df):
    """Format results as LaTeX table."""
    logger.info("Formatting results as LaTeX table")

    # Sort by model family and rank
    df_sorted = df.sort_values(["model_family", "rank"])

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Top 20 Most Frequent News Sources by Model Family}",
        "\\label{tab:top_news_sources}",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Model Family & Domain & Frequency & Cumulative Frequency \\\\",
        "\\midrule",
    ]

    current_family = None
    for _, row in df_sorted.iterrows():
        family = row["model_family"]
        domain = row["domain"]
        frequency = row["frequency"]
        cumulative_freq = row["cumulative_frequency"]

        # Only show family name for first entry of each family
        if family != current_family:
            family_display = family
            current_family = family
        else:
            family_display = ""

        latex_lines.append(
            f"{family_display} & {domain} & {frequency:.3f} & {cumulative_freq:.3f} \\\\"
        )

        # Add spacing after each family group
        if family_display:
            latex_lines.append("\\addlinespace[0.5em]")

    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    return "\n".join(latex_lines)


def save_results(results_df, latex_table, output_dir):
    """Save results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV for reference
    csv_path = output_dir / "top_news_sources_by_family.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV results to {csv_path}")

    # Save LaTeX table
    latex_path = output_dir / "top_news_sources_latex_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    logger.info(f"Saved LaTeX table to {latex_path}")

    # Also print to console for immediate use
    print("\n" + "=" * 80)
    print("LATEX TABLE FOR PAPER:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    return csv_path, latex_path


def main():
    """Main function."""
    # Get paths from Snakemake
    input_path = snakemake.input[0]
    output_csv = snakemake.output.csv
    output_latex = snakemake.output.latex

    # Load data
    news_citations = load_news_citations(input_path)

    # Calculate top sources by family
    results_df = calculate_top_sources_by_family(news_citations, "model_family")

    # Format as LaTeX table
    latex_table = format_latex_table(results_df)

    # Save CSV results
    results_df.to_csv(output_csv, index=False)
    logger.info(f"Saved CSV results to {output_csv}")

    # Save LaTeX table
    with open(output_latex, "w") as f:
        f.write(latex_table)
    logger.info(f"Saved LaTeX table to {output_latex}")

    # Print summary
    logger.info("=== TOP NEWS SOURCES BY FAMILY SUMMARY ===")
    families = results_df["model_family"].unique()
    for family in sorted(families):
        family_data = results_df[results_df["model_family"] == family]
        logger.info(f"{family}: {len(family_data)} top domains")

    # Also print LaTeX table to console
    print("\n" + "=" * 80)
    print("LATEX TABLE FOR PAPER:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    logger.info("✅ Top news sources analysis completed!")


if __name__ == "__main__":
    main()
