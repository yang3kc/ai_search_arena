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

        for rank, (domain, frequency) in enumerate(top_domains.items(), 1):
            all_results.append(
                {
                    "model_family": family,
                    "rank": rank,
                    "domain": domain,
                    "frequency": frequency,
                }
            )

    results_df = pd.DataFrame(all_results)
    logger.info(f"Generated results for {len(results_df)} entries")

    return results_df


def format_latex_table_sidebyside(df, families_to_show=3):
    """Format results as LaTeX table with families side by side."""
    logger.info("Formatting results as side-by-side LaTeX table")
    
    # Get top families by total citations
    family_totals = df.groupby('model_family')['frequency'].sum().sort_values(ascending=False)
    top_families = family_totals.head(families_to_show).index.tolist()
    
    logger.info(f"Showing top {families_to_show} families: {top_families}")
    
    # Prepare data for each family
    family_data = {}
    for family in top_families:
        family_df = df[df['model_family'] == family].head(20)  # Top 20 for each
        family_data[family] = family_df[['domain', 'frequency']].values.tolist()
    
    # Start LaTeX table
    latex_lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Top 20 Most Frequent News Sources by Model Family}",
        "\\label{tab:top_news_sources}",
        "\\begin{tabular}{lr|lr|lr}",
        "\\toprule"
    ]
    
    # Model family header row
    family_header_parts = []
    for family in top_families:
        family_header_parts.extend([f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{family}}}}}"])
    # Remove the last | from the last column
    family_header_parts[-1] = family_header_parts[-1].replace('c|', 'c')
    
    latex_lines.append(" & ".join(family_header_parts) + " \\\\")
    latex_lines.append("\\midrule")
    
    # Column headers
    header_parts = []
    for i, family in enumerate(top_families):
        if i < len(top_families) - 1:
            header_parts.extend(["Domain", "\\%"])
        else:
            header_parts.extend(["Domain", "\\%"])
    
    latex_lines.append(" & ".join(header_parts) + " \\\\")
    latex_lines.append("\\midrule")
    
    # Data rows - find max length
    max_rows = max(len(data) for data in family_data.values())
    
    # Calculate totals for each family
    family_totals = {}
    for family in top_families:
        total_pct = sum(freq for _, freq in family_data[family]) * 100
        family_totals[family] = total_pct
    
    for i in range(max_rows):
        row_parts = []
        for family in top_families:
            if i < len(family_data[family]):
                domain, freq = family_data[family][i]
                # Escape special LaTeX characters and truncate long domains
                domain_escaped = domain.replace('_', '\\_').replace('&', '\\&')
                if len(domain_escaped) > 25:
                    domain_escaped = domain_escaped[:22] + "..."
                row_parts.extend([domain_escaped, f"{freq*100:.1f}"])
            else:
                row_parts.extend(["", ""])
        
        latex_lines.append(" & ".join(row_parts) + " \\\\")
    
    # Add sum row
    latex_lines.append("\\midrule")
    sum_row_parts = []
    for family in top_families:
        sum_row_parts.extend(["\\textbf{Total}", f"\\textbf{{{family_totals[family]:.1f}}}"])
    latex_lines.append(" & ".join(sum_row_parts) + " \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    
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
    latex_table = format_latex_table_sidebyside(results_df)

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
