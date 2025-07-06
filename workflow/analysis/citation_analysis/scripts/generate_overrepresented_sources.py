#!/usr/bin/env python3
"""
Generate overrepresented news sources for each model family using log-odds ratios.

This script analyzes news citations to identify which news sources are
overrepresented in each model family compared to all others using log-odds
ratios. The output is formatted as a LaTeX table for inclusion in research papers.
"""

import pandas as pd
from collections import Counter
from math import log, sqrt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def logodds(corpora_dic, bg_counter):
    """Calculate the log odds ratio of term i's frequency between
    a target corpus and another corpus, with the prior information from
    a background corpus.

    Args:
        corpora_dic: dictionary of Counter objects (corpora of our interest)
        bg_counter: Counter object (background corpus)

    Returns:
        dictionary of dictionaries containing log odds ratio of each word
    """
    corp_size = dict([(c, sum(corpora_dic[c].values())) for c in corpora_dic])
    bg_size = sum(bg_counter.values())
    result = dict([(c, {}) for c in corpora_dic])

    for name, c in corpora_dic.items():
        for word in c:
            fi = c[word]
            fj = sum(co.get(word, 0) for x, co in corpora_dic.items() if x != name)
            fbg = bg_counter[word]
            ni = corp_size[name]
            nj = sum(x for idx, x in corp_size.items() if idx != name)
            nbg = bg_size
            oddsratio = (
                log(fi + fbg)
                - log(ni + nbg - (fi + fbg))
                - log(fj + fbg)
                + log(nj + nbg - (fj + fbg))
            )
            std = 1.0 / (fi + fbg) + 1.0 / (fj + fbg)
            z = oddsratio / sqrt(std)
            result[name][word] = z

    return result


def load_news_citations(input_path):
    """Load news citations data."""
    logger.info(f"Loading news citations from {input_path}")
    news_citations = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(news_citations):,} news citations")
    return news_citations


def calculate_overrepresented_sources(df, top_n=20):
    """Calculate overrepresented sources for each model family using log-odds."""
    logger.info(f"Calculating top {top_n} overrepresented sources by model family")

    # Get all model families
    families = ["gpt", "gemini", "perplexity"]
    logger.info(f"Found {len(families)} model families: {list(families)}")

    # Create counters for each family
    family_counters = {}
    for family in families:
        if pd.isna(family):
            continue
        family_data = df[df["model_family"] == family]
        family_counters[family] = Counter(family_data["domain"])
        logger.info(
            f"{family}: {len(family_data):,} citations, {len(family_counters[family])} unique domains"
        )

    # Background counter (all citations)
    all_counter = Counter(df["domain"])

    all_results = []

    # Calculate log-odds for each family vs all others
    for target_family in family_counters.keys():
        logger.info(f"Processing {target_family} vs others")

        # Create other families counter (all except target)
        other_families_data = df[df["model_family"] != target_family]
        other_counter = Counter(other_families_data["domain"])

        # Calculate log-odds
        logodds_result = logodds(
            {target_family: family_counters[target_family], "other": other_counter},
            all_counter,
        )

        # Extract results for target family and sort by log-odds (descending)
        family_logodds = logodds_result[target_family]
        sorted_domains = sorted(
            family_logodds.items(), key=lambda x: x[1], reverse=True
        )

        # Take top N overrepresented domains
        top_domains = sorted_domains[:top_n]

        for rank, (domain, logodds_score) in enumerate(top_domains, 1):
            # Get frequency data for context
            frequency = family_counters[target_family][domain]
            total_family_citations = sum(family_counters[target_family].values())
            percentage = (frequency / total_family_citations) * 100
            
            # Get political leaning and quality info for this domain
            family_data = df[df["model_family"] == target_family]
            domain_data = family_data[family_data["domain"] == domain]
            
            # Get most common political leaning and quality for this domain
            political_leaning = (
                domain_data["political_leaning"].mode().iloc[0]
                if len(domain_data["political_leaning"].mode()) > 0
                else "unknown_leaning"
            )
            domain_quality = (
                domain_data["domain_quality"].mode().iloc[0]
                if len(domain_data["domain_quality"].mode()) > 0
                else "unknown_quality"
            )

            all_results.append(
                {
                    "model_family": target_family,
                    "rank": rank,
                    "domain": domain,
                    "logodds_score": logodds_score,
                    "frequency": frequency,
                    "percentage": percentage,
                    "political_leaning": political_leaning,
                    "domain_quality": domain_quality,
                }
            )

    results_df = pd.DataFrame(all_results)
    logger.info(f"Generated results for {len(results_df)} entries")

    return results_df


def format_political_leaning_code(political_leaning):
    """Format political leaning as single letter code."""
    return {
        "left_leaning": "L",
        "center_leaning": "C",
        "right_leaning": "R",
        "unknown_leaning": "U",
    }.get(political_leaning, "U")


def format_quality_code(domain_quality):
    """Format domain quality as single letter code."""
    return {"high_quality": "H", "low_quality": "L", "unknown_quality": "U"}.get(
        domain_quality, "U"
    )


def format_latex_table_sidebyside(df):
    """Format results as LaTeX table with families side by side."""
    logger.info("Formatting results as side-by-side LaTeX table")

    # Get top families by average log-odds score
    top_families = ["gpt", "gemini", "perplexity"]

    # Prepare data for each family
    family_data = {}
    for family in top_families:
        family_df = df[df["model_family"] == family].head(20)  # Top 20 for each
        # Include domain, log-odds score, and political/quality info
        family_data[family] = family_df[
            ["domain", "logodds_score", "political_leaning", "domain_quality"]
        ].values.tolist()

    # Start LaTeX table
    latex_lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Top 20 Overrepresented News Sources by Model Family (Log-Odds Ratios) with Political Leaning and Quality. "
        "Political leaning: L=Left, C=Center, R=Right, U=Unknown. "
        "Quality: H=High, L=Low, U=Unknown.}",
        "\\label{tab:overrepresented_news_sources}",
        "\\begin{tabular}{lrcc|lrcc|lrcc}",
        "\\toprule",
    ]

    # Model family header row
    family_header_parts = []
    for family in top_families:
        family_header_parts.extend(
            [f"\\multicolumn{{4}}{{c|}}{{\\textbf{{{family}}}}}"]
        )
    # Remove the last | from the last column
    family_header_parts[-1] = family_header_parts[-1].replace("c|", "c")

    latex_lines.append(" & ".join(family_header_parts) + " \\\\")
    latex_lines.append("\\midrule")

    # Column headers
    header_parts = []
    for i, family in enumerate(top_families):
        header_parts.extend(["Domain", "Log-Odds", "L", "Q"])

    latex_lines.append(" & ".join(header_parts) + " \\\\")
    latex_lines.append("\\midrule")

    # Data rows - find max length
    max_rows = max(len(data) for data in family_data.values())

    for i in range(max_rows):
        row_parts = []
        for family in top_families:
            if i < len(family_data[family]):
                domain, logodds_score, political_leaning, domain_quality = family_data[family][i]
                # Escape special LaTeX characters and truncate long domains
                domain_escaped = domain.replace("_", "\\_").replace("&", "\\&")
                if len(domain_escaped) > 18:  # Reduced to make room for new columns
                    domain_escaped = domain_escaped[:15] + "..."
                
                # Generate separate political leaning and quality codes
                leaning_code = format_political_leaning_code(political_leaning)
                quality_code = format_quality_code(domain_quality)
                
                row_parts.extend([domain_escaped, f"{logodds_score:.2f}", leaning_code, quality_code])
            else:
                row_parts.extend(["", "", "", ""])

        latex_lines.append(" & ".join(row_parts) + " \\\\")

    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])

    return "\n".join(latex_lines)


def main():
    """Main function."""
    # Get paths from Snakemake
    input_path = snakemake.input[0]
    output_csv = snakemake.output.csv
    output_latex = snakemake.output.latex

    # Load data
    news_citations = load_news_citations(input_path)

    # Calculate overrepresented sources by family
    results_df = calculate_overrepresented_sources(news_citations)

    # Format as LaTeX table
    latex_table = format_latex_table_sidebyside(results_df)

    # Create output directory
    output_dir = Path(output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV results
    results_df.to_csv(output_csv, index=False)
    logger.info(f"Saved CSV results to {output_csv}")

    # Save LaTeX table
    with open(output_latex, "w") as f:
        f.write(latex_table)
    logger.info(f"Saved LaTeX table to {output_latex}")

    # Print summary
    logger.info("=== OVERREPRESENTED NEWS SOURCES BY FAMILY SUMMARY ===")
    families = results_df["model_family"].unique()
    for family in sorted(families):
        family_data = results_df[results_df["model_family"] == family]
        avg_logodds = family_data["logodds_score"].mean()
        logger.info(
            f"{family}: {len(family_data)} top domains, avg log-odds: {avg_logodds:.2f}"
        )

    # Also print LaTeX table to console
    print("\n" + "=" * 80)
    print("LATEX TABLE FOR PAPER:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    logger.info("✅ Overrepresented news sources analysis completed!")


if __name__ == "__main__":
    main()
