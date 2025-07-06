#!/usr/bin/env python3
"""
Generate a LaTeX table with model statistics for the research paper.

This script creates a table showing:
- Model name
- Number of responses
- Number of citations
- Number of news citations (count and percentage)
"""

import pandas as pd
import sys
import os


def load_data(responses_path, citations_path):
    """Load the required data files."""
    print("Loading data...")

    # Load responses data
    responses = pd.read_parquet(responses_path)
    print(f"Loaded {len(responses):,} responses")

    # Load enriched citations data
    citations = pd.read_parquet(citations_path)
    print(f"Loaded {len(citations):,} citations")

    return responses, citations


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


def generate_model_stats(responses, citations):
    """Generate model statistics table."""
    print("Generating model statistics...")

    # Get model response counts
    model_responses = responses.groupby("model_name_raw").size().reset_index()
    model_responses.columns = ["model_name", "num_responses"]

    # Get citation counts per model
    # First, join citations with responses to get model info
    citations_with_model = citations.merge(
        responses[["response_id", "model_name_raw"]], on="response_id", how="left"
    )

    # Count total citations per model
    model_citations = (
        citations_with_model.groupby("model_name_raw").size().reset_index()
    )
    model_citations.columns = ["model_name", "num_citations"]

    # Count news citations per model
    news_citations = (
        citations_with_model[citations_with_model["domain_classification"] == "news"]
        .groupby("model_name_raw")
        .size()
        .reset_index()
    )
    news_citations.columns = ["model_name", "num_news_citations"]

    # Merge all statistics
    stats = model_responses.merge(model_citations, on="model_name", how="left")
    stats = stats.merge(news_citations, on="model_name", how="left")

    # Fill NaN values with 0 for models with no citations
    stats["num_citations"] = stats["num_citations"].fillna(0).astype(int)
    stats["num_news_citations"] = stats["num_news_citations"].fillna(0).astype(int)

    # Calculate news citation percentage
    stats["news_citation_percentage"] = (
        stats["num_news_citations"] / stats["num_citations"] * 100
    ).round(1)

    # Handle division by zero (models with no citations)
    stats.loc[stats["num_citations"] == 0, "news_citation_percentage"] = 0.0

    # Add provider column
    stats["provider"] = stats["model_name"].apply(get_model_provider)

    # Sort by provider first, then by number of responses (descending) within each provider
    stats = stats.sort_values(["provider", "num_responses"], ascending=[True, False])

    return stats


def generate_latex_table(stats):
    """Generate LaTeX table from statistics using pandas to_latex."""
    print("Generating LaTeX table...")

    # Prepare data for LaTeX table
    table_data = stats.copy()

    # Format numbers with commas and percentage
    table_data["num_responses_formatted"] = table_data["num_responses"].apply(
        lambda x: f"{x:,}"
    )
    table_data["num_citations_formatted"] = table_data["num_citations"].apply(
        lambda x: f"{x:,}"
    )
    table_data["num_news_citations_formatted"] = table_data["num_news_citations"].apply(
        lambda x: f"{x:,}"
    )
    table_data["news_percentage_formatted"] = table_data[
        "news_citation_percentage"
    ].apply(lambda x: f"{x:.1f}\\%")

    # Add totals row
    total_responses = stats["num_responses"].sum()
    total_citations = stats["num_citations"].sum()
    total_news_citations = stats["num_news_citations"].sum()
    total_news_percentage = (
        (total_news_citations / total_citations * 100) if total_citations > 0 else 0.0
    )

    totals_row = pd.DataFrame(
        {
            "provider": ["\\textbf{All}"],
            "model_name": ["\\textbf{Total}"],
            "num_responses_formatted": [f"\\textbf{{{total_responses:,}}}"],
            "num_citations_formatted": [f"\\textbf{{{total_citations:,}}}"],
            "num_news_citations_formatted": [f"\\textbf{{{total_news_citations:,}}}"],
            "news_percentage_formatted": [
                f"\\textbf{{{total_news_percentage:.1f}\\%}}"
            ],
        }
    )

    # Select columns for output
    display_columns = [
        "provider",
        "model_name",
        "num_responses_formatted",
        "num_citations_formatted",
        "num_news_citations_formatted",
        "news_percentage_formatted",
    ]

    # Combine data and totals
    table_data_formatted = table_data[display_columns]
    table_data_final = pd.concat([table_data_formatted, totals_row], ignore_index=True)

    # Generate LaTeX table using to_latex
    latex_table = table_data_final.to_latex(
        index=False,
        column_format="llrrrr",
        columns=display_columns,
        header=[
            "Provider",
            "Model",
            "Responses",
            "Citations",
            "News Citations",
            "News \\%",
        ],
        escape=False,  # Don't escape LaTeX commands
        caption="Model Statistics: Response Counts and Citation Patterns by Provider",
        label="tab:model_stats",
        position="htbp",
    )

    # Add midrule before totals row
    lines = latex_table.split("\n")
    # Find the line with Total and add midrule before it
    for i, line in enumerate(lines):
        if "\\textbf{Total}" in line:
            lines.insert(i, "\\midrule")
            break

    return "\n".join(lines)


def main():
    """Main function."""
    # Get parameters from snakemake or command line
    if "snakemake" in globals():
        responses_path = snakemake.input.responses
        citations_path = snakemake.input.citations_enriched
        output_file = snakemake.output[0]
    else:
        if len(sys.argv) != 4:
            print(
                "Usage: python generate_model_stats_table.py <responses_path> <citations_path> <output_file>"
            )
            sys.exit(1)
        responses_path = sys.argv[1]
        citations_path = sys.argv[2]
        output_file = sys.argv[3]

    # Load data
    responses, citations = load_data(responses_path, citations_path)

    # Generate statistics
    stats = generate_model_stats(responses, citations)

    # Print summary
    print("\nModel Statistics Summary:")
    print(stats.to_string(index=False))

    # Generate LaTeX table
    latex_table = generate_latex_table(stats)

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(latex_table)

    print(f"\nLaTeX table saved to: {output_file}")
    print("\nLaTeX table:")
    print(latex_table)


if __name__ == "__main__":
    main()
