#!/usr/bin/env python3
"""
Extract unique domains from citations data.

This script extracts unique domains from citations and creates an intermediate
file with domain frequencies. This allows for efficient signal enrichment
without repeatedly loading the full citations dataset.
"""

import pandas as pd


def extract_domains_from_citations(citations_df):
    """Extract unique domains with their citation frequencies."""
    print("Extracting unique domains from citations...")

    # Get domain frequencies
    domain_counts = citations_df["domain"].value_counts()

    # Create domains dataframe
    domains_df = pd.DataFrame(
        {"domain": domain_counts.index, "citation_count": domain_counts.values}
    ).reset_index(drop=True)

    # Sort by citation count (descending) for easier analysis
    domains_df = domains_df.sort_values("citation_count", ascending=False).reset_index(
        drop=True
    )

    print(f"Extracted {len(domains_df)} unique domains")
    print(f"Total citations represented: {domains_df['citation_count'].sum():,}")
    print(f"Top 5 most cited domains:")
    for i, row in domains_df.head().iterrows():
        print(f"  {row['domain']}: {row['citation_count']:,} citations")

    return domains_df


def main():
    """Main function for Snakemake execution."""
    # Get paths from Snakemake
    citations_path = snakemake.input.citations
    output_path = snakemake.output[0]

    # Load citations data
    print(f"Loading citations from {citations_path}")
    citations = pd.read_parquet(citations_path)
    print(f"Loaded {len(citations):,} citations")

    # Extract unique domains
    domains_df = extract_domains_from_citations(citations)

    # Save domains dataset
    print(f"\nSaving domains to {output_path}")
    domains_df.to_parquet(output_path, index=False)

    print(f"✅ Domain extraction completed!")
    print(f"Output: {len(domains_df):,} unique domains")


if __name__ == "__main__":
    main()
