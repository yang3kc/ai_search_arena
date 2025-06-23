#!/usr/bin/env python3
"""
Enrich citations with political leaning scores.

This script adds political leaning metrics to citations data:
- political_leaning_score (continuous): from DomainDemo_political_leaning.csv.gz
- political_leaning (categorical): left_leaning/right_leaning/unknown
"""

import pandas as pd
import gzip
from pathlib import Path


def load_political_leaning_data(filepath):
    """Load and process political leaning data."""
    print(f"Loading political leaning data from {filepath}")
    
    with gzip.open(filepath, "rt") as f:
        df = pd.read_csv(f)
    
    print(f"Loaded {len(df)} domains with political leaning scores")
    print(f"Columns: {list(df.columns)}")
    
    # Clean domain names (remove any whitespace)
    df["domain"] = df["domain"].str.strip()
    
    # Keep only leaning_score_users and rename it
    df = df[["domain", "leaning_score_users"]].copy()
    df = df.rename(columns={"leaning_score_users": "political_leaning_score"})
    
    # Create categorical political leaning variable
    df["political_leaning"] = "unknown"
    df.loc[df["political_leaning_score"] < 0, "political_leaning"] = "left_leaning"
    df.loc[df["political_leaning_score"] >= 0, "political_leaning"] = "right_leaning"
    df.loc[df["political_leaning_score"].isna(), "political_leaning"] = "unknown"
    
    print(f"Political leaning distribution:")
    leaning_counts = df["political_leaning"].value_counts()
    for leaning, count in leaning_counts.items():
        print(f"  {leaning}: {count} ({count / len(df) * 100:.1f}%)")
    
    return df


def enrich_with_political_leaning(citations_df, political_leaning_data):
    """Merge citations with political leaning data."""
    print("Enriching citations with political leaning data...")
    
    # Merge with political leaning on base domain
    enriched = citations_df.merge(
        political_leaning_data, 
        on="domain", 
        how="left", 
        suffixes=("", "_pol")
    )
    
    # Count matches
    pol_matches = enriched["political_leaning_score"].notna().sum()
    total_citations = len(enriched)
    pol_domains_covered = enriched[enriched["political_leaning_score"].notna()]["domain"].nunique()
    total_unique_domains = enriched["domain"].nunique()
    
    print(f"Political leaning matches: {pol_matches:,} / {total_citations:,} ({pol_matches / total_citations * 100:.1f}%)")
    print(f"Political leaning domain coverage: {pol_domains_covered:,} / {total_unique_domains:,} unique domains ({pol_domains_covered / total_unique_domains * 100:.1f}%)")
    
    return enriched


def main():
    """Main function for Snakemake execution."""
    # Get paths from Snakemake
    citations_path = snakemake.input.citations
    political_leaning_path = snakemake.input.political_leaning
    output_path = snakemake.output[0]
    
    # Load data
    print("Loading citations data...")
    citations = pd.read_parquet(citations_path)
    print(f"Loaded {len(citations)} citations")
    
    political_leaning_data = load_political_leaning_data(political_leaning_path)
    
    # Enrich citations
    enriched_citations = enrich_with_political_leaning(citations, political_leaning_data)
    
    # Save enriched dataset
    print(f"\nSaving enriched citations to {output_path}")
    enriched_citations.to_parquet(output_path, index=False)
    
    print(f"✅ Political leaning enrichment completed!")
    print(f"Output: {len(enriched_citations):,} rows, {len(enriched_citations.columns)} columns")


if __name__ == "__main__":
    main()