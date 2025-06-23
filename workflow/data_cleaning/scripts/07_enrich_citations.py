#!/usr/bin/env python3
"""
Enrich citations with political leaning and domain quality metrics.

This script merges the cleaned citations data with:
1. Domain political leaning scores (DomainDemo_political_leaning.csv.gz)
2. Domain credibility/quality ratings (lin_domain_ratings.csv.gz)

The output is an enriched citations table that serves as the primary dataset
for bias and credibility analysis.
"""

import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import sys


def load_political_leaning(filepath):
    """Load and process political leaning data."""
    print(f"Loading political leaning data from {filepath}")

    with gzip.open(filepath, "rt") as f:
        df = pd.read_csv(f)

    print(f"Loaded {len(df)} domains with political leaning scores")
    print(f"Columns: {list(df.columns)}")

    # Clean domain names (remove any whitespace)
    df["domain"] = df["domain"].str.strip()

    # Keep only leaning_score_users and create binary variable
    df = df[["domain", "leaning_score_users"]].copy()

    # Create binary leaning variable: negative = left leaning (True), positive/zero = right/center leaning (False)
    df["is_left_leaning"] = df["leaning_score_users"] < 0

    print(f"Political leaning distribution:")
    print(
        f"  Left leaning domains: {df['is_left_leaning'].sum()} ({df['is_left_leaning'].mean() * 100:.1f}%)"
    )
    print(
        f"  Right/center leaning domains: {(~df['is_left_leaning']).sum()} ({(~df['is_left_leaning']).mean() * 100:.1f}%)"
    )

    return df


def load_domain_ratings(filepath):
    """Load and process domain quality ratings."""
    print(f"Loading domain ratings from {filepath}")

    with gzip.open(filepath, "rt") as f:
        df = pd.read_csv(f)

    print(f"Loaded {len(df)} domains with quality ratings")
    print(f"Columns: {list(df.columns)}")

    # Clean domain names
    df["domain"] = df["domain"].str.strip()

    # Keep only pc1 and rename it to domain_quality
    df = df[["domain", "pc1"]].copy()
    df = df.rename(columns={"pc1": "domain_quality"})

    print(f"Domain quality distribution:")
    quality_stats = df["domain_quality"].describe()
    print(f"  Mean: {quality_stats['mean']:.3f}")
    print(f"  Std:  {quality_stats['std']:.3f}")
    print(f"  Min:  {quality_stats['min']:.3f}")
    print(f"  Max:  {quality_stats['max']:.3f}")

    return df


def load_citations(filepath):
    """Load citations data."""
    print(f"Loading citations from {filepath}")

    citations = pd.read_parquet(filepath)

    print(f"Loaded {len(citations)} citations")
    print(f"Columns: {list(citations.columns)}")

    return citations


def enrich_citations(citations, political_leaning, domain_ratings):
    """Merge citations with political leaning and domain quality data."""
    print("Enriching citations with political leaning and domain quality metrics...")

    # Start with citations
    enriched = citations.copy()

    # Merge with political leaning on base domain
    print(f"Merging with political leaning data...")
    enriched = enriched.merge(
        political_leaning, on="domain", how="left", suffixes=("", "_pol")
    )

    # Count matches
    pol_matches = enriched["leaning_score_users"].notna().sum()
    print(
        f"Political leaning matches: {pol_matches:,} / {len(enriched):,} ({pol_matches / len(enriched) * 100:.1f}%)"
    )

    # Merge with domain ratings on base domain
    print(f"Merging with domain quality ratings...")
    enriched = enriched.merge(
        domain_ratings, on="domain", how="left", suffixes=("", "_qual")
    )

    # Count matches
    qual_matches = enriched["domain_quality"].notna().sum()
    print(
        f"Domain quality matches: {qual_matches:,} / {len(enriched):,} ({qual_matches / len(enriched) * 100:.1f}%)"
    )

    # Combined matches
    both_matches = (
        (enriched["leaning_score_users"].notna()) & (enriched["domain_quality"].notna())
    ).sum()
    print(
        f"Both metrics available: {both_matches:,} / {len(enriched):,} ({both_matches / len(enriched) * 100:.1f}%)"
    )

    return enriched


def generate_summary_stats(enriched_citations):
    """Generate summary statistics for the enriched dataset."""
    print("\n=== ENRICHED CITATIONS SUMMARY ===")

    total_citations = len(enriched_citations)
    print(f"Total citations: {total_citations:,}")

    # Political leaning coverage
    pol_coverage = enriched_citations["leaning_score_users"].notna().sum()
    print(
        f"Political leaning coverage: {pol_coverage:,} ({pol_coverage / total_citations * 100:.1f}%)"
    )

    # Domain quality coverage
    qual_coverage = enriched_citations["domain_quality"].notna().sum()
    print(
        f"Domain quality coverage: {qual_coverage:,} ({qual_coverage / total_citations * 100:.1f}%)"
    )

    # Combined coverage
    combined_coverage = (
        (enriched_citations["leaning_score_users"].notna())
        & (enriched_citations["domain_quality"].notna())
    ).sum()
    print(
        f"Combined coverage: {combined_coverage:,} ({combined_coverage / total_citations * 100:.1f}%)"
    )

    # Top domains by citation count
    print(f"\nTop 10 cited domains:")
    top_domains = enriched_citations["domain"].value_counts().head(10)
    for domain, count in top_domains.items():
        print(f"  {domain}: {count:,} citations")

    # Political leaning distribution
    if pol_coverage > 0:
        print(f"\nPolitical leaning distribution (users score):")
        pol_stats = enriched_citations["leaning_score_users"].describe()
        print(f"  Mean: {pol_stats['mean']:.3f}")
        print(f"  Std:  {pol_stats['std']:.3f}")
        print(f"  Min:  {pol_stats['min']:.3f}")
        print(f"  Max:  {pol_stats['max']:.3f}")

        # Binary leaning distribution
        left_leaning_count = enriched_citations["is_left_leaning"].sum()
        left_leaning_pct = left_leaning_count / pol_coverage * 100
        print(
            f"  Left leaning citations: {left_leaning_count:,} ({left_leaning_pct:.1f}% of citations with leaning data)"
        )

    # Domain quality distribution
    if qual_coverage > 0:
        print(f"\nDomain quality distribution:")
        qual_stats = enriched_citations["domain_quality"].describe()
        print(f"  Mean: {qual_stats['mean']:.3f}")
        print(f"  Std:  {qual_stats['std']:.3f}")
        print(f"  Min:  {qual_stats['min']:.3f}")
        print(f"  Max:  {qual_stats['max']:.3f}")


def main():
    # Get paths from Snakemake
    citations_path = snakemake.input.citations
    political_path = snakemake.input.political_leaning
    ratings_path = snakemake.input.domain_ratings
    output_path = snakemake.output[0]

    # Load all datasets
    citations = load_citations(citations_path)
    political_leaning = load_political_leaning(political_path)
    domain_ratings = load_domain_ratings(ratings_path)

    # Enrich citations
    enriched_citations = enrich_citations(citations, political_leaning, domain_ratings)

    # Generate summary statistics
    generate_summary_stats(enriched_citations)

    # Save enriched dataset
    print(f"\nSaving enriched citations to {output_path}")
    enriched_citations.to_parquet(output_path, index=False)

    print(f"✅ Enriched citations saved successfully!")
    print(
        f"Final dataset: {len(enriched_citations):,} rows, {len(enriched_citations.columns)} columns"
    )


if __name__ == "__main__":
    main()
