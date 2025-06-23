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


def load_domain_ratings(filepath):
    """Load and process domain quality ratings."""
    print(f"Loading domain ratings from {filepath}")

    with gzip.open(filepath, "rt") as f:
        df = pd.read_csv(f)

    print(f"Loaded {len(df)} domains with quality ratings")
    print(f"Columns: {list(df.columns)}")

    # Clean domain names
    df["domain"] = df["domain"].str.strip()

    # Keep only pc1 and rename it to domain_quality_score
    df = df[["domain", "pc1"]].copy()
    df = df.rename(columns={"pc1": "domain_quality_score"})

    # Create categorical domain quality variable using 0.7 threshold
    df["domain_quality"] = "unknown"
    df.loc[df["domain_quality_score"] >= 0.7, "domain_quality"] = "high_quality"
    df.loc[df["domain_quality_score"] < 0.7, "domain_quality"] = "low_quality"
    df.loc[df["domain_quality_score"].isna(), "domain_quality"] = "unknown"

    print(f"Domain quality distribution:")
    quality_stats = df["domain_quality_score"].describe()
    print(f"  Score Mean: {quality_stats['mean']:.3f}")
    print(f"  Score Std:  {quality_stats['std']:.3f}")
    print(f"  Score Min:  {quality_stats['min']:.3f}")
    print(f"  Score Max:  {quality_stats['max']:.3f}")
    
    print(f"Domain quality categories:")
    quality_counts = df["domain_quality"].value_counts()
    for quality, count in quality_counts.items():
        print(f"  {quality}: {count} ({count / len(df) * 100:.1f}%)")

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
    pol_matches = enriched["political_leaning_score"].notna().sum()
    print(
        f"Political leaning matches: {pol_matches:,} / {len(enriched):,} ({pol_matches / len(enriched) * 100:.1f}%)"
    )

    # Merge with domain ratings on base domain
    print(f"Merging with domain quality ratings...")
    enriched = enriched.merge(
        domain_ratings, on="domain", how="left", suffixes=("", "_qual")
    )

    # Count matches
    qual_matches = enriched["domain_quality_score"].notna().sum()
    print(
        f"Domain quality matches: {qual_matches:,} / {len(enriched):,} ({qual_matches / len(enriched) * 100:.1f}%)"
    )

    # Combined matches
    both_matches = (
        (enriched["political_leaning_score"].notna()) & (enriched["domain_quality_score"].notna())
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
    pol_coverage = enriched_citations["political_leaning_score"].notna().sum()
    pol_domains_covered = enriched_citations[enriched_citations["political_leaning_score"].notna()]["domain"].nunique()
    total_unique_domains = enriched_citations["domain"].nunique()
    print(
        f"Political leaning coverage: {pol_coverage:,} citations ({pol_coverage / total_citations * 100:.1f}%)"
    )
    print(
        f"Political leaning domain coverage: {pol_domains_covered:,} / {total_unique_domains:,} unique domains ({pol_domains_covered / total_unique_domains * 100:.1f}%)"
    )

    # Domain quality coverage
    qual_coverage = enriched_citations["domain_quality_score"].notna().sum()
    qual_domains_covered = enriched_citations[enriched_citations["domain_quality_score"].notna()]["domain"].nunique()
    print(
        f"Domain quality coverage: {qual_coverage:,} citations ({qual_coverage / total_citations * 100:.1f}%)"
    )
    print(
        f"Domain quality domain coverage: {qual_domains_covered:,} / {total_unique_domains:,} unique domains ({qual_domains_covered / total_unique_domains * 100:.1f}%)"
    )

    # Combined coverage
    combined_coverage = (
        (enriched_citations["political_leaning_score"].notna())
        & (enriched_citations["domain_quality_score"].notna())
    ).sum()
    combined_domains_covered = enriched_citations[
        (enriched_citations["political_leaning_score"].notna()) & 
        (enriched_citations["domain_quality_score"].notna())
    ]["domain"].nunique()
    print(
        f"Combined coverage: {combined_coverage:,} citations ({combined_coverage / total_citations * 100:.1f}%)"
    )
    print(
        f"Combined domain coverage: {combined_domains_covered:,} / {total_unique_domains:,} unique domains ({combined_domains_covered / total_unique_domains * 100:.1f}%)"
    )

    # Top domains by citation count
    print(f"\nTop 10 cited domains:")
    top_domains = enriched_citations["domain"].value_counts().head(10)
    for domain, count in top_domains.items():
        print(f"  {domain}: {count:,} citations")

    # Political leaning distribution
    if pol_coverage > 0:
        print(f"\nPolitical leaning distribution:")
        pol_stats = enriched_citations["political_leaning_score"].describe()
        print(f"  Score Mean: {pol_stats['mean']:.3f}")
        print(f"  Score Std:  {pol_stats['std']:.3f}")
        print(f"  Score Min:  {pol_stats['min']:.3f}")
        print(f"  Score Max:  {pol_stats['max']:.3f}")

        # Categorical leaning distribution
        print(f"  Political leaning categories:")
        leaning_counts = enriched_citations["political_leaning"].value_counts()
        for leaning, count in leaning_counts.items():
            pct = count / total_citations * 100
            print(f"    {leaning}: {count:,} ({pct:.1f}% of all citations)")

    # Domain quality distribution
    if qual_coverage > 0:
        print(f"\nDomain quality distribution:")
        qual_stats = enriched_citations["domain_quality_score"].describe()
        print(f"  Score Mean: {qual_stats['mean']:.3f}")
        print(f"  Score Std:  {qual_stats['std']:.3f}")
        print(f"  Score Min:  {qual_stats['min']:.3f}")
        print(f"  Score Max:  {qual_stats['max']:.3f}")

        # Categorical quality distribution
        print(f"  Domain quality categories:")
        quality_counts = enriched_citations["domain_quality"].value_counts()
        for quality, count in quality_counts.items():
            pct = count / total_citations * 100
            print(f"    {quality}: {count:,} ({pct:.1f}% of all citations)")


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
