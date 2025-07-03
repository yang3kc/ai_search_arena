#!/usr/bin/env python3
"""
Enrich citations with domain quality scores.

This script adds domain quality metrics to citations data:
- domain_quality_score (continuous): from lin_domain_ratings.csv.gz (pc1 column)
- domain_quality (categorical): high_quality/low_quality/unknown (threshold: 0.5)
"""

import pandas as pd
import gzip


def load_domain_quality_data(filepath):
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

    # Create categorical domain quality variable using 0.5 threshold
    df["domain_quality"] = "unknown_quality"
    df.loc[df["domain_quality_score"] >= 0.5, "domain_quality"] = "high_quality"
    df.loc[df["domain_quality_score"] < 0.5, "domain_quality"] = "low_quality"
    df.loc[df["domain_quality_score"].isna(), "domain_quality"] = "unknown_quality"

    print("Domain quality distribution:")
    quality_stats = df["domain_quality_score"].describe()
    print(f"  Score Mean: {quality_stats['mean']:.3f}")
    print(f"  Score Std:  {quality_stats['std']:.3f}")
    print(f"  Score Min:  {quality_stats['min']:.3f}")
    print(f"  Score Max:  {quality_stats['max']:.3f}")

    print("Domain quality categories:")
    quality_counts = df["domain_quality"].value_counts()
    for quality, count in quality_counts.items():
        print(f"  {quality}: {count} ({count / len(df) * 100:.1f}%)")

    return df


def enrich_with_domain_quality(domains_df, domain_quality_data):
    """Merge domains with domain quality data."""
    print("Enriching domains with domain quality data...")

    # Merge with domain ratings on base domain
    enriched = domains_df.merge(
        domain_quality_data, on="domain", how="left", suffixes=("", "_qual")
    )
    enriched["domain_quality"].fillna("unknown_quality", inplace=True)

    # Count matches
    qual_matches = enriched["domain_quality_score"].notna().sum()
    total_citations = len(enriched)
    qual_domains_covered = enriched[enriched["domain_quality_score"].notna()][
        "domain"
    ].nunique()
    total_unique_domains = enriched["domain"].nunique()

    print(
        f"Domain quality matches: {qual_matches:,} / {total_citations:,} ({qual_matches / total_citations * 100:.1f}%)"
    )
    print(
        f"Domain quality domain coverage: {qual_domains_covered:,} / {total_unique_domains:,} unique domains ({qual_domains_covered / total_unique_domains * 100:.1f}%)"
    )

    return enriched
