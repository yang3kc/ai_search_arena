#!/usr/bin/env python3
"""
Enrich unique domains with all available signals.

This script enriches the unique domains dataset (not the full citations)
with all available signals for efficiency. The enriched domains can then
be merged back with citations as needed.
"""

import pandas as pd
import sys
import os

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import individual signal enrichment modules
import enrich_political_leaning
import enrich_domain_quality
import enrich_domain_classification


def generate_domains_summary_stats(enriched_domains):
    """Generate summary statistics for the enriched domains dataset."""
    print("\n=== ENRICHED DOMAINS SUMMARY ===")

    total_domains = len(enriched_domains)
    total_citations = enriched_domains["citation_count"].sum()
    print(f"Total unique domains: {total_domains:,}")
    print(f"Total citations represented: {total_citations:,}")

    # Political leaning coverage
    if "political_leaning_score" in enriched_domains.columns:
        pol_domains_covered = enriched_domains["political_leaning_score"].notna().sum()
        pol_citations_covered = enriched_domains[
            enriched_domains["political_leaning_score"].notna()
        ]["citation_count"].sum()

        print("\nPolitical leaning coverage:")
        print(
            f"  Domains: {pol_domains_covered:,} / {total_domains:,} ({pol_domains_covered / total_domains * 100:.1f}%)"
        )
        print(
            f"  Citations: {pol_citations_covered:,} / {total_citations:,} ({pol_citations_covered / total_citations * 100:.1f}%)"
        )

        if "political_leaning" in enriched_domains.columns:
            print("  Political leaning categories (by citation count):")
            for category in ["left_leaning", "right_leaning", "unknown_leaning"]:
                cat_citations = enriched_domains[
                    enriched_domains["political_leaning"] == category
                ]["citation_count"].sum()
                cat_pct = cat_citations / total_citations * 100
                cat_domains = (enriched_domains["political_leaning"] == category).sum()
                print(
                    f"    {category}: {cat_citations:,} citations ({cat_pct:.1f}%) from {cat_domains:,} domains"
                )

    # Domain quality coverage
    if "domain_quality_score" in enriched_domains.columns:
        qual_domains_covered = enriched_domains["domain_quality_score"].notna().sum()
        qual_citations_covered = enriched_domains[
            enriched_domains["domain_quality_score"].notna()
        ]["citation_count"].sum()

        print("\nDomain quality coverage:")
        print(
            f"  Domains: {qual_domains_covered:,} / {total_domains:,} ({qual_domains_covered / total_domains * 100:.1f}%)"
        )
        print(
            f"  Citations: {qual_citations_covered:,} / {total_citations:,} ({qual_citations_covered / total_citations * 100:.1f}%)"
        )

        if "domain_quality" in enriched_domains.columns:
            print("  Domain quality categories (by citation count):")
            for category in ["high_quality", "low_quality", "unknown_quality"]:
                cat_citations = enriched_domains[
                    enriched_domains["domain_quality"] == category
                ]["citation_count"].sum()
                cat_pct = cat_citations / total_citations * 100
                cat_domains = (enriched_domains["domain_quality"] == category).sum()
                print(
                    f"    {category}: {cat_citations:,} citations ({cat_pct:.1f}%) from {cat_domains:,} domains"
                )

    # Domain classification coverage
    if "domain_classification" in enriched_domains.columns:
        print("\nDomain classification coverage:")
        print(
            f"  All domains classified: {total_domains:,} / {total_domains:,} (100.0%)"
        )
        print(
            f"  All citations classified: {total_citations:,} / {total_citations:,} (100.0%)"
        )

        print("  Domain classification categories (by citation count):")
        class_counts = enriched_domains["domain_classification"].value_counts()
        for category in class_counts.index:
            cat_citations = enriched_domains[
                enriched_domains["domain_classification"] == category
            ]["citation_count"].sum()
            cat_pct = cat_citations / total_citations * 100
            cat_domains = (enriched_domains["domain_classification"] == category).sum()
            print(
                f"    {category}: {cat_citations:,} citations ({cat_pct:.1f}%) from {cat_domains:,} domains"
            )

    # Combined coverage
    if (
        "political_leaning_score" in enriched_domains.columns
        and "domain_quality_score" in enriched_domains.columns
    ):
        combined_domains = (
            (enriched_domains["political_leaning_score"].notna())
            & (enriched_domains["domain_quality_score"].notna())
        ).sum()
        combined_citations = enriched_domains[
            (enriched_domains["political_leaning_score"].notna())
            & (enriched_domains["domain_quality_score"].notna())
        ]["citation_count"].sum()

        print("\nCombined signal coverage:")
        print(
            f"  Domains: {combined_domains:,} / {total_domains:,} ({combined_domains / total_domains * 100:.1f}%)"
        )
        print(
            f"  Citations: {combined_citations:,} / {total_citations:,} ({combined_citations / total_citations * 100:.1f}%)"
        )

    # Top domains by citation count
    print("\nTop 10 cited domains:")
    top_domains = enriched_domains.head(10)
    for _, row in top_domains.iterrows():
        domain_info = f"  {row['domain']}: {row['citation_count']:,} citations"

        # Add signal info if available
        signals = []
        if (
            "political_leaning" in row
            and pd.notna(row["political_leaning"])
            and row["political_leaning"] != "unknown_leaning"
        ):
            signals.append(f"pol:{row['political_leaning']}")
        if (
            "domain_quality" in row
            and pd.notna(row["domain_quality"])
            and row["domain_quality"] != "unknown_quality"
        ):
            signals.append(f"qual:{row['domain_quality']}")
        if "domain_classification" in row and pd.notna(row["domain_classification"]):
            signals.append(f"class:{row['domain_classification']}")

        if signals:
            domain_info += f" [{', '.join(signals)}]"

        print(domain_info)


def main():
    """Main function for enriching unique domains with all signals."""
    # Get paths from Snakemake
    domains_path = snakemake.input.domains
    political_leaning_path = snakemake.input.political_leaning
    domain_ratings_path = snakemake.input.domain_ratings
    manual_classification_path = snakemake.input.manual_classification
    news_domains_path = snakemake.input.news_domains
    output_path = snakemake.output[0]

    # Load unique domains data
    print(f"Loading domains from {domains_path}")
    domains = pd.read_parquet(domains_path)
    print(
        f"Loaded {len(domains):,} unique domains representing {domains['citation_count'].sum():,} citations"
    )

    # Start with base domains
    enriched = domains.copy()

    # === Signal 1: Political Leaning ===
    print("\n" + "=" * 50)
    print("ENRICHING WITH POLITICAL LEANING SIGNAL")
    print("=" * 50)

    political_leaning_data = enrich_political_leaning.load_political_leaning_data(
        political_leaning_path
    )
    enriched = enrich_political_leaning.enrich_with_political_leaning(
        enriched, political_leaning_data
    )

    # === Signal 2: Domain Quality ===
    print("\n" + "=" * 50)
    print("ENRICHING WITH DOMAIN QUALITY SIGNAL")
    print("=" * 50)

    domain_quality_data = enrich_domain_quality.load_domain_quality_data(
        domain_ratings_path
    )
    enriched = enrich_domain_quality.enrich_with_domain_quality(
        enriched, domain_quality_data
    )

    # === Signal 3: Domain Classification ===
    print("\n" + "=" * 50)
    print("ENRICHING WITH DOMAIN CLASSIFICATION SIGNAL")
    print("=" * 50)

    manual_classification_data = (
        enrich_domain_classification.load_manual_classification_data(
            manual_classification_path
        )
    )
    news_domains_set = enrich_domain_classification.load_news_domains_data(
        news_domains_path
    )
    enriched = enrich_domain_classification.enrich_with_domain_classification(
        enriched, manual_classification_data, news_domains_set
    )

    # === Future signals can be added here ===

    # Generate comprehensive summary statistics
    generate_domains_summary_stats(enriched)

    # Save enriched domains dataset
    print(f"\nSaving enriched domains to {output_path}")
    enriched.to_parquet(output_path, index=False)

    print("\n✅ Domain enrichment completed successfully!")
    print(f"Final dataset: {len(enriched):,} domains, {len(enriched.columns)} columns")
    print("Signals added: political_leaning, domain_quality, domain_classification")


if __name__ == "__main__":
    main()
