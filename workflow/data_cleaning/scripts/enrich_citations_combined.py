#!/usr/bin/env python3
"""
Orchestrator script to enrich citations with multiple signals.

This script combines all individual signal enrichments:
1. Political leaning scores and categories
2. Domain quality scores and categories

Additional signals can be easily added by importing their modules
and calling their enrichment functions.
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import individual signal enrichment modules
import enrich_political_leaning
import enrich_domain_quality


def generate_combined_summary_stats(enriched_citations):
    """Generate comprehensive summary statistics for the enriched dataset."""
    print("\n=== ENRICHED CITATIONS SUMMARY ===")
    
    total_citations = len(enriched_citations)
    total_unique_domains = enriched_citations["domain"].nunique()
    print(f"Total citations: {total_citations:,}")
    print(f"Total unique domains: {total_unique_domains:,}")
    
    # Political leaning coverage
    if "political_leaning_score" in enriched_citations.columns:
        pol_coverage = enriched_citations["political_leaning_score"].notna().sum()
        pol_domains_covered = enriched_citations[enriched_citations["political_leaning_score"].notna()]["domain"].nunique()
        print(f"\nPolitical leaning coverage: {pol_coverage:,} citations ({pol_coverage / total_citations * 100:.1f}%)")
        print(f"Political leaning domain coverage: {pol_domains_covered:,} / {total_unique_domains:,} unique domains ({pol_domains_covered / total_unique_domains * 100:.1f}%)")
    
    # Domain quality coverage
    if "domain_quality_score" in enriched_citations.columns:
        qual_coverage = enriched_citations["domain_quality_score"].notna().sum()
        qual_domains_covered = enriched_citations[enriched_citations["domain_quality_score"].notna()]["domain"].nunique()
        print(f"\nDomain quality coverage: {qual_coverage:,} citations ({qual_coverage / total_citations * 100:.1f}%)")
        print(f"Domain quality domain coverage: {qual_domains_covered:,} / {total_unique_domains:,} unique domains ({qual_domains_covered / total_unique_domains * 100:.1f}%)")
    
    # Combined coverage
    if ("political_leaning_score" in enriched_citations.columns and 
        "domain_quality_score" in enriched_citations.columns):
        combined_coverage = (
            (enriched_citations["political_leaning_score"].notna()) &
            (enriched_citations["domain_quality_score"].notna())
        ).sum()
        combined_domains_covered = enriched_citations[
            (enriched_citations["political_leaning_score"].notna()) & 
            (enriched_citations["domain_quality_score"].notna())
        ]["domain"].nunique()
        print(f"\nCombined coverage: {combined_coverage:,} citations ({combined_coverage / total_citations * 100:.1f}%)")
        print(f"Combined domain coverage: {combined_domains_covered:,} / {total_unique_domains:,} unique domains ({combined_domains_covered / total_unique_domains * 100:.1f}%)")
    
    # Top domains by citation count
    print(f"\nTop 10 cited domains:")
    top_domains = enriched_citations["domain"].value_counts().head(10)
    for domain, count in top_domains.items():
        print(f"  {domain}: {count:,} citations")
    
    # Political leaning distribution
    if "political_leaning_score" in enriched_citations.columns:
        pol_coverage = enriched_citations["political_leaning_score"].notna().sum()
        if pol_coverage > 0:
            print(f"\nPolitical leaning distribution:")
            pol_stats = enriched_citations["political_leaning_score"].describe()
            print(f"  Score Mean: {pol_stats['mean']:.3f}")
            print(f"  Score Std:  {pol_stats['std']:.3f}")
            print(f"  Score Min:  {pol_stats['min']:.3f}")
            print(f"  Score Max:  {pol_stats['max']:.3f}")
            
            # Categorical leaning distribution
            if "political_leaning" in enriched_citations.columns:
                print(f"  Political leaning categories:")
                leaning_counts = enriched_citations["political_leaning"].value_counts()
                for leaning, count in leaning_counts.items():
                    pct = count / total_citations * 100
                    print(f"    {leaning}: {count:,} ({pct:.1f}% of all citations)")
    
    # Domain quality distribution
    if "domain_quality_score" in enriched_citations.columns:
        qual_coverage = enriched_citations["domain_quality_score"].notna().sum()
        if qual_coverage > 0:
            print(f"\nDomain quality distribution:")
            qual_stats = enriched_citations["domain_quality_score"].describe()
            print(f"  Score Mean: {qual_stats['mean']:.3f}")
            print(f"  Score Std:  {qual_stats['std']:.3f}")
            print(f"  Score Min:  {qual_stats['min']:.3f}")
            print(f"  Score Max:  {qual_stats['max']:.3f}")
            
            # Categorical quality distribution
            if "domain_quality" in enriched_citations.columns:
                print(f"  Domain quality categories:")
                quality_counts = enriched_citations["domain_quality"].value_counts()
                for quality, count in quality_counts.items():
                    pct = count / total_citations * 100
                    print(f"    {quality}: {count:,} ({pct:.1f}% of all citations)")


def main():
    """Main orchestrator function for combining all signal enrichments."""
    # Get paths from Snakemake
    citations_path = snakemake.input.citations
    political_leaning_path = snakemake.input.political_leaning
    domain_ratings_path = snakemake.input.domain_ratings
    output_path = snakemake.output[0]
    
    # Load base citations data
    print("Loading base citations data...")
    citations = pd.read_parquet(citations_path)
    print(f"Loaded {len(citations)} citations with {len(citations.columns)} columns")
    
    # Start with base citations
    enriched = citations.copy()
    
    # === Signal 1: Political Leaning ===
    print("\n" + "="*50)
    print("ENRICHING WITH POLITICAL LEANING SIGNAL")
    print("="*50)
    
    political_leaning_data = enrich_political_leaning.load_political_leaning_data(political_leaning_path)
    enriched = enrich_political_leaning.enrich_with_political_leaning(enriched, political_leaning_data)
    
    # === Signal 2: Domain Quality ===
    print("\n" + "="*50)
    print("ENRICHING WITH DOMAIN QUALITY SIGNAL")
    print("="*50)
    
    domain_quality_data = enrich_domain_quality.load_domain_quality_data(domain_ratings_path)
    enriched = enrich_domain_quality.enrich_with_domain_quality(enriched, domain_quality_data)
    
    # === Future signals can be added here ===
    # Signal 3: Add new signal enrichment
    # Signal 4: Add another signal enrichment
    
    # Generate comprehensive summary statistics
    generate_combined_summary_stats(enriched)
    
    # Save final enriched dataset
    print(f"\nSaving enriched citations to {output_path}")
    enriched.to_parquet(output_path, index=False)
    
    print(f"\n✅ Citations enrichment completed successfully!")
    print(f"Final dataset: {len(enriched):,} rows, {len(enriched.columns)} columns")
    print(f"Added signals: political_leaning, domain_quality")


if __name__ == "__main__":
    main()