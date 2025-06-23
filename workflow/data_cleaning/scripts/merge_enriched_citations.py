#!/usr/bin/env python3
"""
Merge enriched domains back to citations.

This script takes the enriched domains dataset and merges it back
with the original citations to create the final enriched citations dataset.
"""

import pandas as pd


def merge_enriched_domains_to_citations(citations_df, enriched_domains_df):
    """Merge enriched domains data back to citations."""
    print("Merging enriched domains back to citations...")
    
    # Remove citation_count column from domains (not needed in final citations)
    domain_signals = enriched_domains_df.drop(columns=['citation_count'])
    
    # Merge citations with enriched domains
    enriched_citations = citations_df.merge(
        domain_signals,
        on='domain',
        how='left'
    )
    
    print(f"Merge completed: {len(enriched_citations):,} citations with {len(enriched_citations.columns)} columns")
    
    # Verify no citations were lost
    if len(enriched_citations) != len(citations_df):
        print(f"⚠️  Warning: Citation count changed from {len(citations_df)} to {len(enriched_citations)}")
    else:
        print("✅ All citations preserved in merge")
    
    return enriched_citations


def generate_final_summary_stats(enriched_citations):
    """Generate final summary statistics for enriched citations."""
    print("\n=== FINAL ENRICHED CITATIONS SUMMARY ===")
    
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
        
        if "political_leaning" in enriched_citations.columns:
            print(f"Political leaning categories:")
            leaning_counts = enriched_citations["political_leaning"].value_counts()
            for leaning, count in leaning_counts.items():
                pct = count / total_citations * 100
                print(f"  {leaning}: {count:,} ({pct:.1f}% of all citations)")
    
    # Domain quality coverage
    if "domain_quality_score" in enriched_citations.columns:
        qual_coverage = enriched_citations["domain_quality_score"].notna().sum()
        qual_domains_covered = enriched_citations[enriched_citations["domain_quality_score"].notna()]["domain"].nunique()
        print(f"\nDomain quality coverage: {qual_coverage:,} citations ({qual_coverage / total_citations * 100:.1f}%)")
        print(f"Domain quality domain coverage: {qual_domains_covered:,} / {total_unique_domains:,} unique domains ({qual_domains_covered / total_unique_domains * 100:.1f}%)")
        
        if "domain_quality" in enriched_citations.columns:
            print(f"Domain quality categories:")
            quality_counts = enriched_citations["domain_quality"].value_counts()
            for quality, count in quality_counts.items():
                pct = count / total_citations * 100
                print(f"  {quality}: {count:,} ({pct:.1f}% of all citations)")
    
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


def main():
    """Main function for merging enriched domains back to citations."""
    # Get paths from Snakemake
    citations_path = snakemake.input.citations
    enriched_domains_path = snakemake.input.enriched_domains
    output_path = snakemake.output[0]
    
    # Load data
    print(f"Loading citations from {citations_path}")
    citations = pd.read_parquet(citations_path)
    print(f"Loaded {len(citations):,} citations")
    
    print(f"\nLoading enriched domains from {enriched_domains_path}")
    enriched_domains = pd.read_parquet(enriched_domains_path)
    print(f"Loaded {len(enriched_domains):,} enriched domains")
    
    # Merge enriched domains back to citations
    enriched_citations = merge_enriched_domains_to_citations(citations, enriched_domains)
    
    # Generate final summary statistics
    generate_final_summary_stats(enriched_citations)
    
    # Save final enriched citations
    print(f"\nSaving enriched citations to {output_path}")
    enriched_citations.to_parquet(output_path, index=False)
    
    print(f"\n✅ Final citations enrichment completed!")
    print(f"Final dataset: {len(enriched_citations):,} rows, {len(enriched_citations.columns)} columns")


if __name__ == "__main__":
    main()