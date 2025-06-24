#!/usr/bin/env python3
"""
News Citations Extraction Script.

This script extracts news citations from the integrated dataset for
political bias and source quality analysis.
"""

import pandas as pd
from pathlib import Path


def load_integrated_data(data_path):
    """Load the integrated citation dataset."""
    print(f"Loading integrated citations from {data_path}")
    data = pd.read_parquet(data_path)
    print(f"Loaded {len(data):,} citations with {len(data.columns)} columns")
    return data


def extract_news_citations(data):
    """Extract citations from news domains."""
    print("\n=== EXTRACTING NEWS CITATIONS ===")

    # Filter for news domain citations
    news_citations = data[data["domain_classification"] == "news"].copy()

    print(
        f"Found {len(news_citations):,} news citations ({len(news_citations) / len(data) * 100:.1f}% of total)"
    )

    # Basic statistics about news citations
    print(f"\nNews citations statistics:")
    print(f"  Unique news domains: {news_citations['domain'].nunique():,}")
    print(
        f"  Date range: {pd.to_datetime(news_citations['timestamp']).min()} to {pd.to_datetime(news_citations['timestamp']).max()}"
    )

    # Top news domains
    print(f"\nTop 10 most cited news domains:")
    top_news_domains = news_citations["domain"].value_counts().head(10)
    for domain, count in top_news_domains.items():
        pct = count / len(news_citations) * 100
        print(f"  {domain}: {count:,} citations ({pct:.1f}%)")

    return news_citations


def analyze_political_bias_coverage(news_data):
    """Analyze political bias score coverage in news citations."""
    print("\n=== POLITICAL BIAS COVERAGE ANALYSIS ===")

    # Check for missing political bias scores
    total_news = len(news_data)
    has_bias_score = news_data["political_leaning_score"].notna().sum()
    has_bias_label = news_data["political_leaning"].notna().sum()

    print(f"Political bias data coverage:")
    print(f"  Total news citations: {total_news:,}")
    print(
        f"  With political leaning score: {has_bias_score:,} ({has_bias_score / total_news * 100:.1f}%)"
    )
    print(
        f"  With political leaning label: {has_bias_label:,} ({has_bias_label / total_news * 100:.1f}%)"
    )

    # Political leaning distribution
    if has_bias_label > 0:
        print(f"\nPolitical leaning distribution:")
        bias_dist = news_data["political_leaning"].value_counts()
        for leaning, count in bias_dist.items():
            pct = count / has_bias_label * 100
            print(f"  {leaning}: {count:,} citations ({pct:.1f}%)")

    # Score distribution statistics
    if has_bias_score > 0:
        bias_scores = news_data["political_leaning_score"].dropna()
        print(f"\nPolitical leaning score statistics:")
        print(f"  Mean: {bias_scores.mean():.3f}")
        print(f"  Median: {bias_scores.median():.3f}")
        print(f"  Std: {bias_scores.std():.3f}")
        print(f"  Range: {bias_scores.min():.3f} to {bias_scores.max():.3f}")

    return {
        "total_news": total_news,
        "has_bias_score": has_bias_score,
        "has_bias_label": has_bias_label,
        "bias_coverage_pct": has_bias_score / total_news * 100 if total_news > 0 else 0,
    }


def analyze_quality_coverage(news_data):
    """Analyze source quality score coverage in news citations."""
    print("\n=== SOURCE QUALITY COVERAGE ANALYSIS ===")

    # Check for missing quality scores
    total_news = len(news_data)
    has_quality_score = news_data["domain_quality_score"].notna().sum()
    has_quality_label = news_data["domain_quality"].notna().sum()

    print(f"Source quality data coverage:")
    print(f"  Total news citations: {total_news:,}")
    print(
        f"  With quality score: {has_quality_score:,} ({has_quality_score / total_news * 100:.1f}%)"
    )
    print(
        f"  With quality label: {has_quality_label:,} ({has_quality_label / total_news * 100:.1f}%)"
    )

    # Quality label distribution
    if has_quality_label > 0:
        print(f"\nQuality label distribution:")
        quality_dist = news_data["domain_quality"].value_counts()
        for quality, count in quality_dist.items():
            pct = count / has_quality_label * 100
            print(f"  {quality}: {count:,} citations ({pct:.1f}%)")

    # Score distribution statistics
    if has_quality_score > 0:
        quality_scores = news_data["domain_quality_score"].dropna()
        print(f"\nQuality score statistics:")
        print(f"  Mean: {quality_scores.mean():.3f}")
        print(f"  Median: {quality_scores.median():.3f}")
        print(f"  Std: {quality_scores.std():.3f}")
        print(f"  Range: {quality_scores.min():.3f} to {quality_scores.max():.3f}")

    return {
        "total_news": total_news,
        "has_quality_score": has_quality_score,
        "has_quality_label": has_quality_label,
        "quality_coverage_pct": has_quality_score / total_news * 100
        if total_news > 0
        else 0,
    }


def analyze_model_news_patterns(news_data):
    """Analyze news citation patterns by AI model."""
    print("\n=== MODEL NEWS CITATION PATTERNS ===")

    if "model_name_raw" not in news_data.columns:
        print("Model information not available")
        return {}

    # Citations per model
    model_citations = news_data["model_name_raw"].value_counts()
    print(f"News citations by model:")
    for model, count in model_citations.items():
        pct = count / len(news_data) * 100
        print(f"  {model}: {count:,} citations ({pct:.1f}%)")

    # Model families
    if "model_family" in news_data.columns:
        family_citations = news_data["model_family"].value_counts()
        print(f"\nNews citations by model family:")
        for family, count in family_citations.items():
            pct = count / len(news_data) * 100
            print(f"  {family}: {count:,} citations ({pct:.1f}%)")

    return {
        "model_citations": model_citations.to_dict(),
        "family_citations": family_citations.to_dict()
        if "model_family" in news_data.columns
        else {},
    }


def create_bias_quality_subset(news_data):
    """Create subset with both bias and quality data for joint analysis."""
    print("\n=== CREATING BIAS + QUALITY SUBSET ===")

    # Filter for citations with both bias and quality data
    complete_data = news_data[
        news_data["political_leaning_score"].notna()
        & news_data["domain_quality_score"].notna()
    ].copy()

    print(f"News citations with both bias and quality data: {len(complete_data):,}")
    print(f"Percentage of total news: {len(complete_data) / len(news_data) * 100:.1f}%")

    if len(complete_data) > 0:
        print(
            f"Unique domains with complete data: {complete_data['domain'].nunique():,}"
        )

        # Model distribution in complete dataset
        if "model_name_raw" in complete_data.columns:
            print(f"\nModel representation in complete dataset:")
            model_dist = complete_data["model_name_raw"].value_counts()
            for model, count in model_dist.head(5).items():
                pct = count / len(complete_data) * 100
                print(f"  {model}: {count:,} citations ({pct:.1f}%)")

    return complete_data


def generate_summary_report(
    news_data, bias_stats, quality_stats, model_patterns, complete_data
):
    """Generate summary statistics for news citations."""
    print("\n=== GENERATING SUMMARY REPORT ===")

    summary = {
        "extraction_summary": {
            "total_citations": len(news_data),
            "unique_domains": news_data["domain"].nunique(),
            "date_range": {
                "start": pd.to_datetime(news_data["timestamp"]).min().isoformat(),
                "end": pd.to_datetime(news_data["timestamp"]).max().isoformat(),
            },
        },
        "bias_analysis_readiness": {
            "citations_with_bias_score": bias_stats["has_bias_score"],
            "bias_coverage_percentage": bias_stats["bias_coverage_pct"],
            "ready_for_analysis": bias_stats["has_bias_score"]
            > 1000,  # Threshold for meaningful analysis
        },
        "quality_analysis_readiness": {
            "citations_with_quality_score": quality_stats["has_quality_score"],
            "quality_coverage_percentage": quality_stats["quality_coverage_pct"],
            "ready_for_analysis": quality_stats["has_quality_score"] > 1000,
        },
        "joint_analysis_readiness": {
            "citations_with_both_scores": len(complete_data),
            "joint_coverage_percentage": len(complete_data) / len(news_data) * 100
            if len(news_data) > 0
            else 0,
            "ready_for_analysis": len(complete_data) > 500,
        },
        "model_patterns": model_patterns,
    }

    return summary


def main():
    """Main function for news citations extraction."""
    # Get paths from Snakemake
    input_path = snakemake.input.integrated_citations
    news_output_path = snakemake.output.news_citations
    summary_output_path = snakemake.output.extraction_summary

    # Create output directory
    output_dir = Path(news_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load integrated data
    data = load_integrated_data(input_path)

    # Extract news citations
    news_data = extract_news_citations(data)

    # Analyze coverage and readiness for different analyses
    bias_stats = analyze_political_bias_coverage(news_data)
    quality_stats = analyze_quality_coverage(news_data)
    model_patterns = analyze_model_news_patterns(news_data)

    # Create complete dataset for joint analysis
    complete_data = create_bias_quality_subset(news_data)

    # Generate summary report
    summary = generate_summary_report(
        news_data, bias_stats, quality_stats, model_patterns, complete_data
    )

    # Save news citations dataset
    print(f"\nSaving news citations to {news_output_path}")
    news_data.to_parquet(news_output_path, index=False)

    # Save summary as JSON
    import json

    print(f"Saving extraction summary to {summary_output_path}")
    with open(summary_output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✅ News citations extraction completed!")
    print(f"News citations: {news_output_path}")
    print(f"Extraction summary: {summary_output_path}")
    print(
        f"Ready for bias analysis: {summary['bias_analysis_readiness']['ready_for_analysis']}"
    )
    print(
        f"Ready for quality analysis: {summary['quality_analysis_readiness']['ready_for_analysis']}"
    )
    print(
        f"Ready for joint analysis: {summary['joint_analysis_readiness']['ready_for_analysis']}"
    )


if __name__ == "__main__":
    main()
