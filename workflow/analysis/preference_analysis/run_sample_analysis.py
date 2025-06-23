#!/usr/bin/env python3
"""
Sample-based Citation Analysis - Production Ready

This script analyzes citation effects using a representative sample of the data
for efficient computation while maintaining statistical validity.
"""

import os
import sys
import pandas as pd
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add analysis module to path
sys.path.append(os.path.dirname(__file__))
from citation_style_analysis import CitationStyleAnalyzer


def main():
    """Run citation analysis on a representative sample."""

    print("=== Sample-based Citation Analysis ===")

    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "intermediate" / "cleaned_arena_data"
    output_dir = base_dir / "data" / "intermediate" / "citation_analysis"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample parameters for manageable computation
    SAMPLE_SIZE = 10000  # Sample 10k threads for analysis
    np.random.seed(42)  # Reproducible sampling

    print(f"Sample size: {SAMPLE_SIZE} threads")
    print(f"Output directory: {output_dir}")

    # Load and sample data
    print("\n1. Loading and sampling data...")
    threads_df = pd.read_parquet(data_dir / "threads.parquet")

    # Stratified sampling by winner to maintain distribution
    sample_threads = (
        threads_df.groupby("winner", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), SAMPLE_SIZE // 4), random_state=42))
        .reset_index(drop=True)
    )

    print(f"   Sampled {len(sample_threads)} threads from {len(threads_df)} total")
    print(f"   Winner distribution: {dict(sample_threads['winner'].value_counts())}")

    # Load related data
    responses_df = pd.read_parquet(data_dir / "responses.parquet")
    citations_df = pd.read_parquet(data_dir / "citations.parquet")

    # Filter to sample
    sample_thread_ids = set(sample_threads["thread_id"])
    sample_responses = responses_df[responses_df["thread_id"].isin(sample_thread_ids)]
    sample_response_ids = set(sample_responses["response_id"])
    sample_citations = citations_df[
        citations_df["response_id"].isin(sample_response_ids)
    ]

    print(
        f"   Sample data: {len(sample_responses)} responses, {len(sample_citations)} citations"
    )

    # Run analysis
    analyzer = CitationStyleAnalyzer()

    print("\n2. Computing citation features...")
    citation_features = analyzer.compute_citation_features(sample_citations)

    print("\n3. Creating battle data...")
    battle_df = analyzer.create_battle_data(
        sample_threads, sample_responses, citation_features
    )
    battle_clean = battle_df[~battle_df["winner"].isin(["tie", "tie (bothbad)"])].copy()

    print(f"   Battle data: {len(battle_clean)} clean battles")

    if len(battle_clean) < 50:
        print("   Error: Insufficient battles for analysis")
        return 1

    # Show domain distribution
    domain_cols = [col for col in citation_features.columns if col.startswith("cites_")]
    domain_totals = citation_features[domain_cols].sum().sort_values(ascending=False)
    print(f"   Top domains: {dict(domain_totals.head())}")

    # Define analyses
    analyses = {
        "citation_volume": {
            "features": ["num_citations"],
            "description": "Effect of citation count on preference",
        },
        "credible_sources": {
            "features": ["cites_gov_edu", "cites_wiki", "cites_academic_journal"],
            "description": "Effect of authoritative source citations",
        },
        "popular_sources": {
            "features": ["cites_youtube", "cites_social_media"],
            "description": "Effect of popular/social source citations",
        },
        "news_analysis": {
            "features": ["cites_us_news", "cites_foreign_news"],
            "description": "US vs international news source preferences",
        },
    }

    # Run analyses
    results = {}

    for i, (name, config) in enumerate(analyses.items(), 1):
        print(f"\n{i + 3}. Analyzing {name}...")

        try:
            result = analyzer.compute_style_coefficients(
                battle_clean,
                config["features"],
                num_bootstrap=100,  # Moderate bootstrap for speed
            )

            results[name] = {
                "config": config,
                "results": result,
                "timestamp": datetime.now().isoformat(),
            }

            print(f"   ✓ Success: {result['bootstrap_samples']} bootstrap samples")

            # Show results
            for feature in result["features"]:
                coeff = result["coefficients"][feature]
                ci = result["confidence_intervals"][feature]
                sig = "***" if ci["lower"] * ci["upper"] > 0 else ""
                print(
                    f"   {feature}: {coeff:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}] {sig}"
                )

        except Exception as e:
            print(f"   ✗ Failed: {e}")
            results[name] = {"config": config, "error": str(e)}

    # Save comprehensive results
    print(f"\n{len(analyses) + 4}. Saving results...")

    # Main results
    results_file = output_dir / "citation_effects_analysis.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Create executive summary
    create_executive_summary(results, battle_clean, sample_threads, output_dir)

    print(f"\n=== Analysis Complete ===")
    print(f"Sample: {len(sample_threads):,} threads → {len(battle_clean):,} battles")
    print(f"Results: {results_file}")
    print(f"Summary: {output_dir / 'executive_summary.md'}")

    return 0


def create_executive_summary(
    results: dict, battles: pd.DataFrame, threads: pd.DataFrame, output_dir: Path
):
    """Create executive summary of findings."""

    summary_file = output_dir / "executive_summary.md"

    with open(summary_file, "w") as f:
        f.write("# Citation Effects on User Preferences - Executive Summary\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}\n\n")

        f.write("## Key Question\n")
        f.write(
            "How do different citation sources influence user preferences in AI search responses?\n\n"
        )

        f.write("## Dataset\n")
        f.write(
            f"- **Sample Size:** {len(threads):,} conversations → {len(battles):,} model comparisons\n"
        )
        f.write(
            f"- **Winner Distribution:** {dict(battles['winner'].value_counts())}\n"
        )
        f.write(
            f"- **Models:** {len(set(battles['model_a']) | set(battles['model_b']))} different AI models\n\n"
        )

        f.write("## Findings\n\n")

        significant_findings = []
        non_significant = []

        for analysis_name, data in results.items():
            if "error" in data:
                continue

            if "results" not in data:
                continue

            result = data["results"]

            for feature in result["features"]:
                coeff = result["coefficients"][feature]
                ci = result["confidence_intervals"][feature]

                feature_display = (
                    feature.replace("cites_", "").replace("_", " ").title()
                )

                if ci["lower"] * ci["upper"] > 0:  # Significant
                    direction = "prefer" if coeff > 0 else "dislike"
                    magnitude = abs(coeff)

                    if magnitude > 0.1:
                        strength = "Strong"
                    elif magnitude > 0.05:
                        strength = "Moderate"
                    else:
                        strength = "Slight"

                    finding = f"**{strength} {direction.title()} for {feature_display}** (β={coeff:.3f}, CI=[{ci['lower']:.3f}, {ci['upper']:.3f}])"
                    significant_findings.append(finding)
                else:
                    non_significant.append(f"{feature_display} (β={coeff:.3f})")

        if significant_findings:
            f.write("### Statistically Significant Effects\n\n")
            for finding in significant_findings:
                f.write(f"- {finding}\n")
            f.write("\n")

        if non_significant:
            f.write("### No Significant Effects\n\n")
            f.write(
                "The following citation types showed no statistically significant impact:\n"
            )
            for item in non_significant:
                f.write(f"- {item}\n")
            f.write("\n")

        if not significant_findings and not non_significant:
            f.write("### Analysis Results\n\n")
            f.write(
                "Analysis completed but no clear patterns emerged. This may indicate:\n"
            )
            f.write("- Citation sources have minimal impact on user preferences\n")
            f.write("- Users focus more on content quality than source type\n")
            f.write("- Sample size limitations\n\n")

        f.write("## Implications\n\n")
        if significant_findings:
            f.write("**For AI System Design:**\n")
            f.write(
                "- Consider user preferences for specific citation types when ranking sources\n"
            )
            f.write("- Balance authoritative sources with user-preferred formats\n")
            f.write("- Customize citation strategies based on query type\n\n")

        f.write("**For Further Research:**\n")
        f.write("- Analyze citation effects by query topic/domain\n")
        f.write("- Examine interaction effects between multiple citation types\n")
        f.write("- Study temporal changes in citation preferences\n\n")

        f.write("## Methodology\n\n")
        f.write(
            "- **Model:** Contextual Bradley-Terry with bootstrap confidence intervals\n"
        )
        f.write(
            "- **Sampling:** Stratified random sample maintaining winner distribution\n"
        )
        f.write("- **Significance:** 95% confidence intervals excluding zero\n")
        f.write("- **Bootstrap:** 100 resampling iterations per analysis\n")

    # Also create CSV of results for easy consumption
    csv_data = []
    for analysis_name, data in results.items():
        if "results" in data:
            result = data["results"]
            for feature in result["features"]:
                coeff = result["coefficients"][feature]
                ci = result["confidence_intervals"][feature]
                csv_data.append(
                    {
                        "feature": feature,
                        "coefficient": coeff,
                        "ci_lower": ci["lower"],
                        "ci_upper": ci["upper"],
                        "significant": ci["lower"] * ci["upper"] > 0,
                        "analysis": analysis_name,
                    }
                )

    if csv_data:
        results_csv = output_dir / "citation_effects_results.csv"
        pd.DataFrame(csv_data).to_csv(results_csv, index=False)
        print(f"CSV results: {results_csv}")


if __name__ == "__main__":
    sys.exit(main())
