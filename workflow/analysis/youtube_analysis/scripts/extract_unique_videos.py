#!/usr/bin/env python3
"""
Extract unique video IDs from YouTube citations.

This script processes YouTube citations to create a dataset of unique video IDs
with citation frequencies, optimizing subsequent API calls.
"""

import pandas as pd


def extract_unique_videos(youtube_citations):
    """Extract unique video IDs with citation statistics."""
    print(f"Extracting unique videos from {len(youtube_citations):,} citations...")

    # Group by video_id to get statistics
    unique_videos = (
        youtube_citations.groupby("video_id")
        .agg(
            {
                "citation_id": "count",  # Number of times this video is cited
                "response_id": "nunique",  # Number of unique responses citing this video
                "url": "first",  # Keep one example URL for reference
                "citation_number": ["min", "max"],  # Citation position range
                "citation_order": ["min", "max"],  # Global citation order range
            }
        )
        .reset_index()
    )

    # Flatten column names
    unique_videos.columns = [
        "video_id",
        "citation_count",
        "unique_response_count",
        "sample_url",
        "min_citation_number",
        "max_citation_number",
        "min_citation_order",
        "max_citation_order",
    ]

    # Sort by citation frequency (most cited videos first)
    unique_videos = unique_videos.sort_values("citation_count", ascending=False)

    print(f"Extracted {len(unique_videos):,} unique videos")
    print(f"Total citations represented: {unique_videos['citation_count'].sum():,}")

    return unique_videos


def generate_unique_videos_summary(unique_videos):
    """Generate summary statistics for unique videos dataset."""
    print("\n=== UNIQUE VIDEOS SUMMARY ===")

    total_videos = len(unique_videos)
    total_citations = unique_videos["citation_count"].sum()

    print(f"Total unique videos: {total_videos:,}")
    print(f"Total citations: {total_citations:,}")
    print(f"Average citations per video: {total_citations / total_videos:.1f}")

    # Citation frequency distribution
    citation_counts = unique_videos["citation_count"]
    print("\nCitation frequency distribution:")
    print(f"  Videos cited once: {(citation_counts == 1).sum():,}")
    print(
        f"  Videos cited 2-5 times: {((citation_counts >= 2) & (citation_counts <= 5)).sum():,}"
    )
    print(
        f"  Videos cited 6-10 times: {((citation_counts >= 6) & (citation_counts <= 10)).sum():,}"
    )
    print(f"  Videos cited 11+ times: {(citation_counts >= 11).sum():,}")

    # Response distribution
    response_counts = unique_videos["unique_response_count"]
    print("\nResponse distribution:")
    print(f"  Videos cited in 1 response: {(response_counts == 1).sum():,}")
    print(
        f"  Videos cited in 2-3 responses: {((response_counts >= 2) & (response_counts <= 3)).sum():,}"
    )
    print(f"  Videos cited in 4+ responses: {(response_counts >= 4).sum():,}")

    # Most cited videos
    print("\nTop 15 most cited videos:")
    top_videos = unique_videos.head(15)
    for _, row in top_videos.iterrows():
        print(
            f"  {row['video_id']}: {row['citation_count']} citations in {row['unique_response_count']} responses"
        )
        print(f"    URL: {row['sample_url']}")
        print()

    # Citation statistics
    print("Citation statistics:")
    print(f"  Total citations: {citation_counts.sum():,}")
    print(f"  Mean citations per video: {citation_counts.mean():.1f}")
    print(f"  Median citations per video: {citation_counts.median():.1f}")
    print(f"  Max citations for single video: {citation_counts.max()}")
    print(f"  Min citations for single video: {citation_counts.min()}")

    # API efficiency calculation
    print("\nAPI efficiency:")
    api_requests_needed = (total_videos + 49) // 50  # Round up to nearest 50
    print(f"  Videos to fetch: {total_videos:,}")
    print(f"  API requests needed: {api_requests_needed:,} (50 videos per request)")
    print(f"  API units required: {api_requests_needed:,} units")


def main():
    """Main function for extracting unique video IDs."""
    # Get paths from Snakemake
    youtube_citations_path = snakemake.input.youtube_citations
    output_path = snakemake.output[0]

    # Load YouTube citations
    print(f"Loading YouTube citations from {youtube_citations_path}")
    youtube_citations = pd.read_parquet(youtube_citations_path)
    print(f"Loaded {len(youtube_citations):,} YouTube citations")

    # Extract unique videos
    unique_videos = extract_unique_videos(youtube_citations)

    # Generate summary statistics
    generate_unique_videos_summary(unique_videos)

    # Save results
    print(f"\nSaving unique videos to {output_path}")
    unique_videos.to_parquet(output_path, index=False)

    print("\n✅ Unique video extraction completed!")
    print(f"Output: {len(unique_videos):,} unique videos")
    print(f"Columns: {list(unique_videos.columns)}")


if __name__ == "__main__":
    main()
