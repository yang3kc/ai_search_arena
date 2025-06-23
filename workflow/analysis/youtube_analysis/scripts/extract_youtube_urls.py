#!/usr/bin/env python3
"""
Extract YouTube URLs and video IDs from citations data.

This script filters the citations dataset for YouTube videos and extracts
video IDs from various YouTube URL formats.
"""

import pandas as pd
import re
from urllib.parse import urlparse, parse_qs


def extract_video_id_from_url(url):
    """
    Extract YouTube video ID from various URL formats.

    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    """
    if not isinstance(url, str):
        return None

    # Handle youtu.be format
    if "youtu.be/" in url:
        try:
            return url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
        except:
            return None

    # Handle youtube.com formats
    if "youtube.com" in url:
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)

            # Standard watch URL
            if "v" in query_params:
                return query_params["v"][0]

            # Embed URL
            if "/embed/" in parsed.path:
                return parsed.path.split("/embed/")[-1].split("?")[0]

        except:
            return None

    return None


def is_youtube_video_url(url):
    """Check if URL is a YouTube video URL (not just a page mentioning YouTube)."""
    if not isinstance(url, str):
        return False

    # YouTube video patterns
    youtube_patterns = [
        r"youtube\.com/watch\?.*[&?]v=",  # watch URL with v parameter (can have other params before)
        r"youtube\.com/watch\?v=",       # watch URL with v as first parameter
        r"youtu\.be/",                   # short URL format
        r"youtube\.com/embed/",          # embed URL format
        r"m\.youtube\.com/watch\?.*[&?]v=",  # mobile with v parameter (can have other params before)
        r"m\.youtube\.com/watch\?v=",    # mobile with v as first parameter
    ]

    return any(re.search(pattern, url, re.IGNORECASE) for pattern in youtube_patterns)


def extract_youtube_citations(citations_df):
    """Extract and process YouTube citations from the citations dataset."""
    print(f"Processing {len(citations_df):,} total citations...")

    # Filter for YouTube video URLs
    youtube_mask = citations_df["url"].apply(is_youtube_video_url)
    youtube_citations = citations_df[youtube_mask].copy()

    print(f"Found {len(youtube_citations):,} YouTube video citations")

    # Extract video IDs
    print("Extracting video IDs from URLs...")
    youtube_citations["video_id"] = youtube_citations["url"].apply(
        extract_video_id_from_url
    )

    # Remove rows where video ID extraction failed
    before_filter = len(youtube_citations)
    youtube_citations = youtube_citations.dropna(subset=["video_id"])
    youtube_citations = youtube_citations[youtube_citations["video_id"] != ""]

    print(f"Successfully extracted {len(youtube_citations):,} video IDs")
    print(f"Failed to extract {before_filter - len(youtube_citations):,} video IDs")

    # Keep all citations - don't deduplicate yet as we need citation-video relationships
    # Deduplication will happen later when we extract unique videos for API calls
    print(f"Preserving all {len(youtube_citations):,} citation-video relationships")

    # Validate video IDs (should be 11 characters, alphanumeric + - and _)
    valid_id_pattern = r"^[a-zA-Z0-9_-]{11}$"
    valid_mask = youtube_citations["video_id"].str.match(valid_id_pattern)
    invalid_count = (~valid_mask).sum()

    if invalid_count > 0:
        print(f"Warning: {invalid_count} video IDs don't match expected format")
        print(
            "Sample invalid IDs:",
            youtube_citations[~valid_mask]["video_id"].head(3).tolist(),
        )

    youtube_citations = youtube_citations[valid_mask]
    print(f"Final dataset: {len(youtube_citations):,} valid YouTube video citations")

    return youtube_citations


def generate_summary_stats(youtube_citations):
    """Generate summary statistics for the YouTube citations dataset."""
    print("\n=== YOUTUBE CITATIONS SUMMARY ===")

    total_citations = len(youtube_citations)
    unique_videos = youtube_citations["video_id"].nunique()
    unique_responses = youtube_citations["response_id"].nunique()

    print(f"Total YouTube citations: {total_citations:,}")
    print(f"Unique videos: {unique_videos:,}")
    print(f"Unique responses containing YouTube: {unique_responses:,}")
    print(f"Average citations per video: {total_citations/unique_videos:.1f}")

    # Citation frequency distribution
    citation_counts = youtube_citations["video_id"].value_counts()
    print(f"\nVideo citation frequency:")
    print(f"  Videos cited once: {(citation_counts == 1).sum():,}")
    print(
        f"  Videos cited 2-5 times: {((citation_counts >= 2) & (citation_counts <= 5)).sum():,}"
    )
    print(f"  Videos cited 6+ times: {(citation_counts >= 6).sum():,}")

    # Most cited videos
    print(f"\nTop 10 most cited videos:")
    top_videos = citation_counts.head(10)
    for video_id, count in top_videos.items():
        sample_url = youtube_citations[youtube_citations["video_id"] == video_id][
            "url"
        ].iloc[0]
        print(f"  {video_id}: {count} citations - {sample_url}")

    # URL format distribution
    print(f"\nURL format distribution:")
    url_formats = {
        "youtube.com/watch": youtube_citations["url"]
        .str.contains("youtube.com/watch", case=False)
        .sum(),
        "youtu.be": youtube_citations["url"]
        .str.contains("youtu.be/", case=False)
        .sum(),
        "youtube.com/embed": youtube_citations["url"]
        .str.contains("youtube.com/embed", case=False)
        .sum(),
        "m.youtube.com": youtube_citations["url"]
        .str.contains("m.youtube.com", case=False)
        .sum(),
    }

    for format_name, count in url_formats.items():
        pct = count / total_citations * 100
        print(f"  {format_name}: {count:,} ({pct:.1f}%)")


def main():
    """Main function for extracting YouTube URLs and video IDs."""
    # Get paths from Snakemake
    citations_path = snakemake.input.citations
    output_path = snakemake.output[0]

    # Load citations data
    print(f"Loading citations from {citations_path}")
    citations = pd.read_parquet(citations_path)
    print(f"Loaded {len(citations):,} citations")

    # Extract YouTube citations
    youtube_citations = extract_youtube_citations(citations)

    # Generate summary statistics
    generate_summary_stats(youtube_citations)

    # Save results
    print(f"\nSaving YouTube citations to {output_path}")
    youtube_citations.to_parquet(output_path, index=False)

    print(f"\n✅ YouTube URL extraction completed!")
    print(f"Output: {len(youtube_citations):,} YouTube video citations")
    print(f"Columns: {list(youtube_citations.columns)}")


if __name__ == "__main__":
    main()
