#!/usr/bin/env python3
"""
Fetch video metadata from YouTube Data API v3.

This script takes YouTube video IDs and fetches comprehensive metadata
including title, description, view counts, channel information, etc.
"""

import pandas as pd
import os
import json
import time
from datetime import datetime
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv


def setup_youtube_api():
    """Initialize YouTube Data API client."""
    load_dotenv()
    api_key = os.getenv(snakemake.params.api_key)

    if not api_key:
        raise ValueError(
            f"YouTube API key not found in environment variable: {snakemake.params.api_key}"
        )

    return build("youtube", "v3", developerKey=api_key)


def batch_video_ids(video_ids, batch_size=50):
    """Split video IDs into batches for API calls."""
    for i in range(0, len(video_ids), batch_size):
        yield video_ids[i : i + batch_size]


def fetch_video_batch(youtube, video_ids_batch):
    """
    Fetch metadata for a batch of video IDs.

    Returns:
        tuple: (success_videos, failed_video_ids)
    """
    video_ids_str = ",".join(video_ids_batch)

    try:
        response = (
            youtube.videos()
            .list(
                part="snippet,statistics,contentDetails,status",
                id=video_ids_str,
                maxResults=50,
            )
            .execute()
        )

        return response["items"], []

    except HttpError as e:
        print(f"API error for batch: {e}")
        return [], video_ids_batch
    except Exception as e:
        print(f"Unexpected error for batch: {e}")
        return [], video_ids_batch


def parse_video_metadata(video_item):
    """Parse video metadata from API response into structured format."""
    snippet = video_item.get("snippet", {})
    statistics = video_item.get("statistics", {})
    content_details = video_item.get("contentDetails", {})
    status = video_item.get("status", {})

    # Extract basic information
    metadata = {
        "video_id": video_item["id"],
        "title": snippet.get("title", ""),
        "description": snippet.get("description", "")[
            : snakemake.config.get("max_description_length", 500)
        ],
        "channel_id": snippet.get("channelId", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "category_id": snippet.get("categoryId", ""),
        "default_language": snippet.get("defaultLanguage", ""),
        "tags": json.dumps(snippet.get("tags", [])),  # Store as JSON string
    }

    # Extract statistics (handle missing values)
    metadata.update(
        {
            "view_count": int(statistics.get("viewCount", 0))
            if statistics.get("viewCount")
            else 0,
            "like_count": int(statistics.get("likeCount", 0))
            if statistics.get("likeCount")
            else 0,
            "comment_count": int(statistics.get("commentCount", 0))
            if statistics.get("commentCount")
            else 0,
        }
    )

    # Extract content details
    metadata.update(
        {
            "duration": content_details.get("duration", ""),
            "captions_available": content_details.get("caption", "false") == "true",
        }
    )

    # Extract status information
    metadata.update(
        {
            "privacy_status": status.get("privacyStatus", ""),
            "upload_status": status.get("uploadStatus", ""),
        }
    )

    # Add fetch timestamp
    metadata["api_fetch_date"] = datetime.now().isoformat()

    return metadata


def save_raw_responses(responses, output_dir, batch_num):
    """Save raw API responses for debugging and future analysis."""
    if not responses:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"video_batch_{batch_num:04d}_{timestamp}.json"
    filepath = Path(output_dir) / filename

    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)


def fetch_all_video_metadata(unique_videos):
    """Fetch metadata for all unique videos with comprehensive error handling."""
    print(f"Processing {len(unique_videos):,} unique videos...")

    youtube = setup_youtube_api()
    video_ids = unique_videos["video_id"].tolist()

    print(f"Video IDs to fetch: {len(video_ids):,}")
    print(f"Total citations represented: {unique_videos['citation_count'].sum():,}")
    print(f"Average citations per video: {unique_videos['citation_count'].mean():.1f}")

    all_video_metadata = []
    failed_video_ids = []
    batch_size = snakemake.params.batch_size
    rate_limit_delay = snakemake.params.rate_limit_delay

    # Process in batches
    for batch_num, video_ids_batch in enumerate(
        batch_video_ids(video_ids, batch_size), 1
    ):
        print(
            f"Processing batch {batch_num}/{(len(video_ids) + batch_size - 1) // batch_size} "
            f"({len(video_ids_batch)} videos)..."
        )

        # Add rate limiting delay
        if batch_num > 1:
            time.sleep(rate_limit_delay)

        # Fetch batch
        success_videos, failed_ids = fetch_video_batch(youtube, video_ids_batch)

        # Save raw responses
        if snakemake.config.get("preserve_raw_responses", True):
            save_raw_responses(
                success_videos, snakemake.output.api_responses_dir, batch_num
            )

        # Process successful videos
        for video_item in success_videos:
            try:
                metadata = parse_video_metadata(video_item)
                all_video_metadata.append(metadata)
            except Exception as e:
                print(f"Error parsing video {video_item.get('id', 'unknown')}: {e}")
                failed_video_ids.append(video_item.get("id", "unknown"))

        # Track failed videos
        failed_video_ids.extend(failed_ids)

        print(f"  Success: {len(success_videos)} videos")
        print(f"  Failed: {len(failed_ids)} videos")

    print("\nFetch completed:")
    print(f"  Total successful: {len(all_video_metadata):,}")
    print(f"  Total failed: {len(failed_video_ids):,}")

    if failed_video_ids:
        print(f"  Failed video IDs: {failed_video_ids[:10]}...")  # Show first 10

    return all_video_metadata, failed_video_ids


def generate_video_metadata_summary(video_metadata_df):
    """Generate summary statistics for video metadata."""
    print("\n=== VIDEO METADATA SUMMARY ===")

    total_videos = len(video_metadata_df)
    print(f"Total videos with metadata: {total_videos:,}")

    # Basic statistics
    print("\nBasic statistics:")
    print(f"  Videos with titles: {video_metadata_df['title'].notna().sum():,}")
    print(
        f"  Videos with descriptions: {video_metadata_df['description'].notna().sum():,}"
    )
    print(f"  Videos with view counts: {(video_metadata_df['view_count'] > 0).sum():,}")
    print(f"  Videos with like counts: {(video_metadata_df['like_count'] > 0).sum():,}")

    # View count statistics
    if (video_metadata_df["view_count"] > 0).any():
        print("\nView count statistics:")
        view_stats = video_metadata_df[video_metadata_df["view_count"] > 0][
            "view_count"
        ]
        print(f"  Mean views: {view_stats.mean():,.0f}")
        print(f"  Median views: {view_stats.median():,.0f}")
        print(f"  Max views: {view_stats.max():,.0f}")
        print(f"  Min views: {view_stats.min():,.0f}")

    # Channel distribution
    print("\nChannel distribution:")
    channel_counts = video_metadata_df["channel_title"].value_counts()
    print(f"  Unique channels: {len(channel_counts):,}")
    print(f"  Channels with 1 video: {(channel_counts == 1).sum():,}")
    print(f"  Channels with 2+ videos: {(channel_counts >= 2).sum():,}")

    # Top channels by video count
    print("\nTop 10 channels by video count:")
    for channel, count in channel_counts.head(10).items():
        print(f"  {channel}: {count} videos")

    # Publication date analysis
    if "published_at" in video_metadata_df.columns:
        video_metadata_df["published_year"] = pd.to_datetime(
            video_metadata_df["published_at"]
        ).dt.year
        year_counts = video_metadata_df["published_year"].value_counts().sort_index()
        print("\nPublication year distribution (top 10):")
        for year, count in year_counts.tail(10).items():
            if pd.notna(year):
                print(f"  {int(year)}: {count} videos")

    # Duration analysis (if available)
    if "duration" in video_metadata_df.columns:
        duration_available = video_metadata_df["duration"].notna().sum()
        print(f"\nDuration information available for {duration_available:,} videos")


def main():
    """Main function for fetching video metadata."""
    # Get paths from Snakemake
    unique_videos_path = snakemake.input.unique_videos
    output_path = snakemake.output.video_metadata

    # Load unique videos
    print(f"Loading unique videos from {unique_videos_path}")
    unique_videos = pd.read_parquet(unique_videos_path)
    print(f"Loaded {len(unique_videos):,} unique videos")

    # Fetch video metadata
    video_metadata_list, failed_video_ids = fetch_all_video_metadata(unique_videos)

    # Convert to DataFrame
    if video_metadata_list:
        video_metadata_df = pd.DataFrame(video_metadata_list)

        # Generate summary statistics
        generate_video_metadata_summary(video_metadata_df)

        # Save results
        print(f"\nSaving video metadata to {output_path}")
        video_metadata_df.to_parquet(output_path, index=False)

        print("\n✅ Video metadata collection completed!")
        print(f"Successfully processed: {len(video_metadata_df):,} videos")
        print(f"Failed to process: {len(failed_video_ids):,} videos")
        print(f"Columns: {list(video_metadata_df.columns)}")

    else:
        print("\n❌ No video metadata collected!")
        # Create empty DataFrame with expected schema
        empty_df = pd.DataFrame(
            columns=[
                "video_id",
                "title",
                "description",
                "channel_id",
                "channel_title",
                "published_at",
                "category_id",
                "default_language",
                "tags",
                "view_count",
                "like_count",
                "comment_count",
                "duration",
                "captions_available",
                "privacy_status",
                "upload_status",
                "api_fetch_date",
            ]
        )
        empty_df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
