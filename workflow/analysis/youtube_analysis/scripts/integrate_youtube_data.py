#!/usr/bin/env python3
"""
Integrate all YouTube data into final analysis dataset.

This script combines YouTube citations, video metadata, and channel metadata
into a comprehensive dataset ready for analysis.
"""

import pandas as pd


def integrate_youtube_datasets(youtube_citations, video_metadata, channel_metadata):
    """Integrate all YouTube datasets into a comprehensive analysis dataset."""
    print("Integrating datasets:")
    print(f"  YouTube citations: {len(youtube_citations):,} rows")
    print(f"  Video metadata: {len(video_metadata):,} rows")
    print(f"  Channel metadata: {len(channel_metadata):,} rows")

    # Start with YouTube citations as the base
    integrated_data = youtube_citations.copy()
    print(f"\nStarting with {len(integrated_data):,} YouTube citations")

    # Merge with video metadata
    print("Merging with video metadata...")
    before_video_merge = len(integrated_data)
    integrated_data = integrated_data.merge(
        video_metadata, on="video_id", how="left", suffixes=("", "_video_meta")
    )

    print(f"After video merge: {len(integrated_data):,} rows")
    if len(integrated_data) != before_video_merge:
        print("⚠️  Warning: Row count changed during video merge!")

    # Check video metadata coverage
    video_coverage = integrated_data["title"].notna().sum()
    print(
        f"Video metadata coverage: {video_coverage:,}/{len(integrated_data):,} ({video_coverage / len(integrated_data) * 100:.1f}%)"
    )

    # Merge with channel metadata
    print("\nMerging with channel metadata...")
    before_channel_merge = len(integrated_data)
    integrated_data = integrated_data.merge(
        channel_metadata, on="channel_id", how="left", suffixes=("", "_channel_meta")
    )

    print(f"After channel merge: {len(integrated_data):,} rows")
    if len(integrated_data) != before_channel_merge:
        print("⚠️  Warning: Row count changed during channel merge!")

    # Check channel metadata coverage
    channel_coverage = integrated_data["subscriber_count"].notna().sum()
    print(
        f"Channel metadata coverage: {channel_coverage:,}/{len(integrated_data):,} ({channel_coverage / len(integrated_data) * 100:.1f}%)"
    )

    return integrated_data


def clean_and_optimize_dataset(integrated_data):
    """Clean and optimize the integrated dataset."""
    print("\nCleaning and optimizing dataset...")

    # Handle duplicate column names from merges
    duplicate_columns = []
    for col in integrated_data.columns:
        if col.endswith("_video_meta") or col.endswith("_channel_meta"):
            base_col = col.replace("_video_meta", "").replace("_channel_meta", "")
            if base_col in integrated_data.columns:
                # Keep the more detailed version (from metadata)
                integrated_data[base_col] = integrated_data[col].fillna(
                    integrated_data[base_col]
                )
                duplicate_columns.append(col)

    if duplicate_columns:
        print(f"Removing {len(duplicate_columns)} duplicate columns from merges")
        integrated_data = integrated_data.drop(columns=duplicate_columns)

    # Optimize data types
    print("Optimizing data types...")

    # Convert numeric columns
    numeric_columns = [
        "view_count",
        "like_count",
        "comment_count",
        "subscriber_count",
        "video_count",
        "citation_number",
        "citation_order",
    ]
    for col in numeric_columns:
        if col in integrated_data.columns:
            integrated_data[col] = (
                pd.to_numeric(integrated_data[col], errors="coerce")
                .fillna(0)
                .astype("int64")
            )

    # Convert boolean columns
    boolean_columns = [
        "captions_available",
        "hidden_subscriber_count",
        "is_linked",
        "made_for_kids",
    ]
    for col in boolean_columns:
        if col in integrated_data.columns:
            integrated_data[col] = integrated_data[col].astype("bool")

    # Convert datetime columns
    datetime_columns = ["published_at", "api_fetch_date"]
    for col in datetime_columns:
        if col in integrated_data.columns:
            integrated_data[col] = pd.to_datetime(integrated_data[col], errors="coerce")

    print(f"Final dataset shape: {integrated_data.shape}")
    print(
        f"Memory usage: {integrated_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    )

    return integrated_data


def generate_integration_summary(integrated_data):
    """Generate comprehensive summary of the integrated dataset."""
    print("\n=== INTEGRATED YOUTUBE DATASET SUMMARY ===")

    total_citations = len(integrated_data)
    unique_videos = integrated_data["video_id"].nunique()
    unique_channels = integrated_data["channel_id"].nunique()
    unique_responses = integrated_data["response_id"].nunique()

    print("Dataset overview:")
    print(f"  Total YouTube citations: {total_citations:,}")
    print(f"  Unique videos: {unique_videos:,}")
    print(f"  Unique channels: {unique_channels:,}")
    print(f"  Unique responses with YouTube citations: {unique_responses:,}")

    # Data completeness
    print("\nData completeness:")
    key_fields = [
        "title",
        "channel_title",
        "subscriber_count",
        "view_count",
        "published_at",
    ]
    for field in key_fields:
        if field in integrated_data.columns:
            completeness = integrated_data[field].notna().sum()
            pct = completeness / total_citations * 100
            print(f"  {field}: {completeness:,}/{total_citations:,} ({pct:.1f}%)")

    # Video statistics
    if "view_count" in integrated_data.columns:
        video_view_stats = integrated_data[integrated_data["view_count"] > 0][
            "view_count"
        ]
        if len(video_view_stats) > 0:
            print("\nVideo view statistics:")
            print(f"  Videos with view data: {len(video_view_stats):,}")
            print(
                f"  Total views across all cited videos: {video_view_stats.sum():,.0f}"
            )
            print(f"  Mean views per video: {video_view_stats.mean():,.0f}")
            print(f"  Median views per video: {video_view_stats.median():,.0f}")
            print(f"  Most viewed cited video: {video_view_stats.max():,.0f} views")

    # Channel statistics
    if "subscriber_count" in integrated_data.columns:
        channel_sub_stats = integrated_data[integrated_data["subscriber_count"] > 0]
        if len(channel_sub_stats) > 0:
            print("\nChannel subscriber statistics:")
            print(
                f"  Citations from channels with subscriber data: {len(channel_sub_stats):,}"
            )
            channel_subs = channel_sub_stats.groupby("channel_id")[
                "subscriber_count"
            ].first()
            print(f"  Mean subscribers per channel: {channel_subs.mean():,.0f}")
            print(f"  Median subscribers per channel: {channel_subs.median():,.0f}")
            print(f"  Largest channel: {channel_subs.max():,.0f} subscribers")

    # Most cited content
    print("\nMost cited videos:")
    video_citation_counts = integrated_data["video_id"].value_counts().head(10)
    for video_id, count in video_citation_counts.items():
        video_info = integrated_data[integrated_data["video_id"] == video_id].iloc[0]
        title = video_info.get("title", "Unknown Title")[:50]
        channel = video_info.get("channel_title", "Unknown Channel")
        views = video_info.get("view_count", 0)
        print(f'  {count}x: "{title}" by {channel} ({views:,} views)')

    print("\nMost cited channels:")
    channel_citation_counts = (
        integrated_data.groupby("channel_id")
        .agg(
            {"video_id": "count", "channel_title": "first", "subscriber_count": "first"}
        )
        .sort_values("video_id", ascending=False)
        .head(10)
    )

    for _, row in channel_citation_counts.iterrows():
        channel_name = row["channel_title"] or "Unknown Channel"
        citation_count = row["video_id"]
        subscribers = row["subscriber_count"] or 0
        print(f"  {citation_count}x: {channel_name} ({subscribers:,} subscribers)")

    # Publication date analysis
    if "published_at" in integrated_data.columns:
        pub_data = integrated_data.dropna(subset=["published_at"])
        if len(pub_data) > 0:
            pub_data["pub_year"] = pub_data["published_at"].dt.year
            year_counts = pub_data["pub_year"].value_counts().sort_index()
            print("\nCited video publication years (top 10):")
            for year, count in year_counts.tail(10).items():
                print(f"  {int(year)}: {count} citations")

    # Content categories (if available)
    if "category_id" in integrated_data.columns:
        category_counts = integrated_data["category_id"].value_counts().head(10)
        if len(category_counts) > 0:
            print("\nTop video categories (by category ID):")
            for category_id, count in category_counts.items():
                pct = count / total_citations * 100
                print(f"  Category {category_id}: {count:,} citations ({pct:.1f}%)")


def main():
    """Main function for integrating YouTube data."""
    # Get paths from Snakemake
    youtube_citations_path = snakemake.input.youtube_citations
    video_metadata_path = snakemake.input.video_metadata
    channel_metadata_path = snakemake.input.channel_metadata
    output_path = snakemake.output[0]

    # Load all datasets
    print(f"Loading YouTube citations from {youtube_citations_path}")
    youtube_citations = pd.read_parquet(youtube_citations_path)

    print(f"Loading video metadata from {video_metadata_path}")
    video_metadata = pd.read_parquet(video_metadata_path)

    print(f"Loading channel metadata from {channel_metadata_path}")
    channel_metadata = pd.read_parquet(channel_metadata_path)

    # Integrate all datasets
    integrated_data = integrate_youtube_datasets(
        youtube_citations, video_metadata, channel_metadata
    )

    # Clean and optimize the dataset
    integrated_data = clean_and_optimize_dataset(integrated_data)

    # Generate comprehensive summary
    generate_integration_summary(integrated_data)

    # Save final dataset
    print(f"\nSaving integrated YouTube dataset to {output_path}")
    integrated_data.to_parquet(output_path, index=False)

    print("\n✅ YouTube data integration completed!")
    print(
        f"Final dataset: {len(integrated_data):,} rows, {len(integrated_data.columns)} columns"
    )
    print(
        f"File size: {integrated_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    )

    # List all columns for reference
    print("\nColumns in final dataset:")
    for i, col in enumerate(sorted(integrated_data.columns), 1):
        print(f"  {i:2d}. {col}")


if __name__ == "__main__":
    main()
