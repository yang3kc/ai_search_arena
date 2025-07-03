#!/usr/bin/env python3
"""
Extract unique channel data from video metadata.

This script processes video metadata to extract unique YouTube channels
and calculate citation frequencies for each channel.
"""

import pandas as pd


def extract_unique_channels(video_metadata_df):
    """Extract unique channels from video metadata with citation frequencies."""
    print(f"Extracting unique channels from {len(video_metadata_df):,} videos...")
    
    # Group by channel to get statistics
    channel_stats = video_metadata_df.groupby(['channel_id', 'channel_title']).agg({
        'video_id': 'count',  # Number of cited videos per channel
        'view_count': ['sum', 'mean', 'max'],  # View statistics
        'published_at': ['min', 'max']  # Date range of cited videos
    }).reset_index()
    
    # Flatten column names
    channel_stats.columns = [
        'channel_id', 'channel_title', 'cited_video_count',
        'total_views_of_cited_videos', 'avg_views_of_cited_videos', 'max_views_of_cited_videos',
        'earliest_cited_video', 'latest_cited_video'
    ]
    
    # Remove channels with missing channel_id
    before_filter = len(channel_stats)
    channel_stats = channel_stats.dropna(subset=['channel_id'])
    channel_stats = channel_stats[channel_stats['channel_id'] != '']
    
    print(f"Channels before filtering: {before_filter:,}")
    print(f"Valid channels after filtering: {len(channel_stats):,}")
    print(f"Removed {before_filter - len(channel_stats):,} channels with missing IDs")
    
    # Sort by citation frequency (most cited channels first)
    channel_stats = channel_stats.sort_values('cited_video_count', ascending=False)
    
    return channel_stats


def generate_channel_summary(channel_stats):
    """Generate summary statistics for channel data."""
    print("\n=== CHANNEL DATA SUMMARY ===")
    
    total_channels = len(channel_stats)
    total_cited_videos = channel_stats['cited_video_count'].sum()
    
    print(f"Total unique channels: {total_channels:,}")
    print(f"Total cited videos: {total_cited_videos:,}")
    print(f"Average videos per channel: {total_cited_videos / total_channels:.1f}")
    
    # Citation frequency distribution
    citation_counts = channel_stats['cited_video_count'].value_counts().sort_index()
    print("\nChannel citation frequency distribution:")
    print(f"  Channels with 1 cited video: {(channel_stats['cited_video_count'] == 1).sum():,}")
    print(f"  Channels with 2-5 cited videos: {((channel_stats['cited_video_count'] >= 2) & (channel_stats['cited_video_count'] <= 5)).sum():,}")
    print(f"  Channels with 6-10 cited videos: {((channel_stats['cited_video_count'] >= 6) & (channel_stats['cited_video_count'] <= 10)).sum():,}")
    print(f"  Channels with 11+ cited videos: {(channel_stats['cited_video_count'] >= 11).sum():,}")
    
    # Top channels by citation count
    print("\nTop 15 most cited channels:")
    top_channels = channel_stats.head(15)
    for _, row in top_channels.iterrows():
        channel_name = row['channel_title'] if row['channel_title'] else 'Unknown Channel'
        print(f"  {channel_name}: {row['cited_video_count']} videos")
        print(f"    Channel ID: {row['channel_id']}")
        print(f"    Total views of cited videos: {row['total_views_of_cited_videos']:,.0f}")
        print()
    
    # View statistics
    if (channel_stats['total_views_of_cited_videos'] > 0).any():
        view_stats = channel_stats[channel_stats['total_views_of_cited_videos'] > 0]
        print("View statistics for cited videos:")
        print(f"  Channels with view data: {len(view_stats):,}")
        print(f"  Total views across all cited videos: {view_stats['total_views_of_cited_videos'].sum():,.0f}")
        print(f"  Average total views per channel: {view_stats['total_views_of_cited_videos'].mean():,.0f}")
        print(f"  Median total views per channel: {view_stats['total_views_of_cited_videos'].median():,.0f}")
    
    # Channel name availability
    named_channels = channel_stats['channel_title'].notna().sum()
    unnamed_channels = total_channels - named_channels
    print("\nChannel name availability:")
    print(f"  Channels with names: {named_channels:,} ({named_channels/total_channels*100:.1f}%)")
    print(f"  Channels without names: {unnamed_channels:,} ({unnamed_channels/total_channels*100:.1f}%)")


def main():
    """Main function for extracting channel data."""
    # Get paths from Snakemake
    video_metadata_path = snakemake.input.video_metadata
    output_path = snakemake.output[0]
    
    # Load video metadata
    print(f"Loading video metadata from {video_metadata_path}")
    video_metadata = pd.read_parquet(video_metadata_path)
    print(f"Loaded {len(video_metadata):,} video records")
    
    # Extract unique channels
    channel_stats = extract_unique_channels(video_metadata)
    
    # Generate summary statistics
    generate_channel_summary(channel_stats)
    
    # Save results
    print(f"\nSaving channel data to {output_path}")
    channel_stats.to_parquet(output_path, index=False)
    
    print("\n✅ Channel data extraction completed!")
    print(f"Output: {len(channel_stats):,} unique channels")
    print(f"Columns: {list(channel_stats.columns)}")


if __name__ == "__main__":
    main()