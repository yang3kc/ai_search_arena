#!/usr/bin/env python3
"""
Fetch channel metadata from YouTube Data API v3.

This script takes unique channel IDs and fetches comprehensive metadata
including subscriber counts, video counts, channel descriptions, etc.
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
        raise ValueError(f"YouTube API key not found in environment variable: {snakemake.params.api_key}")
    
    return build("youtube", "v3", developerKey=api_key)


def batch_channel_ids(channel_ids, batch_size=50):
    """Split channel IDs into batches for API calls."""
    for i in range(0, len(channel_ids), batch_size):
        yield channel_ids[i:i + batch_size]


def fetch_channel_batch(youtube, channel_ids_batch):
    """
    Fetch metadata for a batch of channel IDs.
    
    Returns:
        tuple: (success_channels, failed_channel_ids)
    """
    channel_ids_str = ','.join(channel_ids_batch)
    
    try:
        response = youtube.channels().list(
            part='snippet,statistics,status,brandingSettings',
            id=channel_ids_str,
            maxResults=50
        ).execute()
        
        return response['items'], []
        
    except HttpError as e:
        print(f"API error for batch: {e}")
        return [], channel_ids_batch
    except Exception as e:
        print(f"Unexpected error for batch: {e}")
        return [], channel_ids_batch


def parse_channel_metadata(channel_item):
    """Parse channel metadata from API response into structured format."""
    snippet = channel_item.get('snippet', {})
    statistics = channel_item.get('statistics', {})
    status = channel_item.get('status', {})
    branding = channel_item.get('brandingSettings', {}).get('channel', {})
    
    # Extract basic information
    metadata = {
        'channel_id': channel_item['id'],
        'title': snippet.get('title', ''),
        'description': snippet.get('description', '')[:snakemake.config.get('max_description_length', 500)],
        'custom_url': snippet.get('customUrl', ''),
        'published_at': snippet.get('publishedAt', ''),
        'country': snippet.get('country', ''),
        'default_language': snippet.get('defaultLanguage', ''),
    }
    
    # Extract statistics (handle missing values gracefully)
    metadata.update({
        'view_count': int(statistics.get('viewCount', 0)) if statistics.get('viewCount') else 0,
        'subscriber_count': int(statistics.get('subscriberCount', 0)) if statistics.get('subscriberCount') else 0,
        'video_count': int(statistics.get('videoCount', 0)) if statistics.get('videoCount') else 0,
        'hidden_subscriber_count': statistics.get('hiddenSubscriberCount', False),
    })
    
    # Extract additional branding/channel information
    metadata.update({
        'keywords': branding.get('keywords', ''),
        'unsubscribed_trailer': branding.get('unsubscribedTrailer', ''),
    })
    
    # Extract status information
    metadata.update({
        'privacy_status': status.get('privacyStatus', ''),
        'is_linked': status.get('isLinked', False),
        'made_for_kids': status.get('madeForKids', False),
    })
    
    # Add fetch timestamp
    metadata['api_fetch_date'] = datetime.now().isoformat()
    
    return metadata


def save_raw_responses(responses, output_dir, batch_num):
    """Save raw API responses for debugging and future analysis."""
    if not responses:
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"channel_batch_{batch_num:04d}_{timestamp}.json"
    filepath = Path(output_dir) / filename
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)


def fetch_all_channel_metadata(unique_channels_df):
    """Fetch metadata for all channels with comprehensive error handling."""
    print(f"Fetching metadata for {len(unique_channels_df):,} channels...")
    
    youtube = setup_youtube_api()
    channel_ids = unique_channels_df['channel_id'].unique().tolist()
    
    print(f"Unique channel IDs to fetch: {len(channel_ids):,}")
    
    all_channel_metadata = []
    failed_channel_ids = []
    batch_size = snakemake.params.batch_size
    rate_limit_delay = snakemake.params.rate_limit_delay
    
    # Process in batches
    for batch_num, channel_ids_batch in enumerate(batch_channel_ids(channel_ids, batch_size), 1):
        print(f"Processing batch {batch_num}/{(len(channel_ids) + batch_size - 1) // batch_size} "
              f"({len(channel_ids_batch)} channels)...")
        
        # Add rate limiting delay
        if batch_num > 1:
            time.sleep(rate_limit_delay)
        
        # Fetch batch
        success_channels, failed_ids = fetch_channel_batch(youtube, channel_ids_batch)
        
        # Save raw responses
        if snakemake.config.get('preserve_raw_responses', True):
            save_raw_responses(success_channels, snakemake.output.api_responses_dir, batch_num)
        
        # Process successful channels
        for channel_item in success_channels:
            try:
                metadata = parse_channel_metadata(channel_item)
                all_channel_metadata.append(metadata)
            except Exception as e:
                print(f"Error parsing channel {channel_item.get('id', 'unknown')}: {e}")
                failed_channel_ids.append(channel_item.get('id', 'unknown'))
        
        # Track failed channels
        failed_channel_ids.extend(failed_ids)
        
        print(f"  Success: {len(success_channels)} channels")
        print(f"  Failed: {len(failed_ids)} channels")
    
    print(f"\nFetch completed:")
    print(f"  Total successful: {len(all_channel_metadata):,}")
    print(f"  Total failed: {len(failed_channel_ids):,}")
    
    if failed_channel_ids:
        print(f"  Failed channel IDs: {failed_channel_ids[:10]}...")  # Show first 10
    
    return all_channel_metadata, failed_channel_ids


def generate_channel_metadata_summary(channel_metadata_df):
    """Generate summary statistics for channel metadata."""
    print("\n=== CHANNEL METADATA SUMMARY ===")
    
    total_channels = len(channel_metadata_df)
    print(f"Total channels with metadata: {total_channels:,}")
    
    # Basic statistics
    print(f"\nBasic statistics:")
    print(f"  Channels with titles: {channel_metadata_df['title'].notna().sum():,}")
    print(f"  Channels with descriptions: {channel_metadata_df['description'].notna().sum():,}")
    print(f"  Channels with custom URLs: {channel_metadata_df['custom_url'].notna().sum():,}")
    print(f"  Channels with country info: {channel_metadata_df['country'].notna().sum():,}")
    
    # Subscriber statistics
    if (channel_metadata_df['subscriber_count'] > 0).any():
        print(f"\nSubscriber statistics:")
        sub_stats = channel_metadata_df[channel_metadata_df['subscriber_count'] > 0]['subscriber_count']
        print(f"  Channels with subscriber data: {len(sub_stats):,}")
        print(f"  Mean subscribers: {sub_stats.mean():,.0f}")
        print(f"  Median subscribers: {sub_stats.median():,.0f}")
        print(f"  Max subscribers: {sub_stats.max():,.0f}")
        print(f"  Min subscribers: {sub_stats.min():,.0f}")
        
        # Subscriber ranges
        print(f"\nSubscriber ranges:")
        print(f"  <1K subscribers: {(sub_stats < 1000).sum():,}")
        print(f"  1K-10K subscribers: {((sub_stats >= 1000) & (sub_stats < 10000)).sum():,}")
        print(f"  10K-100K subscribers: {((sub_stats >= 10000) & (sub_stats < 100000)).sum():,}")
        print(f"  100K-1M subscribers: {((sub_stats >= 100000) & (sub_stats < 1000000)).sum():,}")
        print(f"  1M+ subscribers: {(sub_stats >= 1000000).sum():,}")
    
    # Video count statistics
    if (channel_metadata_df['video_count'] > 0).any():
        print(f"\nVideo count statistics:")
        video_stats = channel_metadata_df[channel_metadata_df['video_count'] > 0]['video_count']
        print(f"  Channels with video count data: {len(video_stats):,}")
        print(f"  Mean videos per channel: {video_stats.mean():,.0f}")
        print(f"  Median videos per channel: {video_stats.median():,.0f}")
        print(f"  Max videos: {video_stats.max():,.0f}")
        print(f"  Min videos: {video_stats.min():,.0f}")
    
    # View count statistics
    if (channel_metadata_df['view_count'] > 0).any():
        print(f"\nChannel view statistics:")
        view_stats = channel_metadata_df[channel_metadata_df['view_count'] > 0]['view_count']
        print(f"  Channels with view data: {len(view_stats):,}")
        print(f"  Mean total views: {view_stats.mean():,.0f}")
        print(f"  Median total views: {view_stats.median():,.0f}")
        print(f"  Max total views: {view_stats.max():,.0f}")
    
    # Country distribution
    if 'country' in channel_metadata_df.columns:
        country_counts = channel_metadata_df['country'].value_counts()
        if len(country_counts) > 0:
            print(f"\nTop 10 countries:")
            for country, count in country_counts.head(10).items():
                print(f"  {country}: {count} channels")
    
    # Creation date analysis
    if 'published_at' in channel_metadata_df.columns:
        channel_metadata_df['created_year'] = pd.to_datetime(channel_metadata_df['published_at'], errors='coerce').dt.year
        year_counts = channel_metadata_df['created_year'].value_counts().sort_index()
        print(f"\nChannel creation year distribution (recent years):")
        for year, count in year_counts.tail(10).items():
            if pd.notna(year):
                print(f"  {int(year)}: {count} channels")
    
    # Top channels by subscribers
    if (channel_metadata_df['subscriber_count'] > 0).any():
        print(f"\nTop 10 channels by subscriber count:")
        top_channels = channel_metadata_df.nlargest(10, 'subscriber_count')
        for _, row in top_channels.iterrows():
            print(f"  {row['title']}: {row['subscriber_count']:,} subscribers")


def main():
    """Main function for fetching channel metadata."""
    # Get paths from Snakemake
    unique_channels_path = snakemake.input.unique_channels
    output_path = snakemake.output.channel_metadata
    
    # Load unique channels data
    print(f"Loading unique channels from {unique_channels_path}")
    unique_channels = pd.read_parquet(unique_channels_path)
    print(f"Loaded {len(unique_channels):,} unique channels")
    
    # Fetch channel metadata
    channel_metadata_list, failed_channel_ids = fetch_all_channel_metadata(unique_channels)
    
    # Convert to DataFrame
    if channel_metadata_list:
        channel_metadata_df = pd.DataFrame(channel_metadata_list)
        
        # Generate summary statistics
        generate_channel_metadata_summary(channel_metadata_df)
        
        # Save results
        print(f"\nSaving channel metadata to {output_path}")
        channel_metadata_df.to_parquet(output_path, index=False)
        
        print(f"\n✅ Channel metadata collection completed!")
        print(f"Successfully processed: {len(channel_metadata_df):,} channels")
        print(f"Failed to process: {len(failed_channel_ids):,} channels")
        print(f"Columns: {list(channel_metadata_df.columns)}")
        
    else:
        print("\n❌ No channel metadata collected!")
        # Create empty DataFrame with expected schema
        empty_df = pd.DataFrame(columns=[
            'channel_id', 'title', 'description', 'custom_url', 'published_at',
            'country', 'default_language', 'view_count', 'subscriber_count',
            'video_count', 'hidden_subscriber_count', 'keywords', 'unsubscribed_trailer',
            'privacy_status', 'is_linked', 'made_for_kids', 'api_fetch_date'
        ])
        empty_df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()