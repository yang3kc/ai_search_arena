# YouTube Analysis Pipeline Plan

## Overview
This pipeline analyzes ~19,749 YouTube video citations from the AI search arena data to understand what types of YouTube content AI chatbots reference in their responses. The pipeline extracts video metadata, channel information, and creates structured datasets for comprehensive analysis.

## Research Questions
- What types of YouTube content do AI chatbots cite most frequently?
- Which YouTube channels are most commonly referenced?
- What are the characteristics of cited videos (duration, view count, age)?
- Do different AI models show different citation patterns for YouTube content?
- What topics/categories dominate YouTube citations?

## Pipeline Architecture

### Phase 1: YouTube URL Extraction
- **Script**: `extract_youtube_urls.py`
- **Input**: `../../../data/intermediate/cleaned_arena_data/citations.parquet` (366K citations)
- **Output**: `../../../data/intermediate/youtube_analysis/youtube_citations.parquet` (~19,749 YouTube URLs)
- **Function**:
  - Filter citations for actual YouTube videos (youtube.com domain and youtu.be URLs)
  - Extract video IDs from various YouTube URL formats
  - Deduplicate based on video ID
  - Preserve citation context (response_id, citation_number, etc.)

### Phase 2: Video Metadata Collection
- **Script**: `fetch_video_metadata.py`
- **Input**: `../../../data/intermediate/youtube_analysis/youtube_citations.parquet`
- **Output**:
  - `../../../data/intermediate/youtube_analysis/video_metadata.parquet` (structured data)
  - `../../../data/intermediate/youtube_analysis/video_api_responses/` (raw JSON responses by date)
- **Function**:
  - Query YouTube Data API v3 for video details
  - Collect: title, description, duration, view_count, like_count, published_date
  - Collect: channel_id, channel_title, category_id, tags, language
  - Handle deleted/private videos gracefully
  - Batch API calls (50 videos per request) for efficiency

### Phase 3: Channel Information Extraction
- **Script**: `extract_channel_data.py`
- **Input**: `../../../data/intermediate/youtube_analysis/video_metadata.parquet`
- **Output**: `../../../data/intermediate/youtube_analysis/unique_channels.parquet`
- **Function**:
  - Extract unique channel IDs from video metadata
  - Calculate citation frequency per channel
  - Prepare for channel metadata collection

### Phase 4: Channel Metadata Collection
- **Script**: `fetch_channel_metadata.py`
- **Input**: `data/unique_channels.parquet`
- **Output**:
  - `../../../data/intermediate/youtube_analysis/channel_metadata.parquet` (structured data)
  - `../../../data/intermediate/youtube_analysis/channel_api_responses/` (raw JSON responses by date)
- **Function**:
  - Query YouTube API for channel statistics and information
  - Collect: subscriber_count, video_count, view_count, created_date
  - Collect: channel description, country, custom_url
  - Batch API calls for efficiency

### Phase 5: Data Integration
- **Script**: `integrate_youtube_data.py`
- **Input**: All previous outputs + original citations
- **Output**: `../../../data/intermediate/youtube_analysis/youtube_analysis_dataset.parquet`
- **Function**:
  - Join video metadata with channel metadata
  - Link back to original citations with full context
  - Create analysis-ready dataset with all YouTube information
  - Generate summary statistics and data quality report

## Technical Implementation Details

### API Efficiency & Rate Limiting
- **Batch Requests**: 50 videos/channels per API call (YouTube API maximum)
- **Rate Limiting**: Respect YouTube API quotas (10,000 units/day default)
- **Request Costs**:
  - Video details: 1 unit per video
  - Channel details: 1 unit per channel
  - Estimated total: ~20K units for videos + ~5K for channels
- **Retry Logic**: Handle temporary API failures with exponential backoff

### Data Quality & Error Handling
- **Invalid Videos**: Handle deleted, private, or unavailable videos
- **Missing Data**: Graceful handling of missing fields in API responses
- **Data Validation**: Verify video IDs, check for API response completeness
- **Incremental Processing**: Support resuming interrupted API collection runs

### YouTube URL Parsing
Support multiple YouTube URL formats:
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`
- `https://m.youtube.com/watch?v=VIDEO_ID` (mobile)

### Data Schema

#### YouTube Citations (`youtube_citations.parquet`)
- `citation_id`: Original citation identifier
- `response_id`: Response that contained this citation
- `citation_number`: Order within response
- `url`: Original YouTube URL
- `video_id`: Extracted YouTube video ID
- `citation_order`: Global citation order

#### Video Metadata (`video_metadata.parquet`)
- `video_id`: YouTube video identifier
- `title`: Video title
- `description`: Video description (truncated to first 500 chars)
- `duration`: Video duration in ISO 8601 format
- `view_count`: Number of views
- `like_count`: Number of likes
- `comment_count`: Number of comments
- `published_at`: Publication date
- `channel_id`: Channel identifier
- `channel_title`: Channel name
- `category_id`: YouTube category ID
- `default_language`: Video language
- `tags`: Video tags (JSON array)
- `captions_available`: Whether captions are available
- `api_fetch_date`: When metadata was collected

#### Channel Metadata (`channel_metadata.parquet`)
- `channel_id`: YouTube channel identifier
- `title`: Channel name
- `description`: Channel description (truncated)
- `custom_url`: Channel custom URL
- `published_at`: Channel creation date
- `country`: Channel country
- `view_count`: Total channel views
- `subscriber_count`: Number of subscribers
- `video_count`: Number of videos
- `api_fetch_date`: When metadata was collected

#### Final Dataset (`youtube_analysis_dataset.parquet`)
Combined dataset with all citation context, video metadata, and channel metadata for comprehensive analysis.

## Dependencies
- **YouTube Data API v3**: Requires API key in `.env` file
- **Python packages**:
  - `google-api-python-client` (YouTube API)
  - `pandas` (data manipulation)
  - `snakemake` (workflow management)
  - `python-dotenv` (environment variables)
  - `urllib.parse` (URL parsing)
- **Rate limiting**: Custom utilities for API quota management

## Expected Outcomes
- **Video Coverage**: ~19,749 unique YouTube videos with full metadata
- **Channel Coverage**: ~5,000-10,000 unique YouTube channels with statistics
- **Citation Analysis**: Complete linkage between citations and YouTube content
- **Content Insights**: Understanding of what YouTube content AI systems reference
- **API Archive**: Complete archive of raw API responses for future analysis

## Usage
```bash
cd workflow/analysis/youtube_analysis
snakemake --cores 1  # Run complete pipeline
snakemake data/youtube_citations.parquet --cores 1  # Run Phase 1 only
```

## Quality Assurance
- Data validation at each pipeline stage
- API response archival for debugging
- Summary statistics and coverage reports
- Error logging and handling documentation