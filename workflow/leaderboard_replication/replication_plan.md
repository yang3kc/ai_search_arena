# Search Arena Leaderboard Replication Plan

## Overview
Replicate the leaderboard generation process from the 7k preference dataset, then adapt it to work with our cleaned 24k dataset pipeline.

## Phase 1: Data Source Analysis & Validation
1. **Compare dataset schemas**: Map fields between `search-arena-v1-preference-7k.parquet` (102 columns) and our cleaned data pipeline
2. **Identify key differences**: 
   - Preference dataset has pre-computed citation metrics (`cites_*`, `num_cites_*`)
   - Our pipeline has normalized relational tables (threads, questions, responses, citations)
3. **Validate shared notebook**: Test the existing leaderboard code with the 7k preference dataset

## Phase 2: Feature Engineering Pipeline
1. **Citation metrics computation**: Create functions to calculate domain-specific citation counts from our citations table
2. **Response metrics extraction**: Compute response length, citation counts per response from our normalized data
3. **Conversation metadata**: Aggregate metrics at conversation level to match preference dataset format

## Phase 3: Leaderboard Implementation
1. **Core algorithms**: Implement Bradley-Terry (BT) rating system with bootstrapping for confidence intervals
2. **Style control**: Add contextual BT for controlling citation format, response length, and domain preferences
3. **Visualization**: Create charts for ratings, confidence intervals, win rates, and feature coefficients

## Phase 4: Integration with Snakemake Pipeline
1. **New script**: `07_generate_leaderboard.py` to compute leaderboard from cleaned data
2. **Configuration**: Add leaderboard parameters to `config.yaml`
3. **Output**: Generate leaderboard tables and visualizations in `data/output/leaderboard/`

## Phase 5: Validation & Comparison
1. **Replication test**: Verify our implementation matches results from the 7k preference dataset
2. **Full dataset results**: Generate leaderboard using all 24k conversations from our pipeline
3. **Analysis differences**: Document any ranking changes when using the full dataset vs. 7k subset

## Deliverables
- Functional leaderboard generation integrated into our Snakemake workflow
- Comparison report showing replication accuracy and full dataset insights
- Production-ready code for ongoing leaderboard updates

## Implementation Notes

### Key Data Mappings
From the analysis of both datasets:

**Preference Dataset (7k)**: 
- Contains 102 columns with pre-computed metrics
- Has `conv_metadata` with domain citation counts (`cites_*_a/b`, `num_cites_*_a/b`)
- Includes system metadata for both models A and B
- Winner determination already available

**Our Cleaned Pipeline (24k)**:
- Normalized into 4 tables: threads, questions, responses, citations
- Citations table has 366k records with URL and domain extraction
- Responses linked to questions, questions linked to threads
- Need to compute conversation-level metrics from normalized data

### Critical Functions from Shared Code
1. **Bradley-Terry Rating**: `compute_bt()` - Core ranking algorithm
2. **Style Control**: `compute_style_control()` - Contextual ranking with feature control
3. **Bootstrap Confidence**: `compute_bootstrap_bt()` - Statistical confidence intervals
4. **Visualization**: Multiple plotting functions for leaderboard presentation

### Domain Categories
The leaderboard uses 11 domain categories:
- `youtube`, `gov_edu`, `wiki`, `us_news`, `foreign_news`
- `social_media`, `community_blog`, `tech_coding`, `map`
- `academic_journal`, `other`

### Integration Strategy
1. Extract and adapt core functions from the notebook
2. Create feature engineering pipeline to match preference dataset format
3. Implement as Snakemake rule with proper input/output specifications
4. Add configuration for different leaderboard variants (style control, domain filtering)