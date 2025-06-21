# Search Arena Data Cleaning Analysis Plan

## Overview
This document outlines the data cleaning strategy for the search-arena-chat-24k.parquet dataset. The file contains 24,069 rows with 53 columns representing complex nested conversation data between AI search systems.

## Data Structure Analysis

### Dataset Characteristics
- **Total rows**: 24,069 conversations
- **Total columns**: 53 (heavily nested)
- **File format**: Parquet with complex nested structures
- **Compression**: SNAPPY with varying compression ratios (16-62% savings)

### Key Data Hierarchies

#### 1. Core Conversation Fields
- `model_a`, `model_b`: AI model identifiers
- `winner`: Comparison result (nullable - only 12,652 non-null values)
- `judge`: Evaluation method
- `turn`: Conversation turn number
- `timestamp`: Conversation timestamp

#### 2. Message Structure (Nested Arrays)
- `messages_a.list.element.{content, role}`: Model A conversation messages
- `messages_b.list.element.{content, role}`: Model B conversation messages
- **Nesting levels**: 4 definition levels, 1 repetition level

#### 3. System Metadata (Highly Nested)
**System A & B Metadata Structure**:
- `system_{a,b}_metadata.citation_format_standardized`
- `system_{a,b}_metadata.client_country`
- `system_{a,b}_metadata.conv_id`

**LLM Configuration**:
- `system_{a,b}_metadata.llm_config.name`
- `system_{a,b}_metadata.llm_config.params.{max_tokens, temperature, top_p}`
- `system_{a,b}_metadata.llm_config.params.web_search_options.{search_context_size, user_location}`

**Web Search Configuration**:
- `system_{a,b}_metadata.web_search_config.{context_manager, scrape_engine, search_engine}`

#### 4. Critical Citation Data (Triple Nested)
**Web Search Trace** - Most complex field:
- `system_{a,b}_metadata.web_search_trace.list.element.list.element.list.element`
- **Nesting levels**: 8 definition levels, 3 repetition levels
- Contains citation URLs and reference numbers
- 46-47% compression ratio suggests high redundancy

#### 5. Intent Classification
- `primary_intent`: Main conversation purpose
- `secondary_intent`: Secondary purpose (13% compression suggests variety)
- `languages.list.element`: Language detection (nested array)

## Data Quality Observations

### Completeness Issues
1. **Winner field**: Only 52.5% completeness (12,652/24,069)
2. **Nested field sparsity**: High definition levels suggest many null values
3. **System metadata**: May have asymmetric completeness between system_a and system_b

### Potential Data Quality Challenges
1. **Citation extraction complexity**: Triple-nested web_search_trace structure
2. **URL parsing**: Need to handle malformed URLs in citation data
3. **Message encoding**: High compression in content fields suggests potential encoding issues
4. **Metadata inconsistency**: Different systems may have different metadata completeness

## Proposed Data Extraction Strategy

### Phase 1: Core Data Extraction
1. **Flatten conversation metadata**
   - Extract model identifiers, winner, judge, turn, timestamp
   - Create conversation-level unique identifiers

2. **Message extraction**
   - Unnest messages_a and messages_b arrays
   - Create message-level records with conversation context
   - Handle role-content pairs

### Phase 2: Citation Data Processing
1. **Web search trace extraction**
   - Parse triple-nested web_search_trace structure
   - Extract citation URLs and reference numbers
   - Handle citation format variations

2. **URL processing**
   - Extract domains from URLs
   - Normalize URL formats
   - Handle malformed URLs gracefully

### Phase 3: Metadata Normalization
1. **System configuration flattening**
   - Extract LLM parameters (temperature, top_p, max_tokens)
   - Normalize search configurations
   - Handle missing/null configurations

2. **Geographic and language processing**
   - Extract user location data
   - Process language detection results
   - Standardize country codes

### Phase 4: Intent and Classification
1. **Intent processing**
   - Clean primary/secondary intent fields
   - Create intent taxonomies
   - Handle missing classifications

## Implementation Plan

### Directory Structure
```
workflow/data_cleaning/
├── analysis_plan.md                 # This document
├── Snakefile                       # Main workflow
├── config/
│   └── config.yaml                 # Configuration parameters
├── scripts/
│   ├── 01_explore_structure.py     # Initial data exploration
│   ├── 02_extract_core.py          # Core conversation data
│   ├── 03_extract_messages.py      # Message unnesting
│   ├── 04_extract_citations.py     # Citation processing
│   ├── 05_extract_metadata.py      # Metadata flattening
│   ├── 06_process_intents.py       # Intent classification
│   └── 07_create_analysis_tables.py # Final analysis-ready tables
└── rules/
    ├── explore.smk                 # Exploration rules
    ├── extract.smk                 # Extraction rules
    └── clean.smk                   # Cleaning rules
```

### Expected Outputs
1. **Core conversation table**: `data/intermediate/cleaned_arena_data/conversations.parquet`
2. **Message table**: `data/intermediate/cleaned_arena_data/messages.parquet`
3. **Citation table**: `data/intermediate/cleaned_arena_data/citations.parquet`
4. **Metadata table**: `data/intermediate/cleaned_arena_data/metadata.parquet`
5. **Analysis-ready table**: `data/output/cleaned_arena_data/search_arena_clean.parquet`

### Data Validation Strategy
1. **Row count validation**: Ensure no data loss during transformations
2. **Citation completeness**: Validate URL extraction accuracy
3. **Metadata consistency**: Check for missing critical fields
4. **Domain extraction**: Verify domain parsing accuracy against known domains

## Next Steps for Implementation
1. Review and modify this plan based on specific analysis requirements
2. Implement initial exploration script to validate assumptions
3. Create Snakemake workflow structure
4. Develop extraction scripts incrementally
5. Test on sample data before full processing

## Notes for Modification
- Adjust extraction priorities based on specific research questions
- Consider memory constraints for large nested data processing
- Plan for incremental processing if full dataset is too large
- Add specific domain validation rules if needed for credibility analysis