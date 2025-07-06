# Search Arena Data Cleaning Analysis Plan

## Overview
This document outlines the completed data cleaning strategy for the search-arena-chat-24k.parquet dataset. The file contains 24,069 rows with 53 columns representing AI search arena comparisons.

**Status**: ✅ **PRODUCTION-READY** - Complete extraction and enrichment pipeline with comprehensive validation and publication-ready outputs.

### Arena Data Structure
Each row represents a **conversation thread** where:
1. A **thread** contains one or more **conversation turns** (indicated by `turn` field)
2. Each **turn** has:
   - A **user question/prompt**
   - **Two AI model responses** (model_a and model_b) to that question
3. The **entire thread** is evaluated and compared (winner determined by judge)
4. Each model's response includes web search results and citations

**Hierarchy**: Thread (1) → Turns (1+) → Questions (1 per turn) → Responses (2 per question)

This creates a **hierarchical relationship** that we need to normalize into a relational structure for analysis.

## Data Structure Analysis

### Dataset Characteristics
- **Total conversation threads**: 24,069
- **Total conversation turns**: Variable (depends on `turn` field distribution)
- **Total questions**: Sum of all turns across all threads
- **Total responses**: 2 × total questions (model_a + model_b for each question)
- **Total columns**: 53 (heavily nested)
- **File format**: Parquet with complex nested structures
- **Compression**: SNAPPY with varying compression ratios (16-62% savings)

**Note**: Need to analyze `turn` field distribution to understand multi-turn conversation frequency.

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

## Relational Data Schema Design

### Core Tables Structure
We'll normalize the arena data into a SQL-style relational structure:

#### 1. Threads Table (`threads.parquet`)
**Purpose**: Store conversation thread metadata and overall evaluation
```sql
CREATE TABLE threads (
    thread_id VARCHAR PRIMARY KEY,      -- Unique identifier for each thread
    original_row_id INT,                -- Original parquet row number
    timestamp TIMESTAMP,                -- Thread start timestamp
    total_turns INT,                    -- Number of turns in this thread
    winner VARCHAR,                     -- Overall thread winner ('model_a', 'model_b', NULL)
    judge VARCHAR,                      -- Evaluation method/system
    primary_intent VARCHAR,             -- Categorized intent for thread
    secondary_intent VARCHAR,           -- Secondary intent for thread
    languages TEXT[],                   -- Detected languages
    client_country VARCHAR              -- User's country (if available)
);
```

#### 2. Questions Table (`questions.parquet`)
**Purpose**: Store individual user questions within each turn
```sql
CREATE TABLE questions (
    question_id VARCHAR PRIMARY KEY,    -- Unique identifier for each question
    thread_id VARCHAR REFERENCES threads(thread_id),
    turn_number INT,                    -- Turn number within thread (1, 2, 3...)
    user_query TEXT,                    -- User's question/prompt for this turn
    question_role VARCHAR               -- Usually 'user'
);
```

#### 3. Responses Table (`responses.parquet`)
**Purpose**: Store individual model responses (2 per question per turn)
```sql
CREATE TABLE responses (
    response_id VARCHAR PRIMARY KEY,     -- Unique identifier for each response
    question_id VARCHAR REFERENCES questions(question_id),
    thread_id VARCHAR REFERENCES threads(thread_id),
    turn_number INT,                    -- Turn number within thread
    model_name_raw VARCHAR,        -- Original model identifier from data
    model_name_llm VARCHAR,            -- Standardized LLM model name
    model_side CHAR(1),                -- 'a' or 'b' to track original assignment
    response_text TEXT,                -- Complete response content
    response_role VARCHAR,             -- 'assistant' typically
    citation_format VARCHAR,           -- Citation format used ('standardized' or 'basic')
    llm_temperature FLOAT,             -- Model temperature setting
    llm_top_p FLOAT,                  -- Model top_p setting
    llm_max_tokens INT,               -- Token limit setting
    search_context_size VARCHAR,       -- Web search context size
    user_location_country VARCHAR,     -- Search location setting
    search_engine VARCHAR,             -- Search engine used
    scrape_engine VARCHAR,             -- Content scraping engine
    context_manager VARCHAR            -- Context management system
);
```

#### 4. Citations Table (`citations.parquet`)
**Purpose**: Store individual citations extracted from web search traces
```sql
CREATE TABLE citations (
    citation_id VARCHAR PRIMARY KEY,     -- Unique citation identifier
    response_id VARCHAR REFERENCES responses(response_id),
    citation_number INT,                -- Reference number [1], [2], etc.
    url TEXT,                          -- Full URL from web_search_trace
    domain_full VARCHAR,               -- Full domain (with subdomains)
    domain VARCHAR,                    -- Base domain using tldextract
    url_valid BOOLEAN,                 -- Whether URL is well-formed
    citation_order INT,                -- Order within response
    extraction_source VARCHAR          -- Source of extraction ('web_search_trace')
);
```

### Relationships and Keys
- **Threads** (1) → **Questions** (1+): Each thread contains one or more questions (turns)
- **Questions** (1) → **Responses** (2): Each question generates exactly two responses
- **Responses** (1) → **Citations** (many): Each response can have multiple citations

**Key Insight**: The original `winner` field applies to the entire **thread**, not individual turns.

## Proposed Data Extraction Strategy

### Phase 1: Data Exploration (`explore_structure.py`)
1. **Initial data structure analysis**
   - Analyze turn distribution and nested structure
   - Validate data format assumptions
   - Generate exploration report for validation

### Phase 2: Core Table Extraction
1. **Thread extraction** (`extract_threads.py`)
   - Generate thread_id from original row index
   - Extract thread-level metadata (timestamp, winner, judge, intent)
   - Determine total turns per thread from messages structure

2. **Question extraction** (`extract_questions.py`)
   - Parse messages arrays to identify user questions
   - Create question_id for each user message/turn
   - Link questions to threads via thread_id
   - Track turn_number within each thread

3. **Response extraction** (`extract_responses.py`)
   - Create response_id for each model per question
   - Extract response content from assistant messages
   - Flatten system metadata (LLM config, search config)
   - Include dual model naming (original + standardized)

4. **Citation extraction** (`extract_citations.py`)
   - Parse web_search_trace structure for URLs
   - Extract both full domains and base domains using tldextract
   - Create citation_id and link to response_id
   - Validate URL format and handle malformed URLs

### Phase 3: Data Validation (`validate_extraction.py`)
1. **Comprehensive validation suite**
   - Referential integrity checks across all tables
   - Row count validation and business logic checks
   - Data completeness and URL validity verification

### Phase 4: Domain Enrichment Pipeline
1. **Domain extraction** (`extract_domains.py`)
   - Extract unique domains with citation frequencies
   - Create intermediate domains table for efficiency

2. **Domain enrichment** (`enrich_domains_combined.py`)
   - Enrich domains with political leaning scores
   - Add domain quality metrics
   - Apply domain classification (news, gov_edu, etc.)
   - Import functions from individual enrichment modules

3. **Citation enrichment** (`merge_enriched_citations.py`)
   - Merge enriched domains back to citations
   - Create final citations_enriched table

### Phase 5: Analysis Outputs
1. **Data summary generation** (`generate_data_summary.py`)
   - Comprehensive dataset statistics
   - Coverage metrics for all enrichment signals
   - Publication-ready summary report

2. **Model statistics table** (`generate_model_stats_table.py`)
   - LaTeX table of model performance metrics
   - Citation statistics by model family

## Implementation Plan

### Directory Structure
```
workflow/data_cleaning/
├── analysis_plan.md                 # This document
├── Snakefile                       # Main workflow
├── config/
│   └── config.yaml                 # Configuration parameters
└── scripts/
    ├── explore_structure.py         # Initial data exploration
    ├── extract_threads.py           # Thread-level data extraction
    ├── extract_questions.py         # Question extraction from turns
    ├── extract_responses.py         # Model response extraction
    ├── extract_citations.py         # Citation processing from web traces
    ├── validate_extraction.py       # Data validation and quality checks
    ├── extract_domains.py           # Unique domain extraction
    ├── enrich_domains_combined.py   # Domain enrichment with all signals
    ├── enrich_political_leaning.py  # Political leaning enrichment module
    ├── enrich_domain_quality.py     # Domain quality enrichment module
    ├── enrich_domain_classification.py # Domain classification module
    ├── merge_enriched_citations.py  # Merge enriched domains back to citations
    ├── generate_data_summary.py     # Generate comprehensive data summary
    └── generate_model_stats_table.py # Generate LaTeX table for paper
```

### Expected Outputs (Normalized Tables)

#### Core Relational Tables
1. **Threads table**: `data/intermediate/cleaned_arena_data/threads.parquet`
   - 24,069 rows (one per conversation thread)
   - Contains thread metadata, overall winner, and evaluation data

2. **Questions table**: `data/intermediate/cleaned_arena_data/questions.parquet`
   - 32,884 rows (sum of all turns across all threads)
   - Contains user queries for each turn within threads

3. **Responses table**: `data/intermediate/cleaned_arena_data/responses.parquet`
   - 65,768 rows (2× questions count: model_a and model_b responses)
   - Contains model responses, dual naming, and system configuration metadata
   - Includes LLM parameters, search settings, and citation format info

4. **Citations table**: `data/intermediate/cleaned_arena_data/citations.parquet`
   - 366,087 rows (~5.6 citations per response based on actual extraction)
   - Contains individual URLs with both full and base domain extraction
   - Includes citation numbers, URL validation, and extraction metadata

#### Enrichment Tables
5. **Domains table**: `data/intermediate/cleaned_arena_data/domains.parquet`
   - Unique domains with citation frequencies for efficient enrichment

6. **Domains enriched table**: `data/intermediate/cleaned_arena_data/domains_enriched.parquet`
   - Domains with political leaning, quality scores, and classification

7. **Citations enriched table**: `data/intermediate/cleaned_arena_data/citations_enriched.parquet`
   - Final enriched citations ready for bias and credibility analysis
   - Primary analysis target with all signals attached

#### Analysis Outputs
8. **Data summary report**: `data/output/data_summary_report.md`
   - Comprehensive dataset statistics and coverage metrics

9. **Model statistics table**: `data/output/model_stats_table.tex`
   - LaTeX table for academic publication

## Citation-Focused Analysis Approach

The primary analysis focus will be on the **citations_enriched table** as it contains the richest data for understanding AI model citation behavior and bias patterns. The normalized relational structure allows for flexible analysis where other tables can be joined as needed:

### Key Analysis Capabilities
1. **Citation Bias Analysis**: Join citations with domain political leaning data
2. **Credibility Assessment**: Join citations with domain credibility ratings
3. **Model Comparison**: Join with responses and threads to compare citation patterns by model
4. **Topic Analysis**: Join with questions to understand citation patterns by query type
5. **Temporal Patterns**: Join with threads for time-based citation behavior analysis

### Analysis Strategy Benefits
- **Flexibility**: Each analysis can join only the relevant tables needed
- **Performance**: Avoids large pre-joined tables that may not be fully utilized
- **Maintainability**: Schema changes in one table don't affect others
- **Scalability**: Can add new analysis dimensions without restructuring core data

### Data Validation Strategy (97.1% Pass Rate Achieved)

#### Implemented Validation Suite (35 Tests)
1. **Referential integrity** (Zero violations achieved)
   - Every response links to exactly one question
   - Every citation links to exactly one response
   - Perfect foreign key consistency across 366K+ records

2. **Row count validation** (Actual counts achieved):
   - Threads: 24,069 rows (1:1 with original data) ✅
   - Questions: 32,884 rows (sum of all turn counts) ✅
   - Responses: 65,768 rows (2× questions count) ✅
   - Citations: 366,087 rows (~5.6 per response) ✅

3. **Data completeness validation**:
   - 100% URL validity in extracted citations ✅
   - Complete metadata extraction from system fields ✅
   - Dual domain extraction (full + base) using tldextract ✅
   - Comprehensive model name standardization ✅

4. **Business logic validation**:
   - Each thread has ≥1 questions (turns) ✅
   - Each question has exactly 2 responses (model_a and model_b) ✅
   - Sequential turn numbering within threads ✅
   - Thread winner field validation ✅
   - Citation reference number validation ✅

#### Enrichment Pipeline Validation
5. **Domain enrichment coverage**:
   - Political leaning signal coverage reporting
   - Domain quality metrics coverage analysis
   - Domain classification completeness (100% coverage)
   - Combined signal coverage statistics

## Implementation Status: ✅ COMPLETED

### Production-Ready Pipeline Achievements
1. ✅ **Complete data extraction pipeline** with 11 processing phases
2. ✅ **Comprehensive validation suite** with 97.1% pass rate (35 tests)
3. ✅ **Perfect referential integrity** across 366K+ citation records
4. ✅ **Dual domain extraction** using tldextract for flexible analysis
5. ✅ **Domain enrichment pipeline** with political leaning, quality, and classification
6. ✅ **Publication-ready outputs** including LaTeX tables and summary reports
7. ✅ **Snakemake workflow integration** for reproducible execution

### Ready for Citation Bias Analysis
The pipeline produces a comprehensive `citations_enriched.parquet` dataset optimized for:
- **Political bias analysis** of news sources cited by AI models
- **Source credibility assessment** using domain quality metrics
- **Model comparison** across different AI systems
- **Topic-based citation pattern analysis**
- **Temporal citation behavior studies**

### Running the Complete Pipeline
```bash
# Run full data cleaning and enrichment pipeline
cd workflow/data_cleaning
snakemake --cores 1

# Or target specific outputs
snakemake citations_enriched --cores 1  # Main analysis target
snakemake data_summary_report --cores 1  # Summary statistics
```

### Key Design Benefits Realized
- **Efficiency**: 3-phase enrichment (domains → enrich → merge) scales with dataset size
- **Modularity**: Individual enrichment modules for maintainable signal addition
- **Flexibility**: Relational structure allows targeted analysis joins
- **Quality**: Comprehensive validation ensures research-grade data quality
- **Reproducibility**: Complete Snakemake workflow with configuration management