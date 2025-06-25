# Search Arena Data Cleaning Analysis Plan

## Overview
This document outlines the data cleaning strategy for the search-arena-chat-24k.parquet dataset. The file contains 24,069 rows with 53 columns representing AI search arena comparisons.

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
    model_name VARCHAR,                 -- Model identifier (from system metadata)
    model_side CHAR(1),                -- 'a' or 'b' to track original assignment
    response_text TEXT,                -- Complete response content
    response_role VARCHAR,             -- 'assistant' typically
    citation_format VARCHAR,           -- Citation format used
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

#### 3. Citations Table (`citations.parquet`)
**Purpose**: Store individual citations extracted from web search traces
```sql
CREATE TABLE citations (
    citation_id VARCHAR PRIMARY KEY,     -- Unique citation identifier
    response_id VARCHAR REFERENCES responses(response_id),
    citation_number INT,                -- Reference number [1], [2], etc.
    url TEXT,                          -- Full URL
    domain VARCHAR,                    -- Extracted domain
    url_valid BOOLEAN,                 -- Whether URL is well-formed
    citation_order INT                 -- Order within response
);
```

### Relationships and Keys
- **Threads** (1) → **Questions** (1+): Each thread contains one or more questions (turns)
- **Questions** (1) → **Responses** (2): Each question generates exactly two responses
- **Responses** (1) → **Citations** (many): Each response can have multiple citations

**Key Insight**: The original `winner` field applies to the entire **thread**, not individual turns.

## Proposed Data Extraction Strategy

### Phase 1: Thread and Question Extraction
1. **Create thread identifiers**
   - Generate thread_id from original row index or conversation ID
   - Extract thread-level metadata (timestamp, winner, judge, intent, languages)
   - Determine total turns per thread from messages structure

2. **Extract questions per turn**
   - Parse messages_a/messages_b arrays to identify user messages
   - Create question_id for each user message/turn
   - Link questions to threads via thread_id
   - Track turn_number within each thread

### Phase 2: Responses Extraction
2. **Process model responses per turn**
   - Create response_id for each model (a/b) per question per turn
   - Extract response content from assistant messages in each turn
   - Flatten system metadata for each model response
   - Link responses to questions and threads via foreign keys
   - Track turn_number for proper sequencing

### Phase 3: Citations Extraction
1. **Web search trace processing**
   - Parse triple-nested web_search_trace structure for each response
   - Extract citation URLs and reference numbers
   - Create citation_id and link to response_id
   - Handle citation format variations

2. **URL processing and validation**
   - Extract domains from URLs
   - Validate URL format and accessibility
   - Normalize URL formats
   - Handle malformed URLs gracefully

### Phase 4: Comparisons Processing
1. **Evaluation data extraction**
   - Link responses back to comparison results
   - Extract winner, judge, and evaluation metadata
   - Create comparison records linking question to both responses

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
│   └── 06_validate_extraction.py  # Data validation and quality checks
└── rules/
    ├── explore.smk                 # Exploration rules
    ├── extract.smk                 # Extraction rules
    └── clean.smk                   # Cleaning rules
```

### Expected Outputs (Normalized Tables)
1. **Threads table**: `data/intermediate/cleaned_arena_data/threads.parquet`
   - 24,069 rows (one per conversation thread)
   - Contains thread metadata, overall winner, and evaluation data

2. **Questions table**: `data/intermediate/cleaned_arena_data/questions.parquet`
   - Variable rows (sum of all turns across all threads)
   - Contains user queries for each turn within threads

3. **Responses table**: `data/intermediate/cleaned_arena_data/responses.parquet`
   - 2× questions count (two per question: model_a and model_b responses)
   - Contains model responses and configuration metadata per turn

4. **Citations table**: `data/intermediate/cleaned_arena_data/citations.parquet`
   - 366,087 rows (~7.6 citations per response based on extraction)
   - Contains individual URLs, domains (full and base), and reference numbers
   - Primary focus for citation bias and credibility analysis
   - Ready for joining with external domain credibility and political leaning datasets

## Citation-Focused Analysis Approach

The primary analysis focus will be on the **citations table** as it contains the richest data for understanding AI model citation behavior and bias patterns. The normalized relational structure allows for flexible analysis where other tables can be joined as needed:

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

### Data Validation Strategy
1. **Referential integrity**: Ensure all foreign key relationships are valid
   - Every response links to exactly one question
   - Every citation links to exactly one response

2. **Row count validation**:
   - Threads: 24,069 rows (1:1 with original data)
   - Questions: 32,884 rows (sum of all turn counts)
   - Responses: 65,768 rows (2× questions count)
   - Citations: 366,087 rows (~7.6 per response)

   **Critical validation**: Sum of turns across all threads should equal total questions

3. **Data completeness validation**:
   - Validate URL extraction accuracy from web_search_trace
   - Check for missing critical fields (user_query, model_name, etc.)
   - Verify domain parsing accuracy against known domains

4. **Business logic validation**:
   - Each thread should have ≥1 questions (turns)
   - Each question should have exactly 2 responses (model_a and model_b)
   - Turn numbers should be sequential within each thread (1, 2, 3...)
   - Thread winner field should reference actual model names
   - Citation numbers should be sequential within each response

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