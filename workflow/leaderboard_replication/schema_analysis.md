# Schema Analysis: Preference Dataset vs Cleaned Pipeline

## Key Findings

### Preference Dataset (7k) Structure
- **Single flat table**: 102 columns with all data denormalized
- **Pre-computed metrics**: Citation counts and domain classifications already calculated
- **Conversation-level aggregation**: Metrics aggregated per conversation 
- **Model comparison format**: Direct A/B comparison with `conv_metadata` containing metrics for both sides

### Cleaned Pipeline (24k) Structure  
- **Normalized relational tables**: 4 separate tables (threads, questions, responses, citations)
- **Raw data**: Individual citations need to be aggregated and classified
- **Response-level granularity**: Data stored at individual response level
- **Flexible relationships**: Can be joined for different analysis needs

## Critical Differences for Leaderboard

### 1. Winner Information
- **Preference**: `winner` column with values like "model_a", "model_b", "tie"
- **Cleaned**: `winner` in threads table, but need to map to model names through responses

### 2. Model Identification
- **Preference**: Direct `model_a`, `model_b` columns
- **Cleaned**: Model names in responses table, need to reconstruct A/B pairing

### 3. Citation Metrics (Major Gap)
- **Preference**: 44 pre-computed citation columns per side:
  - `cites_*_a/b`: Boolean indicators (11 domains × 2 sides)
  - `num_cites_*_a/b`: Count metrics (11 domains × 2 sides)
  - `num_citations_a/b`: Total citation counts
- **Cleaned**: Raw citations table requires:
  - Domain classification (need domain categorization logic)
  - Aggregation by response and conversation
  - A/B side assignment

### 4. Response Metrics
- **Preference**: `response_length_a/b` pre-computed
- **Cleaned**: `response_text` needs length calculation

### 5. System Metadata
- **Preference**: Nested `system_a_metadata`, `system_b_metadata` 
- **Cleaned**: Flattened in responses table (citation_format, llm_*, search_*)

## Domain Categories Required
Both datasets use 11 domain categories:
1. `youtube` 2. `gov_edu` 3. `wiki` 4. `us_news` 5. `foreign_news`
6. `social_media` 7. `community_blog` 8. `tech_coding` 9. `map`
10. `academic_journal` 11. `other`

## Missing Components for Leaderboard
1. **Domain classification logic** - Need to categorize domains from citations
2. **Conversation aggregation** - Group citations/responses by thread and model side
3. **A/B reconstruction** - Map response sides to model_a/model_b format
4. **Standardized citation detection** - Determine citation format per response

## Next Steps
Need to create feature engineering pipeline to transform our normalized data into the format expected by leaderboard algorithms.