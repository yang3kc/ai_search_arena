# Question-News Citation Pattern Analysis Plan

## Overview
This analysis aims to understand how different features of user questions relate to AI models' news citation patterns. The workflow involves filtering target questions, generating question features, computing news citation patterns, and performing regression analysis.

## Data Pipeline Architecture

### Phase 1: Data Preparation and Question Filtering
**Goal**: Filter and prepare English questions for analysis

**Scripts to create**:
- `scripts/filter_english_questions.py`
  - Load questions from `data/intermediate/cleaned_arena_data/questions.parquet`
  - Detect English language using `langdetect` library
  - Filter out non-English questions
  - Output: `data/intermediate/question_analysis/english_questions.parquet`

### Phase 2: Question Embeddings Generation
**Goal**: Generate semantic embeddings for questions

**Scripts to create**:
- `scripts/generate_question_embeddings.py`
  - **Text embeddings**: Use sentence transformers (e.g., `all-MiniLM-L6-v2`) to generate semantic embeddings
  - **Batch processing**: Efficient processing of large question datasets
  - **Embedding validation**: Ensure embedding quality and consistency
  - Output: `data/intermediate/question_analysis/question_embeddings.parquet`

### Phase 3: Question Feature Extraction
**Goal**: Extract features from questions

**Scripts to create**:
- `scripts/extract_question_features.py`
  - **Question intent**: The primary intent of the question, from the thread metadata
  - **Client country**: The country of the client, from the thread metadata
  - **Question characteristics**: Length (characters, words)
  - Output: `data/intermediate/question_analysis/question_features.parquet`

### Phase 4: News Citation Pattern Variables
**Goal**: Compute response-level citation metrics (reuse from preference analysis)

**Scripts to adapt**:
- `scripts/compute_citation_patterns.py` (adapted from `preference_analysis/compute_response_signals.py`)
  - Reuse existing citation pattern computation logic
  - Key metrics to include:
    - Proportion of left/right/center-leaning sources
    - Proportion of high/low quality sources
    - News vs non-news citation ratios
    - Domain diversity metrics
    - Overrepresentation scores for specific news sources
  - Output: `data/intermediate/question_analysis/citation_patterns.parquet`

### Phase 5: Data Integration
**Goal**: Merge question features with citation patterns and model metadata

**Scripts to create**:
- `scripts/integrate_analysis_data.py`
  - Join questions with responses, citations, and threads
  - Merge question embeddings and features with citation patterns
  - Add model family and model-specific variables
  - Handle missing data and create analysis-ready dataset
  - Output: `data/intermediate/question_analysis/integrated_analysis_data.parquet`

### Phase 6: Regression Analysis
**Goal**: Analyze relationships between question features and citation patterns

**Scripts to create**:
- `scripts/regression_analysis.py`
  - **Dependent variables**: News citation patterns (% left-leaning, % high-quality, etc.)
  - **Independent variables**:
    - Question features (embeddings, political classification, length, etc.)
    - Model variables (model family, model side)
    - Control variables (turn number, thread characteristics)
  - **Models to fit**:
    - Linear regression for continuous outcomes
    - Logistic regression for binary outcomes
    - Mixed-effects models accounting for thread/user clustering
  - **Analysis outputs**:
    - Coefficient tables with confidence intervals
    - Feature importance rankings
    - Model diagnostics and fit statistics
  - Output: `data/output/question_analysis/regression_results.json`

### Phase 7: Visualization and Reporting
**Goal**: Generate comprehensive analysis report

**Scripts to create**:
- `scripts/generate_question_analysis_report.py`
  - Summary statistics for question features
  - Correlation matrices between features and outcomes
  - Regression coefficient plots
  - Feature importance visualizations
  - Subgroup analyses (by model family, question type)
  - Output: `data/output/question_analysis/question_analysis_report.html`

## Technical Implementation Details

### Dependencies Required
```python
# New dependencies to add
sentence-transformers==2.2.2
langdetect==1.0.9
spacy>=3.4.0
scikit-learn>=1.1.0
statsmodels>=0.14.0
plotly>=5.0.0
```

### Key Data Structures

**Question Features Schema**:
```
- question_id (str)
- thread_id (str)
- embedding_dim_0 to embedding_dim_383 (float) # sentence transformer embeddings
- is_political (bool)
- topic_category (str)
- question_length_chars (int)
- question_length_words (int)
- question_type (str) # what, how, why, etc.
- has_temporal_reference (bool)
- complexity_score (float)
- named_entities_count (int)
- language_confidence (float)
```

**Citation Patterns Schema** (reused from preference analysis):
```
- response_id (str)
- proportion_left_leaning (float)
- proportion_right_leaning (float)
- proportion_center_leaning (float)
- proportion_high_quality (float)
- proportion_low_quality (float)
- news_proportion_* (float) # news-specific metrics
- num_citations (int)
- domain_diversity_score (float)
```

### Snakemake Workflow Structure
```python
rule all:
    input:
        "data/output/question_analysis/question_analysis_report.html",
        "data/output/question_analysis/regression_results.json"

rule filter_english_questions:
    input:
        questions="data/intermediate/cleaned_arena_data/questions.parquet",
        threads="data/intermediate/cleaned_arena_data/threads.parquet"
    output: "data/intermediate/question_analysis/english_questions.parquet"

rule generate_question_embeddings:
    input: "data/intermediate/question_analysis/english_questions.parquet"
    output: "data/intermediate/question_analysis/question_embeddings.parquet"

rule extract_question_features:
    input:
        questions="data/intermediate/question_analysis/english_questions.parquet",
        threads="data/intermediate/cleaned_arena_data/threads.parquet"
    output: "data/intermediate/question_analysis/question_features.parquet"

rule compute_citation_patterns:
    input:
        questions="data/intermediate/question_analysis/english_questions.parquet",
        citations="data/intermediate/citation_analysis/news_citations.parquet",
        responses="data/intermediate/cleaned_arena_data/responses.parquet"
    output: "data/intermediate/question_analysis/citation_patterns.parquet"

rule integrate_analysis_data:
    input:
        question_embeddings="data/intermediate/question_analysis/question_embeddings.parquet",
        question_features="data/intermediate/question_analysis/question_features.parquet",
        citation_patterns="data/intermediate/question_analysis/citation_patterns.parquet",
        threads="data/intermediate/cleaned_arena_data/threads.parquet"
    output: "data/intermediate/question_analysis/integrated_analysis_data.parquet"

rule regression_analysis:
    input: "data/intermediate/question_analysis/integrated_analysis_data.parquet"
    output: "data/output/question_analysis/regression_results.json"

rule generate_report:
    input:
        data="data/intermediate/question_analysis/integrated_analysis_data.parquet",
        results="data/output/question_analysis/regression_results.json"
    output: "data/output/question_analysis/question_analysis_report.html"
```

## Expected Research Questions

1. **Political Questions**: Do political questions lead to more politically biased news citations?
2. **Question Complexity**: Do more complex questions result in higher-quality or more diverse news sources?
3. **Model Differences**: Do different model families respond differently to question characteristics?
4. **Topic Effects**: How do different question topics influence citation patterns?
5. **Temporal Effects**: Do questions about current events lead to different citation behaviors?

## Validation Strategy

1. **Language Detection Validation**: Manual review of sample questions for language accuracy
2. **Feature Quality Checks**: Correlation analysis between different feature types
3. **Citation Pattern Validation**: Consistency checks with existing preference analysis results
4. **Model Diagnostics**: Residual analysis, multicollinearity checks, cross-validation
5. **Robustness Tests**: Analysis across different subsets and alternative specifications

## Output Deliverables

1. **Analysis Dataset**: Clean, integrated dataset ready for modeling
2. **Regression Results**: Comprehensive statistical analysis results
3. **HTML Report**: Interactive report with visualizations and interpretations
4. **LaTeX Tables**: Regression tables formatted for paper inclusion
5. **Feature Importance Rankings**: Ranked lists of most predictive question features