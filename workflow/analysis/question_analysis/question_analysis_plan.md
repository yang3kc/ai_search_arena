# Question-News Citation Pattern Analysis Plan

## Overview
**Status**: ✅ **PRODUCTION-READY** - Complete question analysis pipeline with topic modeling, regression analysis, and publication-ready outputs.

This analysis understands how different features of user questions relate to AI models' news citation patterns. The workflow involves filtering target questions, generating question features, computing news citation patterns, and performing regression analysis.

## Data Pipeline Architecture

### Phase 1: Data Preparation and Question Filtering ✅ **COMPLETED**
**Goal**: Filter and prepare English questions for analysis

**Scripts completed**:
- `scripts/filter_english_questions.py` ✅
  - Load questions from `data/intermediate/cleaned_arena_data/questions.parquet`
  - Detect English language using `langdetect` library
  - Filter out non-English questions
  - Output: `data/intermediate/question_analysis/english_questions.parquet`

### Phase 2: Question Embeddings Generation ✅ **COMPLETED**
**Goal**: Generate semantic embeddings for questions

**Scripts completed**:
- `scripts/generate_question_embeddings.py` ✅
  - **Text embeddings**: Use sentence transformers (`all-MiniLM-L6-v2`) to generate semantic embeddings
  - **Batch processing**: Efficient processing of large question datasets
  - **Embedding validation**: Ensure embedding quality and consistency
  - Output: `data/intermediate/question_analysis/question_embeddings.parquet`

### Phase 3: Question Topic Modeling ✅ **COMPLETED**
**Goal**: Extract topics from questions

**Scripts created**:
- `scripts/generate_question_topics.py` ✅
  - **Topic modeling**: Use BERTopic to extract topics and topic probabilities from questions
  - **Outlier reduction**: Minimize outliers using embedding-based strategy
  - **Topic distributions**: Generate probability distributions for each question
  - Output:
    - `data/intermediate/question_analysis/question_topics.parquet` (topic assignments)
    - `data/intermediate/question_analysis/question_topic_probabilities.parquet` (topic probabilities)
    - `data/intermediate/question_analysis/topic_info.json` (topic metadata and keywords)

### Phase 4: Question Feature Extraction ✅ **COMPLETED**
**Goal**: Extract features from questions

**Scripts completed**:
- `scripts/extract_question_features.py` ✅
  - **Question intent**: The primary intent of the question, from the thread metadata
  - **Client country**: The country of the client, from the thread metadata
  - **Question characteristics**: Length (characters, words)
  - Output: `data/intermediate/question_analysis/question_features.parquet`

### Phase 5: News Citation Pattern Variables ✅ **COMPLETED**
**Goal**: Compute response-level citation metrics (reuse from preference analysis)

**Scripts completed**:
- `scripts/compute_citation_patterns.py` ✅ (adapted from `preference_analysis/compute_response_signals.py`)
  - Reuse existing citation pattern computation logic
  - Key metrics to include:
    - Proportion of left/right/center-leaning sources
    - Proportion of high/low quality sources
  - Output: `data/intermediate/question_analysis/citation_patterns.parquet`

### Phase 6: Data Integration ✅ **COMPLETED**
**Goal**: Merge question features with citation patterns and model metadata

**Scripts completed**:
- `scripts/integrate_analysis_data.py` ✅
  - Join questions with responses, citations, and threads
  - Merge question embeddings, features, and topic data with citation patterns
  - Add model family and model-specific variables
  - Handle missing data and create analysis-ready dataset
  - Output: `data/intermediate/question_analysis/integrated_analysis_data.parquet`

### Phase 7: Clean and Code Features ✅ **COMPLETED**
**Goal**: Prepare integrated data for regression modeling

**Scripts completed**:
- `scripts/clean_features.py` ✅
  - Convert categorical variables to dummy variables (country, model family, intent, etc.)
  - Transform long-tail distributed variables (response/question length) using log transformation
  - Standardize embedding dimensions for consistent scaling
  - Handle missing values with appropriate imputation strategies
  - Validate final dataset for modeling readiness
  - Output: `data/intermediate/question_analysis/cleaned_features.parquet`

### Phase 8: Regression Analysis ✅ **COMPLETED**
**Goal**: Analyze relationships between question features and citation patterns

**Scripts completed**:
- `scripts/regression_analysis.py` ✅
  - **Dependent variables**: News citation patterns (% left-leaning, % high-quality, etc.)
  - **Independent variables**:
    - Question features (embeddings, topic probabilities, length, etc.)
    - Model variables (model family, model side)
    - Control variables (turn number, thread characteristics)
  - **Models to fit**:
    - Linear regression for continuous outcomes
  - **Analysis outputs**:
    - Coefficient tables with confidence intervals
    - Model diagnostics and fit statistics
  - Output: `data/output/question_analysis/regression_results.json`

### Phase 9: Visualization and Reporting ✅ **COMPLETED**
**Goal**: Generate comprehensive analysis report

**Scripts completed**:
- `scripts/generate_question_analysis_report.py` ✅
  - Summary statistics for question features
  - Correlation matrices between features and outcomes
  - Regression coefficient plots
  - Feature importance visualizations
  - Subgroup analyses (by model family, question type)
  - Output: `data/output/question_analysis/question_analysis_report.html`

### Phase 10: LaTeX Regression Table ✅ **COMPLETED**
**Goal**: Generate publication-ready regression tables

**Scripts completed**:
- `scripts/generate_latex_regression_table.py` ✅
  - Format regression coefficients for academic papers
  - Include confidence intervals and significance indicators
  - Output: `data/output/question_analysis/regression_coefficients_table.tex`

### Phase 11: LaTeX Topics Table ✅ **COMPLETED**
**Goal**: Generate publication-ready topic modeling tables

**Scripts completed**:
- `scripts/generate_latex_topics_table.py` ✅
  - Format topic modeling results for academic papers
  - Include topic keywords and distributions
  - Output: `data/output/question_analysis/topics_table.tex`

## Technical Implementation Details

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

### Implemented Snakemake Workflow ✅ **COMPLETED**
**11-Phase Pipeline**: Complete workflow with all scripts implemented and tested

```python
rule all:
    input:
        # Core analysis outputs
        "data/output/question_analysis/question_analysis_report.html",
        "data/output/question_analysis/regression_results.json",
        # Publication-ready LaTeX tables
        "data/output/question_analysis/regression_coefficients_table.tex",
        "data/output/question_analysis/topics_table.tex",
        # All intermediate datasets
        "data/intermediate/question_analysis/english_questions.parquet",
        "data/intermediate/question_analysis/question_embeddings.parquet",
        "data/intermediate/question_analysis/question_topics.parquet",
        "data/intermediate/question_analysis/cleaned_features.parquet"

# Phase 1: English question filtering
rule filter_english_questions: # ✅ IMPLEMENTED

# Phase 2: Question embeddings generation
rule generate_question_embeddings: # ✅ IMPLEMENTED

# Phase 3: Topic modeling with BERTopic
rule generate_question_topics: # ✅ IMPLEMENTED

# Phase 4: Question feature extraction
rule extract_question_features: # ✅ IMPLEMENTED

# Phase 5: Citation patterns computation
rule compute_citation_patterns: # ✅ IMPLEMENTED

# Phase 6: Data integration
rule integrate_analysis_data: # ✅ IMPLEMENTED

# Phase 7: Feature cleaning and preparation
rule clean_features: # ✅ IMPLEMENTED

# Phase 8: Regression analysis
rule regression_analysis: # ✅ IMPLEMENTED

# Phase 9: HTML report generation
rule generate_report: # ✅ IMPLEMENTED

# Phase 10: LaTeX regression table
rule generate_latex_table: # ✅ IMPLEMENTED

# Phase 11: LaTeX topics table
rule generate_latex_topics_table: # ✅ IMPLEMENTED
```

## Research Questions Addressed ✅ COMPLETED

1. **Political Questions**: Do political questions lead to more politically biased news citations?
2. **Question Complexity**: Do more complex questions result in higher-quality or more diverse news sources?
3. **Model Differences**: Do different model families respond differently to question characteristics?
4. **Topic Effects**: How do different question topics influence citation patterns?
5. **Semantic Clustering**: What semantic topics emerge from user questions and how do they relate to citation behavior?
6. **Embedding Analysis**: How do semantic embeddings predict citation pattern variations?
7. **Statistical Significance**: Which question features have statistically significant effects on citation bias and quality?

## Validation Strategy

1. **Language Detection Validation**: Manual review of sample questions for language accuracy
2. **Feature Quality Checks**: Correlation analysis between different feature types
3. **Citation Pattern Validation**: Consistency checks with existing preference analysis results
4. **Model Diagnostics**: Residual analysis, multicollinearity checks, cross-validation
5. **Robustness Tests**: Analysis across different subsets and alternative specifications

## Implementation Status: ✅ COMPLETED

### Production-Ready Pipeline Achievements
1. ✅ **Complete 11-phase question analysis pipeline**
2. ✅ **Topic modeling integration** with BERTopic for semantic analysis
3. ✅ **Semantic embeddings** using sentence transformers (all-MiniLM-L6-v2)
4. ✅ **Comprehensive regression analysis** with multiple dependent variables
5. ✅ **Publication-ready outputs** including LaTeX tables and HTML reports
6. ✅ **Cross-validation and model diagnostics** for robust statistical inference
7. ✅ **Configurable parameters** for filtering, embeddings, and analysis

### Ready for Question-Citation Pattern Analysis
The pipeline produces comprehensive analysis of:
- **Question characteristics effects** on news citation patterns
- **Topic modeling results** with semantic clustering of questions
- **Regression analysis** linking question features to citation bias and quality
- **Model comparison** across different AI systems and question types
- **Publication-ready tables** for academic papers

### Running the Complete Pipeline
```bash
# Run full question analysis pipeline
cd workflow/analysis/question_analysis
snakemake --cores 1

# Or target specific outputs
snakemake generate_question_topics --cores 1    # Topic modeling
snakemake regression_analysis --cores 1         # Statistical analysis
snakemake generate_latex_table --cores 1        # Publication tables
```

### Output Deliverables ✅ COMPLETED

1. **Analysis Dataset**: Clean, integrated dataset ready for modeling
2. **Topic Modeling Results**: BERTopic analysis with semantic clusters
3. **Regression Results**: Comprehensive statistical analysis results
4. **HTML Report**: Interactive report with visualizations and interpretations
5. **LaTeX Tables**: Regression and topic tables formatted for paper inclusion
6. **Feature Importance Rankings**: Ranked lists of most predictive question features
7. **Embedding Datasets**: Question embeddings for downstream analysis