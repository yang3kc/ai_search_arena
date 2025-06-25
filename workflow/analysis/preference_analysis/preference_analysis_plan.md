# Preference Analysis with News Citation Patterns

## Overview
Create a modular preference analysis pipeline using Snakemake that focuses on competitions where both models cite news sources, analyzing how different citation patterns affect user preferences. This builds on the existing citation style analysis framework but adds news-specific metrics through a series of smaller, manageable scripts.

## Snakemake Workflow Structure

### Pipeline Phases
1. **Data Preparation**: Filter and prepare competition data with news citations
2. **Signal Computation**: Calculate response-level metrics
3. **Battle Data Creation**: Aggregate to thread-level comparisons
4. **Statistical Analysis**: Run Bradley-Terry models with bootstrap
5. **Reporting**: Generate results and visualizations

### Intermediate Files Strategy
Store intermediate outputs for reproducibility and debugging:
- `news_competitions.parquet` - Filtered threads with news citations
- `response_signals.parquet` - Response-level aggregated metrics
- `battle_data.parquet` - Thread-level comparison data
- `preference_results.json` - Statistical analysis results

## Implementation Plan

### 1. Snakemake Workflow Definition
- **File**: `workflow/analysis/preference_analysis/Snakefile`
- **Purpose**: Orchestrate the complete preference analysis pipeline
- **Features**:
  - Define dependencies between analysis steps
  - Manage intermediate file creation
  - Enable parallel execution where possible
  - Provide clear targets for each analysis phase

### 2. Data Preparation Script
- **File**: `workflow/analysis/preference_analysis/scripts/prepare_news_competitions.py`
- **Input**: `citations_enriched.parquet`, `threads.parquet`, `responses.parquet`
- **Output**: `news_competitions.parquet`, `news_competitions_responses.parquet`, `news_competitions_citations.parquet`
- **Purpose**: Filter to threads where at least one model cites news
- **Features**:
  - Identify news-citing responses
  - Filter to valid competitions (exactly 2 models per thread)
  - Remove ties and invalid outcomes
  - Export filtered dataset

### 3. Response Signal Computation Script
- **File**: `workflow/analysis/preference_analysis/scripts/compute_response_signals.py`
- **Input**: `news_competitions.parquet`, `news_competitions_citations.parquet`, `news_competitions_responses.parquet`
- **Output**: `news_competitions_with_response_signals.parquet`
- **Purpose**: Calculate competition-level metrics for each model response
- **Features**:
  - Average response length (character/word count)
  - Average citation count per response
  - Quality metrics (proportion low-quality, average quality)
  - Bias metrics (proportion right-leaning, average bias)
  - Export enriched competition data

### 4. Battle Data Creation Script
- **File**: `workflow/analysis/preference_analysis/scripts/create_battle_data.py`
- **Input**: `news_competitions.parquet`, `response_signals.parquet`
- **Output**: `battle_data.parquet`
- **Purpose**: Aggregate response signals to thread-level comparisons
- **Features**:
  - Create model A vs model B comparison rows
  - Calculate signal differences between competing responses
  - Include thread-level winner information
  - Export battle-ready dataset

### 5. Statistical Analysis Script
- **File**: `workflow/analysis/preference_analysis/scripts/analyze_preferences.py`
- **Input**: `battle_data.parquet`
- **Output**: `preference_results.json`, `preference_coefficients.csv`
- **Purpose**: Run Bradley-Terry models with bootstrap confidence intervals
- **Features**:
  - Multiple analysis configurations (volume, quality, bias effects)
  - Bootstrap resampling for confidence intervals
  - Statistical significance testing
  - Export detailed results

### 6. Reporting and Visualization Script
- **File**: `workflow/analysis/preference_analysis/scripts/generate_report.py`
- **Input**: `preference_results.json`, `battle_data.parquet`
- **Output**: `preference_analysis_report.html`, visualization PNGs
- **Purpose**: Create comprehensive analysis report
- **Features**:
  - Executive summary of findings
  - Effect size visualizations
  - Statistical significance plots
  - Interactive HTML report

### 7. Utility Functions Module
- **File**: `workflow/analysis/preference_analysis/scripts/preference_utils.py`
- **Purpose**: Shared functions across scripts
- **Features**:
  - Bradley-Terry model implementation
  - Bootstrap resampling utilities
  - Data validation functions
  - Common statistical operations

## Snakemake Workflow Configuration

### Target Rules
- `all`: Complete preference analysis pipeline
- `prepare_data`: Data preparation phase only
- `compute_signals`: Through signal computation
- `create_battles`: Through battle data creation
- `analyze`: Through statistical analysis
- `report`: Complete analysis with reporting

### File Organization
```
data/intermediate/preference_analysis/
├── news_competitions.parquet      # Phase 1 output
├── response_signals.parquet       # Phase 2 output
├── battle_data.parquet           # Phase 3 output
├── preference_results.json       # Phase 4 output
├── preference_coefficients.csv   # Phase 4 output
├── preference_analysis_report.html # Phase 5 output
└── visualizations/               # Phase 5 output
    ├── effect_sizes.png
    ├── confidence_intervals.png
    └── statistical_significance.png
```

## Expected Insights
- Do users prefer responses with more or fewer news citations?
- How does news source quality affect user preferences?
- Do users prefer politically balanced vs. biased news sources?
- What's the optimal balance between citation quantity and quality?
- How do different response lengths interact with citation patterns?

## Technical Approach
- **Modular Design**: Each script handles one specific task
- **Intermediate Storage**: All phases save intermediate results
- **Dependency Management**: Snakemake ensures proper execution order
- **Statistical Rigor**: Bootstrap confidence intervals for all estimates
- **Reproducibility**: Clear data lineage and version control
- **Scalability**: Scripts can handle large datasets efficiently

## Benefits of This Approach
- **Debugging**: Easy to identify and fix issues in specific phases
- **Iteration**: Can re-run individual steps during development
- **Transparency**: Clear data flow and intermediate inspection
- **Efficiency**: Snakemake only re-runs changed components
- **Collaboration**: Smaller scripts easier to review and modify