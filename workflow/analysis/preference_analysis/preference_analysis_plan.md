# Preference Analysis with News Citation Patterns

## Overview
**Status**: ✅ **PRODUCTION-READY** - Complete Bradley-Terry preference analysis pipeline with comprehensive news citation pattern analysis.

Modular preference analysis pipeline using Snakemake that focuses on competitions where both models cite news sources, analyzing how different citation patterns affect user preferences. This builds on the existing citation style analysis framework but adds news-specific metrics through a series of smaller, manageable scripts.

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

### 5. Statistical Analysis Scripts (3 separate analyses)

#### 5a. Bradley-Terry Ratings
- **File**: `workflow/analysis/preference_analysis/scripts/bt_ratings.py`
- **Input**: `battle_data.parquet`
- **Output**: `bt_ratings_results.json`, `bt_ratings_coefficients.csv`
- **Purpose**: Compute basic Bradley-Terry ratings with bootstrap confidence intervals
- **Features**:
  - Anchor model for consistent rating scale
  - Bootstrap resampling for confidence intervals
  - Model ranking and rating computation

#### 5b. Individual Effects Analysis
- **File**: `workflow/analysis/preference_analysis/scripts/individual_effects.py`
- **Input**: `battle_data.parquet`
- **Output**: `individual_effects_results.json`, `individual_effects_coefficients.csv`
- **Purpose**: Analyze individual feature effects on user preferences
- **Features**:
  - Response length effects
  - Citation count effects
  - Quality and bias metrics effects
  - Statistical significance testing

#### 5c. Citation Style Effects
- **File**: `workflow/analysis/preference_analysis/scripts/citation_style_effects.py`
- **Input**: `battle_data.parquet`
- **Output**: `citation_style_effects_results.json`, `citation_style_effects_coefficients.csv`
- **Purpose**: Analyze citation style patterns with flexible models
- **Features**:
  - Contextual Bradley-Terry modeling
  - Citation pattern analysis
  - Style control effects

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
- **File**: `workflow/analysis/preference_analysis/scripts/bt_utils.py`
- **Purpose**: Shared Bradley-Terry functions across scripts
- **Features**:
  - Bradley-Terry model implementation
  - Contextual Bradley-Terry models
  - Bootstrap resampling utilities
  - Data validation functions
  - Result handling and export utilities

## Snakemake Workflow Configuration

### Implemented Rules
- `all`: Complete preference analysis pipeline with all outputs
- `prepare_data`: Data preparation phase only
- `compute_response_signals`: Response signal computation
- `create_battles`: Battle data creation
- `compute_bt_ratings`: Bradley-Terry ratings analysis
- `analyze_individual_effects`: Individual feature effects analysis
- `analyze_citation_style_effects`: Citation style effects analysis
- `generate_report`: Comprehensive HTML report with visualizations

### File Organization
```
data/intermediate/preference_analysis/
├── news_competitions.parquet                    # Phase 1 output
├── news_competitions_responses.parquet          # Phase 1 output
├── news_competitions_citations.parquet          # Phase 1 output
├── news_competitions_response_signals.parquet  # Phase 2 output
├── battle_data.parquet                         # Phase 3 output
├── bt_ratings_results.json                     # Phase 4a output
├── bt_ratings_coefficients.csv                 # Phase 4a output
├── individual_effects_results.json             # Phase 4b output
├── individual_effects_coefficients.csv         # Phase 4b output
├── citation_style_effects_results.json         # Phase 4c output
├── citation_style_effects_coefficients.csv     # Phase 4c output
├── preference_analysis_report.html             # Phase 5 output
└── visualizations/                             # Phase 5 output
    ├── individual_effects.png
    ├── citation_style_effects.png
    └── model_comparison.png
```

## Expected Insights
- Do users prefer responses with more or fewer news citations?
- How does news source quality affect user preferences?
- Do users prefer politically balanced vs. biased news sources?
- What's the optimal balance between citation quantity and quality?
- How do different response lengths interact with citation patterns?

## Technical Approach
- **Modular Design**: Each script handles one specific task with clear responsibilities
- **Intermediate Storage**: All phases save intermediate results for debugging and inspection
- **Dependency Management**: Snakemake ensures proper execution order with clear dependencies
- **Statistical Rigor**: Bootstrap confidence intervals for all estimates (configurable sample sizes)
- **Reproducibility**: Clear data lineage, version control, and configurable random seeds
- **Scalability**: Scripts optimized for large datasets with memory-efficient processing
- **Multiple Analysis Types**: Separate scripts for different statistical approaches
- **Comprehensive Reporting**: HTML report with interactive visualizations

## Implementation Status: ✅ COMPLETED

### Production-Ready Pipeline Achievements
1. ✅ **Complete 5-phase preference analysis pipeline**
2. ✅ **Three separate statistical analysis approaches** (ratings, individual effects, style effects)
3. ✅ **Comprehensive utility module** with Bradley-Terry implementations
4. ✅ **Configurable parameters** for statistical analysis and filtering
5. ✅ **HTML reporting** with interactive visualizations
6. ✅ **Bootstrap confidence intervals** for robust statistical inference
7. ✅ **Snakemake workflow integration** for reproducible execution

### Ready for News Citation Preference Analysis
The pipeline produces comprehensive analysis of:
- **User preference patterns** based on news citation behavior
- **Bradley-Terry model rankings** with confidence intervals
- **Individual feature effects** (response length, citation count, quality, bias)
- **Citation style effects** using contextual models
- **Model comparison** across different AI systems

### Running the Complete Pipeline
```bash
# Run full preference analysis pipeline
cd workflow/analysis/preference_analysis
snakemake --cores 1

# Or target specific analyses
snakemake compute_bt_ratings --cores 1      # Basic Bradley-Terry ratings
snakemake analyze_individual_effects --cores 1  # Feature effect analysis
snakemake generate_report --cores 1         # Comprehensive report
```

### Benefits Realized
- **Debugging**: Easy to identify and fix issues in specific phases
- **Iteration**: Can re-run individual steps during development
- **Transparency**: Clear data flow and intermediate inspection
- **Efficiency**: Snakemake only re-runs changed components
- **Collaboration**: Smaller scripts easier to review and modify
- **Statistical Rigor**: Multiple approaches for robust analysis
- **Comprehensive Output**: HTML reports with publication-ready visualizations