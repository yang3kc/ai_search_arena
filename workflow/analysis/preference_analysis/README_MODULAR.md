# Modular Preference Analysis Workflow

This document explains how to use the new modular Snakemake workflow for preference analysis with dedicated steps for each analysis component.

## Overview

The preference analysis has been split into separate Snakemake rules for maximum flexibility:

1. **Bradley-Terry Ratings** (`compute_bt_ratings`) - Basic model performance rankings
2. **Individual Feature Effects** (`analyze_individual_effects`) - Single-feature impact analysis  
3. **Citation Style Effects** (`analyze_citation_style_effects`) - Multi-feature model analysis

## Quick Start

### Run Individual Analysis Components

```bash
# Bradley-Terry ratings only
snakemake bt_ratings_only --cores 1

# Individual feature effects only  
snakemake individual_effects_only --cores 1

# Citation style effects only
snakemake citation_style_effects_only --cores 1
```

### Run All Analyses

```bash
# All analysis components (without report)
snakemake all_analyses --cores 1

# Everything including report and visualizations
snakemake all --cores 1
```

## Advanced Usage

### Custom Citation Style Model Sets

The `analyze_citation_style_custom` rule allows you to run different predefined model sets:

```bash
# Basic models only
snakemake citation_style_effects_basic_results.json --cores 1

# Political bias models only
snakemake citation_style_effects_political_results.json --cores 1

# Source quality models only
snakemake citation_style_effects_quality_results.json --cores 1

# Extended models with news-specific features
snakemake citation_style_effects_extended_results.json --cores 1

# Core models (basic + political + quality)
snakemake citation_style_effects_core_results.json --cores 1

# Advanced models
snakemake citation_style_effects_advanced_results.json --cores 1

# All available models
snakemake citation_style_effects_all_results.json --cores 1
```

### Available Model Sets

- **`basic`**: Basic response characteristics (word count, citation count)
- **`political`**: Political bias and source type analysis
- **`quality`**: Source quality and credibility analysis
- **`extended`**: Extended models with news-specific features
- **`core`**: Essential models (basic + political + quality)
- **`advanced`**: Advanced model combinations
- **`all`**: All available model specifications

## Output Files

Each analysis component produces dedicated output files:

### Bradley-Terry Ratings
- `bt_ratings_results.json` - Model ratings and metadata
- `bt_ratings_coefficients.csv` - Model ratings in CSV format

### Individual Feature Effects
- `individual_effects_results.json` - Individual feature analysis results
- `individual_effects_coefficients.csv` - Feature effects in CSV format

### Citation Style Effects
- `citation_style_effects_results.json` - Multi-model analysis results
- `citation_style_effects_coefficients.csv` - Model comparison in CSV format
- `citation_style_effects_{model_set}_results.json` - Custom model set results
- `citation_style_effects_{model_set}_coefficients.csv` - Custom model set CSV

### Reports and Visualizations
- `preference_analysis_report.html` - Comprehensive HTML report
- `visualizations/effect_sizes.png` - Effect size plots
- `visualizations/confidence_intervals.png` - Confidence interval plots
- `visualizations/statistical_significance.png` - Significance plots
- `visualizations/model_comparison.png` - Model comparison plots

## Configuration

Configure analysis parameters in `config/config.yaml`:

```yaml
statistical_analysis:
  bootstrap_samples: 1000  # Number of bootstrap samples
  random_seed: 42         # Random seed for reproducibility
```

## Workflow Dependencies

```
battle_data.parquet
├── compute_bt_ratings → bt_ratings_results.json
├── analyze_individual_effects → individual_effects_results.json
├── analyze_citation_style_effects → citation_style_effects_results.json
└── analyze_citation_style_custom → citation_style_effects_{model_set}_results.json

All results → generate_report → preference_analysis_report.html
```

## Performance Notes

- **Individual Effects**: Uses fewer bootstrap samples (config value ÷ 2) for faster execution
- **Citation Style Effects**: Full bootstrap samples for comprehensive analysis
- **Bradley-Terry Ratings**: Fastest, no bootstrap sampling required

## Examples

### Development Workflow
```bash
# Quick model rankings
snakemake bt_ratings_only --cores 1

# Test specific feature effects
snakemake individual_effects_only --cores 1

# Experiment with different model specifications
snakemake citation_style_effects_basic_results.json --cores 1
snakemake citation_style_effects_political_results.json --cores 1

# Full analysis when ready
snakemake all --cores 1
```

### Research Workflow
```bash
# Compare different model specifications
snakemake citation_style_effects_core_results.json --cores 1
snakemake citation_style_effects_advanced_results.json --cores 1
snakemake citation_style_effects_all_results.json --cores 1

# Generate comprehensive report
snakemake report_only --cores 1
```

This modular approach provides maximum flexibility for experimentation while maintaining the ability to run comprehensive analyses when needed.