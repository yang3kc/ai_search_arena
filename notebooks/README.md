# Notebooks Directory

This directory contains Jupyter notebooks for exploratory data analysis, figure generation, and research insights. The notebooks are organized into different categories based on their purpose.

## 📊 Paper Figure Generation Notebooks

These notebooks generate the figures and analyses used in the research paper:

### 📈 Main Analysis Figures
- **`paper_figure_classification.ipynb`** - Citation classification analysis
  - Domain type distribution across models
  - Citation category breakdowns
  - Model comparison by citation types

- **`paper_figure_model_news_freq.ipynb`** - News citation frequency analysis
  - News citation rates by model
  - Model family comparisons
  - Statistical significance testing

- **`paper_figure_political_leaning.ipynb`** - Political bias analysis
  - Political leaning distribution in news citations
  - Model-specific bias patterns
  - Temporal bias trends

- **`paper_figure_quality.ipynb`** - Source quality analysis
  - Quality score distributions
  - Model differences in source credibility
  - Quality vs. political leaning correlations

### 📊 Advanced Analyses
- **`paper_figure_news_concentration.ipynb`** - News source concentration
  - Lorenz curves for citation inequality
  - Gini coefficients by model family
  - Top domain analysis

- **`paper_figure_news_domain_similarity.ipynb`** - Domain similarity analysis
  - Jaccard similarity between models
  - Clustering analysis of citation patterns
  - Network visualization of domain relationships

- **`paper_figure_regressions.ipynb`** - Statistical modeling
  - Regression analysis of citation patterns
  - Effect size estimation
  - Model coefficient visualizations

- **`paper_figure_user_preference.ipynb`** - User preference analysis
  - Bradley-Terry model results
  - Citation effects on user choices
  - Preference pattern visualizations

### 📋 Summary & Statistics
- **`paper_numbers.ipynb`** - Key statistics for paper
  - Summary statistics tables
  - Key findings compilation
  - Publication-ready numbers

## 🔍 Exploratory Analysis Notebooks

### Data Exploration
- **`explore.ipynb`** - Initial data exploration
  - Dataset overview and structure
  - Basic statistics and distributions
  - Data quality assessment

- **`check_cleaned_data.ipynb`** - Data validation
  - Verification of cleaned datasets
  - Cross-table validation
  - Data integrity checks

- **`domain_classification.ipynb`** - Domain classification development
  - Manual domain categorization
  - Classification rule development
  - Validation of domain types

### Specialized Analyses
- **`model_domain_logodds.ipynb`** - Model preference analysis
  - Log-odds calculations for domain preferences
  - Statistical significance testing
  - Model comparison visualizations

- **`preference_analysis.ipynb`** - User preference modeling
  - Bradley-Terry model development
  - Citation effect analysis
  - Preference pattern exploration

- **`questions_explore.ipynb`** - Question analysis
  - User question categorization
  - Topic modeling exploration
  - Question-citation relationships

- **`topic_modeling.ipynb`** - Advanced topic modeling
  - BERTopic implementation
  - Topic evolution analysis
  - Question clustering

### Data Requirements
- All Snakemake pipelines must be completed before running paper figure notebooks
- Raw data must be downloaded (see main README.md)
- Intermediate data tables must be generated