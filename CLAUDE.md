# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive data analysis project that explores citation patterns in AI search systems using the AI Search Arena dataset.
The focus is on analyzing news sources cited by AI chatbots to identify patterns in their citation behavior, including political bias and source credibility metrics.

**Current Status**: ✅ **PUBLIC-RELEASE-READY** - Complete analysis ecosystem with comprehensive documentation, multiple analysis pipelines, and reproducible workflows.

## Architecture

The project follows a comprehensive data science workflow with multiple completed analysis pipelines:

- **Data Layer**: Raw datasets in `data/raw_data/` including search arena conversations, domain political leaning scores, and domain credibility ratings
- **Extraction Pipeline**: Complete Snakemake-based pipeline in `workflow/data_cleaning/` that normalizes arena data into relational tables
- **Analysis Pipelines**: Four specialized analysis workflows in `workflow/analysis/`:
  - `citation_analysis/` - Domain classification, political bias, and source quality analysis
  - `preference_analysis/` - Bradley-Terry modeling and user preference analysis
  - `question_analysis/` - Topic modeling and regression analysis of questions
- **Notebook Layer**: Jupyter notebooks in `notebooks/` for figure generation and exploratory analysis
- **Output Layer**: Publication-ready outputs in `data/output/` including LaTeX tables, figures, and reports

## Key Data Sources

The project works with three main datasets:

1. **Search Arena Data** (`search-arena-chat-24k.parquet`): Contains 24k+ conversations between users and AI search systems, including web search traces with cited URLs
2. **Domain Political Leaning** (`DomainDemo_political_leaning.csv.gz`): Political bias scores for domains
3. **Domain Credibility** (`lin_domain_ratings.csv.gz`): Credibility and reliability metrics for domains

## Tech Stack

- **Python 3.10+** as the primary language for data analysis
- **pandas** for data manipulation and analysis
- **matplotlib & seaborn** for data visualization
- **Snakemake** for workflow management and reproducible pipelines, use context7 to fetch documentation regarding snakemake
- **Jupyter notebooks** for exploratory analysis and figure generation
- **Machine Learning**: scikit-learn, sentence-transformers, BERTopic for topic modeling
- **Statistical Analysis**: scipy, statsmodels for regression and Bradley-Terry modeling

## Completed Data Extraction Pipeline

The project has a **complete, production-ready data extraction pipeline** in `workflow/data_cleaning/`:

### Extracted Tables (Ready for Analysis)
1. **threads.parquet** (24,069 rows) - Conversation metadata, winners, timestamps
2. **questions.parquet** (32,884 rows) - User queries from multi-turn conversations
3. **responses.parquet** (65,768 rows) - AI model responses with dual naming and complete metadata
4. **citations.parquet** (366,087 rows) - Web citations with dual domain extraction ⭐ **PRIMARY ANALYSIS TARGET**

### Pipeline Features
- **Comprehensive validation**: 35-test suite with 97.1% pass rate
- **Perfect referential integrity**: Zero foreign key violations across 366K+ records
- **Dual domain extraction**: Full domains (with subdomains) + base domains using tldextract
- **Professional data quality**: 100% URL validity, complete metadata extraction
- **Citation-focused design**: Optimized for bias and credibility research

### Running the Pipelines

**Individual Pipelines**:
```bash
# Data Cleaning Pipeline (required first)
cd workflow/data_cleaning
snakemake --cores 1

# Analysis Pipelines (after data cleaning)
cd workflow/analysis/citation_analysis && snakemake --cores 1
cd workflow/analysis/preference_analysis && snakemake --cores 1
cd workflow/analysis/question_analysis && snakemake --cores 1
cd workflow/analysis/youtube_analysis && snakemake --cores 1
```

## Working with the Data

- **Raw data**: Download `search-arena-chat-24k.parquet` from https://huggingface.co/datasets/lmarena-ai/search-arena-24k
- **Extracted data**: Use normalized tables in `data/intermediate/cleaned_arena_data/`
- **Primary analysis**: Focus on `citations.parquet` (366K records) for bias research
- **Join as needed**: Flexible relational structure allows joining tables for specific analysis

## Analysis Pipelines

The project includes four comprehensive analysis pipelines:

### 1. Citation Analysis Pipeline (`workflow/analysis/citation_analysis/`)
- **Domain classification**: Categorizes domains into news, academic, government, etc.
- **Political bias analysis**: Analyzes political leaning of news sources cited
- **Source quality analysis**: Evaluates credibility and reliability of cited sources
- **Model comparison**: Compares citation patterns across different AI models
- **News statistics**: Comprehensive reporting on news citation patterns
- **LaTeX table generation**: Publication-ready tables for academic papers

### 2. Preference Analysis Pipeline (`workflow/analysis/preference_analysis/`)
- **News competition analysis**: Focuses on threads where both responses cite news sources
- **Bradley-Terry modeling**: Statistical analysis of how citation patterns affect user preferences
- **Response signals**: Computes features like citation count, political balance, source quality
- **Effect size analysis**: Quantifies how different citation strategies impact user choice
- **Individual effects**: Model-specific preference patterns and rankings

### 3. Question Analysis Pipeline (`workflow/analysis/question_analysis/`)
- **Topic modeling**: BERTopic-based clustering of user questions into semantic topics
- **Language filtering**: Identification and processing of English questions
- **Citation patterns**: Analysis of how question types relate to citation behavior
- **Regression analysis**: Statistical modeling of factors affecting citation patterns
- **Feature engineering**: Extraction of question metadata and characteristics

## Generated Outputs & Documentation

### Publication-Ready Outputs
- **LaTeX Tables**: Ready for academic papers in `data/output/*/`
  - `top_news_sources_latex_table.tex` - Most cited news sources by model family
  - `overrepresented_sources_latex_table.tex` - Sources with highest log-odds ratios
  - `regression_coefficients_table.tex` - Statistical model results
  - `model_stats_table.tex` - Model performance and citation statistics

### Analysis Reports
- **Comprehensive Statistics**: `news_citation_statistics_report.md` with 200+ metrics
- **Data Summary Report**: `data_summary_report.md` with dataset overview
- **HTML Reports**: Interactive analysis reports with visualizations
- **Model Performance**: CSV summaries of model rankings and performance

### Visualization Outputs
- **PDF Figures**: Publication-quality visualizations for paper figures
- **PNG/SVG**: Web-compatible formats for presentations
- **Interactive Plots**: Plotly-based visualizations for exploration

### Notebooks for Paper Figures
The `notebooks/` directory contains specialized notebooks for generating paper figures:
- `paper_figure_*.ipynb` - Generate specific figures for academic publication
- `paper_numbers.ipynb` - Extract key statistics and numbers for the paper
- Clear execution order and dependencies documented in `notebooks/README.md`

### Documentation Structure
- **README.md**: Comprehensive project overview and quick start guide
- **CONTRIBUTING.md**: Detailed contributor guidelines and development setup
- **LICENSE**: MIT license for open-source distribution
- **requirements.txt**: Complete Python dependency specification
- **workflow/README.md**: Detailed pipeline documentation with architecture overview
- **notebooks/README.md**: Notebook organization and execution guidelines
- **CHANGELOG.md**: Version history and release notes

## Additional tools in the environment

### parquet-tools

You are working in an environment where the tool `parquet-tools` is installed.
The tool can be used to view and inspect the parquet files.

Use

```bash
parquet-tools show -n 10 <path_to_parquet_file>
```

to view the first 10 rows of the parquet file.
Always specify the top n rows to view, otherwise the tool will show all rows, which is not helpful.

Use

```bash
parquet-tools inspect <path_to_parquet_file>
```

to view the schema of the parquet file.

### ast-grep

You run in an environment where `ast-grep` is available.
Whenever a search requires syntax-aware or structural matching, default to

```bash
ast-grep --lang python -p '<pattern>'
```

or set `--lang` appropriately, and avoid falling back to text-only tools like `rg` or `grep` unless I explicitly request a plain-text search.

## Important Working Guidelines

### Pipeline Dependencies
- **Always run `data_cleaning` pipeline first** before any analysis pipelines
- Analysis pipelines can be run independently after data cleaning is complete

### Development Best Practices
- **Follow existing code patterns** and naming conventions
- **Update documentation** when adding new features or analysis
- **Test pipelines** on sample data before running on full dataset
- **Validate outputs** using existing validation scripts

### Data Handling
- **Never commit sensitive data** like API keys or credentials
- **Use environment variables** for configuration (see `config.yaml` files)
- **Validate data quality** using built-in validation scripts
- **Document data sources** and methodology clearly

### Publication Workflow
- **Generated LaTeX tables** are ready for academic papers
- **Figures** are generated programmatically via notebooks
- **Statistics** are extracted automatically for paper writing
- **Methodology** is documented in pipeline scripts and notebooks

### Project Status
- **Production-ready** with comprehensive validation
- **Public-release-ready** with complete documentation
- **Reproducible** workflows with clear dependencies
- **Extensible** architecture for additional analyses
