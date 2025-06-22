# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data analysis project that explores web search results from the AI search arena data.
The focus is on analyzing information sources cited by AI chatbots to identify patterns in their citation behavior, including political leaning and credibility metrics of cited domains.

**Current Status**: ✅ **PRODUCTION-READY** - Complete data extraction pipeline implemented with comprehensive validation.

## Architecture

The project follows a data science workflow with a completed extraction pipeline:

- **Data Layer**: Raw datasets in `data/raw_data/` including search arena conversations, domain political leaning scores, and domain credibility ratings
- **Extraction Pipeline**: Complete Snakemake-based pipeline in `workflow/data_cleaning/` that normalizes arena data into relational tables
- **Analysis Layer**: Jupyter notebooks in `notebooks/` for exploratory data analysis
- **Output Layer**: Normalized tables in `data/intermediate/cleaned_arena_data/` ready for analysis

## Key Data Sources

The project works with three main datasets:

1. **Search Arena Data** (`search-arena-chat-24k.parquet`): Contains 24k+ conversations between users and AI search systems, including web search traces with cited URLs
2. **Domain Political Leaning** (`DomainDemo_political_leaning.csv.gz`): Political bias scores for domains
3. **Domain Credibility** (`lin_domain_ratings.csv.gz`): Credibility and reliability metrics for domains

## Tech Stack

- **Python** as the primary language for data analysis
- **pandas** for data manipulation and analysis
- **matplotlib** for data visualization
- **Snakemake** for workflow management and reproducible pipelines, use context7 to fetch documentation regarding snakemake
- **Jupyter notebooks** for exploratory analysis

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

### Running the Pipeline
```bash
cd workflow/data_cleaning
snakemake --cores 1  # Runs complete extraction + validation
```

## Working with the Data

- **Raw data**: Download `search-arena-chat-24k.parquet` from https://huggingface.co/datasets/lmarena-ai/search-arena-24k
- **Extracted data**: Use normalized tables in `data/intermediate/cleaned_arena_data/`
- **Primary analysis**: Focus on `citations.parquet` (366K records) for bias research
- **Join as needed**: Flexible relational structure allows joining tables for specific analysis

## Analysis Strategy

The pipeline is designed for **citation-focused analysis**:
- **Citation bias analysis**: Join citations with domain political leaning data
- **Credibility assessment**: Join citations with domain credibility ratings
- **Model comparison**: Join with responses to compare citation patterns by model
- **Topic analysis**: Join with questions to understand citation patterns by query type

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
