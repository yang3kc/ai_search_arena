# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data analysis project that explores web search results from the AI search arena data.
The focus is on analyzing information sources cited by AI chatbots to identify patterns in their citation behavior, including political leaning and credibility metrics of cited domains.

## Architecture

The project follows a data science workflow organized around these key components:

- **Data Layer**: Raw datasets in `data/raw_data/` including search arena conversations, domain political leaning scores, and domain credibility ratings
- **Analysis Layer**: Jupyter notebooks in `notebooks/` for exploratory data analysis
- **Workflow Layer**: Snakemake-based pipeline in `workflow/` for reproducible data processing
- **Output Layer**: Processed results in `data/intermediate/` and `data/output/`

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

## Development Workflow

The project emphasizes atomic, modular scripts managed by Snakemake:

1. **Data Cleaning**: Process raw data files for downstream analysis
2. **Data Exploration**: Exploratory analysis in Jupyter notebooks
3. **Data Analysis**: Formal analysis pipeline using cleaned data
4. **Data Visualization**: Generate visualizations to understand patterns

## Working with the Data

- The main search arena dataset (`search-arena-chat-24k.parquet`) is large and not committed to the repo - download it from https://huggingface.co/datasets/lmarena-ai/search-arena-24k before running analysis
- Web search citations are stored in the `web_search_trace` field of the metadata columns
- Political leaning scores are in the `leaning_score_users` column
- Domain credibility scores are in the `pc1` column

## Common Development Tasks

- Start exploratory analysis in `notebooks/explore.ipynb`
- All input/output files should be managed through Snakemake workflows
- Keep scripts focused on single data processing tasks
- Use Snakemake to manage dependencies between processing steps

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
