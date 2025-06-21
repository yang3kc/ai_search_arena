# Introduction

This project aims to explore the web search results from the AI search arena data.
We are focusing on the information sources cited by the AI chatbots and understand if there are any particular patterns.

# Data

Refer to the `data/raw_data/README.md` for the details of the raw data files.

# Tech stack

This project mainly use Python to analyze the data.
Specifically, we use:
- `snakemake` to manage the workflow
- `pandas` to analyze the data
- `matplotlib` to visualize the data

# Workflow

We try to keep each script as atomic as possible, then use `snakemake` to run them in the right order.
All the files related to the workflow are in the `workflow` folder.
All the input and output files should be managed by `snakemake`, and the scripts should only focus on the data processing.

We use Jupyter Notebook to explore the data.
All the notebooks are in the `notebooks` folder.

The workflow is roughly split into the following steps:
1. Data cleaning: cleaning the raw data files to facilitate the downstream analysis.
2. Data exploration: exploring the data to understand the patterns.
3. Data analysis: the formal data pipeline to analyze the cleaned data.
4. Data visualization: visualizing the results to understand the patterns.