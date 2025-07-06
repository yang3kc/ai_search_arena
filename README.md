# AI Search Arena: News Citation Analysis

**A comprehensive analysis of news sources cited by AI search systems**

This project analyzes citation patterns in AI search systems using data from the [AI Search Arena](https://huggingface.co/datasets/lmarena-ai/search-arena-24k), focusing on the news sources cited by different AI models and their potential biases in terms of political leaning and source credibility.

## 🎯 Research Goals

- **Citation Pattern Analysis**: Understand how different AI models cite news sources
- **Political Bias Detection**: Analyze political leaning in news sources cited by AI systems
- **Source Quality Assessment**: Evaluate the credibility and reliability of cited sources

## 📊 Dataset Overview

The project analyzes **24,000+ conversations** with AI search systems, containing:
- **366,087 news citations** across different models
- **65,768 AI responses** from 13 different models
- **Political bias scores** for 13,000+ news domains
- **Quality ratings** for 1,500+ news domains

### Key Models Analyzed
- **OpenAI**: GPT-4o variants with search capabilities
- **Perplexity**: Sonar family models (Pro, Reasoning variants)
- **Google**: Gemini 2.0/2.5 with grounding

## 🏗️ Architecture

```
📁 ai_search_arena/
├── 📊 data/
│   ├── raw_data/           # Original datasets (download required)
│   ├── intermediate/       # Processed data tables
│   └── output/            # Analysis results, figures, LaTeX tables
├── 📓 notebooks/          # Jupyter notebooks for exploration & figures
├── ⚙️ workflow/           # Snakemake pipelines
│   ├── data_cleaning/     # Data extraction & normalization
│   ├── analysis/          # Analysis pipelines
│   │   ├── citation_analysis/    # Domain classification & bias analysis
│   │   ├── preference_analysis/  # User preference modeling
│   │   └── question_analysis/    # Question topic analysis
└── 📄 Generated outputs/  # LaTeX tables, figures, reports
```

## 🚀 Quick Start

### Prerequisites

The project is mainly written in Python with the following dependencies:
- **Python 3.10+**
- **Snakemake** for workflow management
- **pandas, matplotlib** for data analysis
- **Additional tools**: see `requirements.txt`

The project is mostly written with [Claude Code](https://www.anthropic.com/claude-code).
Please refer to `CLAUDE.md` for more details.

Since the project mainly use Parquet files, it's advised to install the [parquet-tools](https://pypi.org/project/parquet-tools/) CLI tool to allow Claude Code to check the output data.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yang3kc/ai_search_arena.git
   cd ai_search_arena
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required datasets**
   Please refer to `data/README.md` for more details.

### Basic Usage

1. **Run data extraction pipeline**
   ```bash
   cd workflow/data_cleaning
   snakemake --cores 1
   ```

2. **Run analysis pipelines**
   ```bash
   # Citation analysis (political bias, source quality)
   cd workflow/analysis/citation_analysis
   snakemake --cores 1

   # User preference analysis
   cd workflow/analysis/preference_analysis
   snakemake --cores 1

   # Question analysis
   cd workflow/analysis/question_analysis
   snakemake --cores 1
   ```

3. **Generate paper figures**
   Please refer to `notebooks/` for more details.

## 📋 Workflow Overview

### 1. Data Cleaning Pipeline (`workflow/data_cleaning/`)
- **Input**: Raw conversation data (24k conversations)
- **Output**: Normalized relational tables (threads, questions, responses, citations)
- **Features**: 35-test validation suite, referential integrity checks

### 2. Citation Analysis (`workflow/analysis/citation_analysis/`)
- Domain classification (news, academic, government, etc.)
- Political bias analysis using domain leaning scores
- Source quality evaluation using credibility ratings
- Model-specific citation pattern analysis

### 3. Preference Analysis (`workflow/analysis/preference_analysis/`)
- Bradley-Terry modeling of user preferences
- Citation style effect analysis
- News competition analysis (head-to-head comparisons)

### 4. Question Analysis (`workflow/analysis/question_analysis/`)
- Topic modeling of user questions
- Citation pattern analysis by question type
- Regression analysis of factors affecting citations

## 📖 Citation

If you use this work in your research, please cite:

**TO be added**