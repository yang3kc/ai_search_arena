# Citation Analysis Plan

## Overview
Create a comprehensive citation analysis pipeline to investigate how different AI models cite sources with varying political leanings, quality scores, and domain classifications. This analysis will reveal model-specific biases, source preferences, and citation patterns across different query types and contexts.

## Research Objectives

### **1. Domain Classification Patterns** (PRIMARY FOCUS)
- Which source types (news, academic, social media, etc.) do models cite most?
- Do models show specialization (academic sources for technical queries)?
- How do citation patterns vary by primary/secondary intent?
- Are there model-specific domain preferences?
- What is the overall distribution of citations across domain types?

### **2. Political Bias Analysis** (NEWS SOURCES ONLY)
- Do AI models show systematic bias toward left-leaning vs right-leaning news sources?
- How does political leaning vary by model type within news citations?
- Do winning models have different political citation patterns for news?
- What percentage of news citations have political leaning data?

### **3. Source Quality Preferences** (NEWS SOURCES ONLY)
- Do AI models prefer high-quality vs low-quality news sources?
- Is there a trade-off between source quality and political neutrality in news?
- How does news quality preference vary by model?
- Do quality patterns correlate with user satisfaction (winners) for news citations?


## Technical Implementation

### **Phase 1: Data Integration Pipeline**
- Join enriched citations with responses, questions, and threads
- Create model comparison datasets (A vs B)
- Generate citation position and frequency metrics

### **Phase 2: Domain Classification Analysis**
- Overall distribution analysis of citations across all domain types
- Model-specific domain citation profiles
- Domain specialization patterns by query intent
- Model A vs Model B domain preference differences

### **Phase 3: News Source Analysis**
- Filter citations to news sources only
- Political bias analysis within news citations
- Source quality analysis within news citations
- Model comparison for news citation patterns

### **Phase 4: Comparative Analysis**
- Model A vs Model B citation pattern differences
- Winner vs loser citation pattern analysis
- Query intent impact on citation behavior

## Data Sources
- **Primary**: `citations_enriched.parquet` (366K citations with all signals)
- **Secondary**: `responses.parquet`, `threads.parquet`, `questions.parquet`
- **Enrichment**: Model metadata, intents, winner information

## Deliverables
- **Snakemake Pipeline**: Automated analysis workflow
- **Interactive Notebooks**: Exploratory analysis and visualization
- **Statistical Reports**: Model comparison and bias detection results
- **Visualization Dashboard**: Citation pattern exploration tools
- **Research Report**: Comprehensive findings and recommendations

## Expected Insights
- Domain classification distribution patterns across AI models
- Model specialization in citing different source types
- Political bias patterns specifically within news sources
- News source quality preferences by model
- Model-specific citation behavior differences