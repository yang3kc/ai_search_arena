#!/usr/bin/env python3
"""
Generate a LaTeX table with model statistics for the research paper.

This script creates a table showing:
- Model name
- Number of responses
- Number of citations
- Number of news citations (count and percentage)
"""

import pandas as pd
import sys
import os
from pathlib import Path

def load_data(intermediate_dir):
    """Load the required data files."""
    print("Loading data...")
    
    # Load responses data
    responses_path = Path(intermediate_dir) / "responses.parquet"
    responses = pd.read_parquet(responses_path)
    print(f"Loaded {len(responses):,} responses")
    
    # Load enriched citations data
    citations_path = Path(intermediate_dir) / "citations_enriched.parquet"
    citations = pd.read_parquet(citations_path)
    print(f"Loaded {len(citations):,} citations")
    
    return responses, citations

def generate_model_stats(responses, citations):
    """Generate model statistics table."""
    print("Generating model statistics...")
    
    # Get model response counts
    model_responses = responses.groupby('model_name_llm').size().reset_index()
    model_responses.columns = ['model_name', 'num_responses']
    
    # Get citation counts per model
    # First, join citations with responses to get model info
    citations_with_model = citations.merge(
        responses[['response_id', 'model_name_llm']], 
        on='response_id', 
        how='left'
    )
    
    # Count total citations per model
    model_citations = citations_with_model.groupby('model_name_llm').size().reset_index()
    model_citations.columns = ['model_name', 'num_citations']
    
    # Count news citations per model
    news_citations = citations_with_model[
        citations_with_model['domain_classification'] == 'news'
    ].groupby('model_name_llm').size().reset_index()
    news_citations.columns = ['model_name', 'num_news_citations']
    
    # Merge all statistics
    stats = model_responses.merge(model_citations, on='model_name', how='left')
    stats = stats.merge(news_citations, on='model_name', how='left')
    
    # Fill NaN values with 0 for models with no citations
    stats['num_citations'] = stats['num_citations'].fillna(0).astype(int)
    stats['num_news_citations'] = stats['num_news_citations'].fillna(0).astype(int)
    
    # Calculate news citation percentage
    stats['news_citation_percentage'] = (
        stats['num_news_citations'] / stats['num_citations'] * 100
    ).round(1)
    
    # Handle division by zero (models with no citations)
    stats.loc[stats['num_citations'] == 0, 'news_citation_percentage'] = 0.0
    
    # Sort by number of responses (descending)
    stats = stats.sort_values('num_responses', ascending=False)
    
    return stats

def format_model_name(model_name):
    """Format model name for LaTeX display."""
    # Handle common model names
    name_mapping = {
        'sonar': 'Perplexity Sonar',
        'gpt-4o': 'GPT-4o',
        'gpt-4o-mini': 'GPT-4o-mini',
        'claude-3-5-sonnet': 'Claude-3.5-Sonnet',
        'claude-3-haiku': 'Claude-3-Haiku',
        'claude-3-opus': 'Claude-3-Opus',
        'gemini-1.5-pro': 'Gemini-1.5-Pro',
        'gemini-1.5-flash': 'Gemini-1.5-Flash',
        'llama-3.1-405b': 'Llama-3.1-405B',
        'llama-3.1-70b': 'Llama-3.1-70B',
        'llama-3.1-8b': 'Llama-3.1-8B',
    }
    
    return name_mapping.get(model_name, model_name.replace('_', '-'))

def generate_latex_table(stats):
    """Generate LaTeX table from statistics."""
    print("Generating LaTeX table...")
    
    # Start table
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Model Statistics: Response Counts and Citation Patterns}")
    latex.append("\\label{tab:model_stats}")
    latex.append("\\begin{tabular}{lrrrr}")
    latex.append("\\toprule")
    latex.append("Model & Responses & Citations & News Citations & News \\% \\\\")
    latex.append("\\midrule")
    
    # Add data rows
    for _, row in stats.iterrows():
        model_name = format_model_name(row['model_name'])
        latex.append(f"{model_name} & {row['num_responses']:,} & {row['num_citations']:,} & {row['num_news_citations']:,} & {row['news_citation_percentage']:.1f}\\% \\\\")
    
    # Add totals row
    total_responses = stats['num_responses'].sum()
    total_citations = stats['num_citations'].sum()
    total_news_citations = stats['num_news_citations'].sum()
    total_news_percentage = (total_news_citations / total_citations * 100) if total_citations > 0 else 0.0
    
    latex.append("\\midrule")
    latex.append(f"\\textbf{{Total}} & \\textbf{{{total_responses:,}}} & \\textbf{{{total_citations:,}}} & \\textbf{{{total_news_citations:,}}} & \\textbf{{{total_news_percentage:.1f}\\%}} \\\\")
    
    # End table
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return '\n'.join(latex)

def main():
    """Main function."""
    # Get parameters from snakemake or command line
    if 'snakemake' in globals():
        intermediate_dir = snakemake.params.intermediate_dir
        output_file = snakemake.output[0]
    else:
        if len(sys.argv) != 3:
            print("Usage: python generate_model_stats_table.py <intermediate_dir> <output_file>")
            sys.exit(1)
        intermediate_dir = sys.argv[1]
        output_file = sys.argv[2]
    
    # Load data
    responses, citations = load_data(intermediate_dir)
    
    # Generate statistics
    stats = generate_model_stats(responses, citations)
    
    # Print summary
    print("\nModel Statistics Summary:")
    print(stats.to_string(index=False))
    
    # Generate LaTeX table
    latex_table = generate_latex_table(stats)
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to: {output_file}")
    print("\nLaTeX table:")
    print(latex_table)

if __name__ == "__main__":
    main()