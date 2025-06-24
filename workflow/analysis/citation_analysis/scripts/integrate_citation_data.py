#!/usr/bin/env python3
"""
Integrate citation data with responses, threads, and questions.

This script joins the enriched citations dataset with response metadata,
thread information, and question context to create a comprehensive dataset
for citation analysis.
"""

import pandas as pd
import sys
from pathlib import Path


def load_datasets(citations_path, responses_path, threads_path, questions_path):
    """Load all required datasets."""
    print("Loading datasets...")
    
    # Load citations with all enrichment signals
    print(f"Loading citations from {citations_path}")
    citations = pd.read_parquet(citations_path)
    print(f"Loaded {len(citations):,} citations")
    
    # Load responses with model information
    print(f"Loading responses from {responses_path}")
    responses = pd.read_parquet(responses_path)
    print(f"Loaded {len(responses):,} responses")
    
    # Load threads with winner information and intents
    print(f"Loading threads from {threads_path}")
    threads = pd.read_parquet(threads_path)
    print(f"Loaded {len(threads):,} threads")
    
    # Load questions for query context
    print(f"Loading questions from {questions_path}")
    questions = pd.read_parquet(questions_path)
    print(f"Loaded {len(questions):,} questions")
    
    return citations, responses, threads, questions


def integrate_citation_data(citations, responses, threads, questions):
    """Integrate all datasets into a comprehensive citation analysis dataset."""
    print("\n=== INTEGRATING CITATION DATA ===")
    
    # Start with citations as the base
    integrated = citations.copy()
    print(f"Starting with {len(integrated):,} citations")
    
    # Step 1: Join with responses to get model information
    print("\nStep 1: Joining with responses...")
    response_cols = [
        'response_id', 'question_id', 'thread_id', 'turn_number',
        'model_name_llm', 'model_name_raw', 'model_side',
        'citation_format', 'primary_intent', 'secondary_intent'
    ]
    
    # Select only columns that exist in responses
    available_response_cols = [col for col in response_cols if col in responses.columns]
    if 'primary_intent' not in responses.columns:
        print("  Warning: primary_intent not found in responses, will get from threads")
    if 'secondary_intent' not in responses.columns:
        print("  Warning: secondary_intent not found in responses, will get from threads")
    
    integrated = integrated.merge(
        responses[available_response_cols],
        on='response_id',
        how='left'
    )
    print(f"  After joining with responses: {len(integrated):,} citations")
    
    # Step 2: Join with threads to get winner information and intents
    print("\nStep 2: Joining with threads...")
    thread_cols = [
        'thread_id', 'winner', 'primary_intent', 'secondary_intent',
        'total_turns', 'timestamp'
    ]
    
    # Select only columns that exist in threads and aren't already present
    available_thread_cols = [col for col in thread_cols if col in threads.columns]
    
    # If intents weren't in responses, get them from threads
    if 'primary_intent' not in integrated.columns:
        thread_cols_to_merge = available_thread_cols
    else:
        # Remove intent columns to avoid duplication
        thread_cols_to_merge = [col for col in available_thread_cols 
                               if col not in ['primary_intent', 'secondary_intent']]
    
    integrated = integrated.merge(
        threads[thread_cols_to_merge],
        on='thread_id',
        how='left'
    )
    print(f"  After joining with threads: {len(integrated):,} citations")
    
    # Step 3: Join with questions for query context (optional, for future analysis)
    print("\nStep 3: Joining with questions...")
    question_cols = ['question_id', 'question_text']
    
    # Only join if question_text exists and we have question_id
    if 'question_text' in questions.columns and 'question_id' in integrated.columns:
        integrated = integrated.merge(
            questions[question_cols],
            on='question_id',
            how='left'
        )
        print(f"  After joining with questions: {len(integrated):,} citations")
    else:
        print("  Skipping question text join (question_id or question_text not available)")
    
    return integrated


def add_derived_features(integrated):
    """Add derived features for analysis."""
    print("\n=== ADDING DERIVED FEATURES ===")
    
    # Model win status
    if 'winner' in integrated.columns and 'model_side' in integrated.columns:
        print("Adding model win status...")
        integrated['model_won'] = (
            (integrated['winner'] == 'model_a') & (integrated['model_side'] == 'a') |
            (integrated['winner'] == 'model_b') & (integrated['model_side'] == 'b')
        )
        integrated['model_lost'] = (
            (integrated['winner'] == 'model_a') & (integrated['model_side'] == 'b') |
            (integrated['winner'] == 'model_b') & (integrated['model_side'] == 'a')
        )
        integrated['model_tied'] = integrated['winner'].isin(['tie', 'tie (bothbad)', ''])
        
        print(f"  Model wins: {integrated['model_won'].sum():,}")
        print(f"  Model losses: {integrated['model_lost'].sum():,}")
        print(f"  Model ties: {integrated['model_tied'].sum():,}")
    
    # Citation position categories
    if 'citation_number' in integrated.columns:
        print("Adding citation position categories...")
        integrated['citation_position'] = pd.cut(
            integrated['citation_number'],
            bins=[0, 1, 3, 5, float('inf')],
            labels=['first', 'early', 'middle', 'late'],
            include_lowest=True
        )
        
        print("  Citation position distribution:")
        for pos, count in integrated['citation_position'].value_counts().items():
            print(f"    {pos}: {count:,}")
    
    # Model family (simplified model names)
    if 'model_name_llm' in integrated.columns:
        print("Adding model family classification...")
        
        def get_model_family(model_name):
            if pd.isna(model_name):
                return 'unknown'
            model_name = str(model_name).lower()
            if 'gpt' in model_name:
                return 'gpt'
            elif 'gemini' in model_name:
                return 'gemini'
            elif 'sonar' in model_name:
                return 'sonar'
            else:
                return 'other'
        
        integrated['model_family'] = integrated['model_name_llm'].apply(get_model_family)
        
        print("  Model family distribution:")
        for family, count in integrated['model_family'].value_counts().items():
            print(f"    {family}: {count:,}")
    
    return integrated


def generate_integration_summary(integrated):
    """Generate summary statistics for the integrated dataset."""
    print("\n=== INTEGRATION SUMMARY ===")
    
    total_citations = len(integrated)
    unique_responses = integrated['response_id'].nunique()
    unique_threads = integrated['thread_id'].nunique()
    unique_domains = integrated['domain'].nunique()
    
    print(f"Total citations: {total_citations:,}")
    print(f"Unique responses: {unique_responses:,}")
    print(f"Unique threads: {unique_threads:,}")
    print(f"Unique domains: {unique_domains:,}")
    
    # Data completeness
    print(f"\nData completeness:")
    for col in ['model_name_llm', 'winner', 'primary_intent', 'domain_classification',
                'political_leaning', 'domain_quality']:
        if col in integrated.columns:
            completeness = (1 - integrated[col].isna().mean()) * 100
            print(f"  {col}: {completeness:.1f}%")
    
    # Domain classification distribution
    if 'domain_classification' in integrated.columns:
        print(f"\nDomain classification distribution:")
        domain_dist = integrated['domain_classification'].value_counts()
        for domain_type, count in domain_dist.items():
            pct = count / total_citations * 100
            print(f"  {domain_type}: {count:,} ({pct:.1f}%)")
    
    # Model distribution
    if 'model_name_llm' in integrated.columns:
        print(f"\nTop 5 models by citation count:")
        model_dist = integrated['model_name_llm'].value_counts().head(5)
        for model, count in model_dist.items():
            pct = count / total_citations * 100
            print(f"  {model}: {count:,} ({pct:.1f}%)")
    
    print(f"\nFinal dataset columns: {list(integrated.columns)}")
    print(f"Dataset shape: {integrated.shape}")


def main():
    """Main function for data integration."""
    # Get paths from Snakemake
    citations_path = snakemake.input.citations
    responses_path = snakemake.input.responses
    threads_path = snakemake.input.threads
    questions_path = snakemake.input.questions
    output_path = snakemake.output[0]
    
    # Load all datasets
    citations, responses, threads, questions = load_datasets(
        citations_path, responses_path, threads_path, questions_path
    )
    
    # Integrate the data
    integrated = integrate_citation_data(citations, responses, threads, questions)
    
    # Add derived features
    integrated = add_derived_features(integrated)
    
    # Generate summary
    generate_integration_summary(integrated)
    
    # Save integrated dataset
    print(f"\nSaving integrated citations to {output_path}")
    integrated.to_parquet(output_path, index=False)
    
    print(f"\n✅ Data integration completed successfully!")
    print(f"Output: {len(integrated):,} citations with {len(integrated.columns)} columns")


if __name__ == "__main__":
    main()