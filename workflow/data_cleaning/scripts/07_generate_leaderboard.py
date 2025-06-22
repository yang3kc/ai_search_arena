"""
Generate leaderboard from cleaned arena data.

This script transforms the normalized data into leaderboard format and 
computes Bradley-Terry ratings with various controls.
"""

import pandas as pd
import numpy as np
import sys
import os
import json
from pathlib import Path

# Add leaderboard modules to path
script_dir = Path(__file__).parent.parent.parent / "leaderboard_replication"
sys.path.append(str(script_dir))

from feature_engineering import create_leaderboard_format
from leaderboard_core import run_leaderboard


def load_cleaned_data(data_dir):
    """Load all cleaned data tables."""
    
    print("Loading cleaned data...")
    
    threads_df = pd.read_parquet(data_dir / "threads.parquet")
    responses_df = pd.read_parquet(data_dir / "responses.parquet") 
    citations_df = pd.read_parquet(data_dir / "citations.parquet")
    
    print(f"Loaded {len(threads_df)} threads, {len(responses_df)} responses, {len(citations_df)} citations")
    
    return threads_df, responses_df, citations_df


def generate_leaderboard_variants(battle_data, output_dir, config):
    """Generate multiple leaderboard variants with different controls."""
    
    anchor_model = config.get('anchor_model', 'api-gpt-4o-search')
    anchor_rating = config.get('anchor_rating', 1000)
    num_bootstrap = config.get('num_bootstrap_samples', 100)
    
    # Find actual anchor model in data
    available_models = set(battle_data['model_a'].unique()) | set(battle_data['model_b'].unique())
    if anchor_model not in available_models:
        # Use most common model as anchor
        anchor_model = battle_data['model_a'].value_counts().index[0]
        print(f"Anchor model not found, using {anchor_model}")
    
    results = {}
    
    # 1. Basic leaderboard (no controls)
    print("\nGenerating basic leaderboard...")
    try:
        leaderboard_basic, bootstrap_basic, _ = run_leaderboard(
            battle_data,
            anchor_model=anchor_model,
            anchor_rating=anchor_rating,
            num_bootstrap_samples=num_bootstrap
        )
        
        results['basic'] = {
            'leaderboard': leaderboard_basic,
            'bootstrap': bootstrap_basic
        }
        
        # Save basic leaderboard
        leaderboard_basic.to_csv(output_dir / "leaderboard_basic.csv", index=True)
        bootstrap_basic.to_csv(output_dir / "bootstrap_basic.csv", index=True)
        
    except Exception as e:
        print(f"Error generating basic leaderboard: {e}")
        results['basic'] = None
    
    # 2. Citation style control
    print("\nGenerating citation style controlled leaderboard...")
    try:
        style_elements = ["standardized_citations_a", "standardized_citations_b"]
        
        leaderboard_style, bootstrap_style, style_coef = run_leaderboard(
            battle_data,
            anchor_model=anchor_model,
            anchor_rating=anchor_rating,
            style_elements=style_elements,
            num_bootstrap_samples=num_bootstrap
        )
        
        results['citation_style'] = {
            'leaderboard': leaderboard_style,
            'bootstrap': bootstrap_style,
            'coefficients': style_coef
        }
        
        # Save style-controlled leaderboard
        leaderboard_style.to_csv(output_dir / "leaderboard_citation_style.csv", index=True)
        bootstrap_style.to_csv(output_dir / "bootstrap_citation_style.csv", index=True)
        
        if style_coef is not None:
            np.save(output_dir / "style_coefficients_citation.npy", style_coef)
            
    except Exception as e:
        print(f"Error generating style-controlled leaderboard: {e}")
        results['citation_style'] = None
    
    # 3. Response length control
    print("\nGenerating response length controlled leaderboard...")
    try:
        length_elements = ["response_length_a", "response_length_b"]
        
        leaderboard_length, bootstrap_length, length_coef = run_leaderboard(
            battle_data,
            anchor_model=anchor_model,
            anchor_rating=anchor_rating,
            style_elements=length_elements,
            num_bootstrap_samples=num_bootstrap
        )
        
        results['response_length'] = {
            'leaderboard': leaderboard_length,
            'bootstrap': bootstrap_length,
            'coefficients': length_coef
        }
        
        # Save length-controlled leaderboard
        leaderboard_length.to_csv(output_dir / "leaderboard_response_length.csv", index=True)
        bootstrap_length.to_csv(output_dir / "bootstrap_response_length.csv", index=True)
        
        if length_coef is not None:
            np.save(output_dir / "style_coefficients_length.npy", length_coef)
            
    except Exception as e:
        print(f"Error generating length-controlled leaderboard: {e}")
        results['response_length'] = None
    
    # 4. Citation count control
    print("\nGenerating citation count controlled leaderboard...")
    try:
        citation_elements = ["num_citations_a", "num_citations_b"]
        
        leaderboard_citations, bootstrap_citations, citation_coef = run_leaderboard(
            battle_data,
            anchor_model=anchor_model,
            anchor_rating=anchor_rating,
            style_elements=citation_elements,
            num_bootstrap_samples=num_bootstrap
        )
        
        results['citation_count'] = {
            'leaderboard': leaderboard_citations,
            'bootstrap': bootstrap_citations,
            'coefficients': citation_coef
        }
        
        # Save citation-controlled leaderboard
        leaderboard_citations.to_csv(output_dir / "leaderboard_citation_count.csv", index=True)
        bootstrap_citations.to_csv(output_dir / "bootstrap_citation_count.csv", index=True)
        
        if citation_coef is not None:
            np.save(output_dir / "style_coefficients_citations.npy", citation_coef)
            
    except Exception as e:
        print(f"Error generating citation-controlled leaderboard: {e}")
        results['citation_count'] = None
    
    return results


def create_summary_report(results, output_dir):
    """Create a summary report of all leaderboard variants."""
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'variants': {}
    }
    
    for variant_name, variant_data in results.items():
        if variant_data is not None:
            leaderboard = variant_data['leaderboard']
            
            report['variants'][variant_name] = {
                'num_models': len(leaderboard),
                'top_3_models': leaderboard.index[:3].tolist(),
                'top_3_ratings': leaderboard['rating'].head(3).tolist(),
                'rating_spread': float(leaderboard['rating'].max() - leaderboard['rating'].min())
            }
    
    # Save report
    with open(output_dir / "leaderboard_summary.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSummary report saved to {output_dir / 'leaderboard_summary.json'}")
    
    return report


def main():
    """Main function for Snakemake execution."""
    
    # Parse command line arguments (Snakemake style)
    if len(sys.argv) != 3:
        print("Usage: python 07_generate_leaderboard.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration (default values)
    config = {
        'anchor_model': 'api-gpt-4o-search',
        'anchor_rating': 1000,
        'num_bootstrap_samples': 100,
        'filter_ties': False
    }
    
    try:
        # Load cleaned data
        threads_df, responses_df, citations_df = load_cleaned_data(input_dir)
        
        # Transform to leaderboard format
        print("\nTransforming to leaderboard format...")
        battle_data = create_leaderboard_format(threads_df, responses_df, citations_df)
        
        print(f"Created leaderboard format with {len(battle_data)} conversations")
        
        # Filter ties if requested
        if config.get('filter_ties', False):
            original_len = len(battle_data)
            battle_data = battle_data[~battle_data['winner'].isin(['tie', 'tie (bothbad)'])]
            print(f"Filtered ties: {original_len} -> {len(battle_data)} conversations")
        
        # Save the transformed data
        battle_data.to_parquet(output_dir / "battle_data.parquet")
        
        # Generate leaderboard variants
        print("\nGenerating leaderboard variants...")
        results = generate_leaderboard_variants(battle_data, output_dir, config)
        
        # Create summary report
        summary = create_summary_report(results, output_dir)
        
        print("\nLeaderboard generation completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        # Print summary
        print("\nSummary:")
        for variant, data in summary['variants'].items():
            print(f"  {variant}: Top 3 = {data['top_3_models']}")
        
    except Exception as e:
        print(f"Error in leaderboard generation: {e}")
        raise


if __name__ == "__main__":
    main()