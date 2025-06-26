#!/usr/bin/env python3
"""
Generate comprehensive news citation statistics report.

This script produces detailed statistics about news citations including
temporal patterns, domain analysis, model comparison, political bias
distribution, and source quality metrics for research papers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_news_citations(data_path):
    """Load the news citations dataset."""
    logger.info(f"Loading news citations from {data_path}")
    data = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(data):,} news citations with {len(data.columns)} columns")
    return data


def analyze_temporal_patterns(data):
    """Analyze temporal patterns in news citations."""
    logger.info("Analyzing temporal patterns...")
    
    stats = {}
    
    if 'timestamp' in data.columns:
        # Convert to datetime
        data['datetime'] = pd.to_datetime(data['timestamp'])
        
        # Basic temporal statistics
        stats['date_range'] = {
            'earliest': data['datetime'].min().isoformat(),
            'latest': data['datetime'].max().isoformat(),
            'span_days': (data['datetime'].max() - data['datetime'].min()).days
        }
        
        # Daily citation patterns
        daily_counts = data['datetime'].dt.date.value_counts().sort_index()
        stats['daily_patterns'] = {
            'avg_citations_per_day': float(daily_counts.mean()),
            'peak_day': daily_counts.idxmax().isoformat(),
            'peak_citations': int(daily_counts.max()),
            'total_active_days': len(daily_counts)
        }
        
        # Weekly patterns
        data['weekday'] = data['datetime'].dt.day_name()
        weekday_counts = data['weekday'].value_counts()
        stats['weekly_patterns'] = {
            'citations_by_weekday': weekday_counts.to_dict(),
            'busiest_weekday': weekday_counts.idxmax(),
            'quietest_weekday': weekday_counts.idxmin()
        }
        
        # Hourly patterns (if time component available)
        if data['datetime'].dt.hour.nunique() > 1:
            hourly_counts = data['datetime'].dt.hour.value_counts().sort_index()
            stats['hourly_patterns'] = {
                'peak_hour': int(hourly_counts.idxmax()),
                'quiet_hour': int(hourly_counts.idxmin()),
                'citations_by_hour': hourly_counts.to_dict()
            }
    
    return stats


def analyze_domain_patterns(data):
    """Analyze news domain citation patterns."""
    logger.info("Analyzing domain patterns...")
    
    stats = {}
    
    # Basic domain statistics
    domain_counts = data['domain'].value_counts()
    stats['domain_overview'] = {
        'unique_domains': len(domain_counts),
        'total_citations': len(data),
        'avg_citations_per_domain': float(domain_counts.mean()),
        'median_citations_per_domain': float(domain_counts.median())
    }
    
    # Top domains analysis
    top_domains = domain_counts.head(20)
    stats['top_domains'] = {
        'top_20_domains': {
            domain: {
                'citations': int(count),
                'percentage': float(count / len(data) * 100)
            }
            for domain, count in top_domains.items()
        },
        'top_10_coverage': float(domain_counts.head(10).sum() / len(data) * 100),
        'top_20_coverage': float(top_domains.sum() / len(data) * 100)
    }
    
    # Domain concentration analysis
    stats['concentration'] = {
        'domains_with_1_citation': int((domain_counts == 1).sum()),
        'domains_with_10_plus': int((domain_counts >= 10).sum()),
        'domains_with_100_plus': int((domain_counts >= 100).sum()),
        'domains_with_1000_plus': int((domain_counts >= 1000).sum())
    }
    
    # Base domain analysis (removing subdomains)
    if 'base_domain' in data.columns:
        base_domain_counts = data['base_domain'].value_counts()
        stats['base_domain_analysis'] = {
            'unique_base_domains': len(base_domain_counts),
            'top_base_domains': {
                domain: int(count) 
                for domain, count in base_domain_counts.head(10).items()
            }
        }
    
    return stats


def analyze_model_comparison(data):
    """Analyze news citation patterns by AI model."""
    logger.info("Analyzing model comparison...")
    
    stats = {}
    
    # Determine model column
    model_col = None
    for col in ['model_name_raw', 'model_name_llm', 'model']:
        if col in data.columns:
            model_col = col
            break
    
    if model_col is None:
        logger.warning("No model column found")
        return stats
    
    # Citations by model
    model_counts = data[model_col].value_counts()
    stats['citations_by_model'] = {
        model: {
            'citations': int(count),
            'percentage': float(count / len(data) * 100)
        }
        for model, count in model_counts.items()
    }
    
    # Model diversity in news citation
    stats['model_overview'] = {
        'unique_models': len(model_counts),
        'most_citing_model': model_counts.idxmax(),
        'least_citing_model': model_counts.idxmin(),
        'citations_range': {
            'max': int(model_counts.max()),
            'min': int(model_counts.min())
        }
    }
    
    # Model family analysis (if available)
    if 'model_family' in data.columns:
        family_counts = data['model_family'].value_counts()
        stats['model_family_analysis'] = {
            'unique_families': len(family_counts),
            'citations_by_family': family_counts.to_dict()
        }
    
    # Average citations per response by model
    if 'response_id' in data.columns:
        model_response_stats = data.groupby(model_col).agg({
            'response_id': ['nunique', 'count']
        }).round(2)
        
        stats['citations_per_response_by_model'] = {}
        for model in model_response_stats.index:
            responses = model_response_stats.loc[model, ('response_id', 'nunique')]
            citations = model_response_stats.loc[model, ('response_id', 'count')]
            stats['citations_per_response_by_model'][model] = {
                'avg_citations_per_response': float(citations / responses),
                'total_responses_with_news': int(responses),
                'total_news_citations': int(citations)
            }
    
    return stats


def analyze_political_bias_patterns(data):
    """Analyze political bias patterns in news citations."""
    logger.info("Analyzing political bias patterns...")
    
    stats = {}
    
    # Check if bias data is available
    bias_data = data[data['political_leaning_score'].notna()].copy()
    if len(bias_data) == 0:
        logger.warning("No political bias data available")
        return stats
    
    logger.info(f"Analyzing {len(bias_data):,} citations with bias scores")
    
    # Overall bias distribution
    bias_scores = bias_data['political_leaning_score']
    stats['bias_score_distribution'] = {
        'total_with_bias_scores': len(bias_data),
        'coverage_percentage': float(len(bias_data) / len(data) * 100),
        'mean_score': float(bias_scores.mean()),
        'median_score': float(bias_scores.median()),
        'std_score': float(bias_scores.std()),
        'min_score': float(bias_scores.min()),
        'max_score': float(bias_scores.max()),
        'quartiles': {
            'q25': float(bias_scores.quantile(0.25)),
            'q75': float(bias_scores.quantile(0.75))
        }
    }
    
    # Political leaning labels
    if 'political_leaning' in bias_data.columns:
        leaning_counts = bias_data['political_leaning'].value_counts()
        stats['political_leaning_distribution'] = {
            label: {
                'count': int(count),
                'percentage': float(count / len(bias_data) * 100)
            }
            for label, count in leaning_counts.items()
        }
    
    # Bias by model analysis
    model_col = None
    for col in ['model_name_raw', 'model_name_llm', 'model']:
        if col in bias_data.columns:
            model_col = col
            break
    
    if model_col:
        model_bias_stats = bias_data.groupby(model_col)['political_leaning_score'].agg([
            'count', 'mean', 'median', 'std'
        ]).round(3)
        
        stats['bias_by_model'] = {
            model: {
                'citations_with_bias': int(row['count']),
                'mean_bias': float(row['mean']),
                'median_bias': float(row['median']),
                'std_bias': float(row['std'])
            }
            for model, row in model_bias_stats.iterrows()
        }
    
    # Most biased domains
    domain_bias = bias_data.groupby('domain')['political_leaning_score'].agg([
        'count', 'mean', 'median'
    ]).sort_values('mean', key=abs, ascending=False)
    
    # Filter domains with multiple citations for reliability
    reliable_domains = domain_bias[domain_bias['count'] >= 5].head(10)
    if len(reliable_domains) > 0:
        stats['most_biased_domains'] = {
            domain: {
                'citations': int(row['count']),
                'mean_bias': float(row['mean']),
                'median_bias': float(row['median'])
            }
            for domain, row in reliable_domains.iterrows()
        }
    
    return stats


def analyze_quality_patterns(data):
    """Analyze source quality patterns in news citations."""
    logger.info("Analyzing source quality patterns...")
    
    stats = {}
    
    # Check if quality data is available
    quality_data = data[data['domain_quality_score'].notna()].copy()
    if len(quality_data) == 0:
        logger.warning("No source quality data available")
        return stats
    
    logger.info(f"Analyzing {len(quality_data):,} citations with quality scores")
    
    # Overall quality distribution
    quality_scores = quality_data['domain_quality_score']
    stats['quality_score_distribution'] = {
        'total_with_quality_scores': len(quality_data),
        'coverage_percentage': float(len(quality_data) / len(data) * 100),
        'mean_score': float(quality_scores.mean()),
        'median_score': float(quality_scores.median()),
        'std_score': float(quality_scores.std()),
        'min_score': float(quality_scores.min()),
        'max_score': float(quality_scores.max()),
        'quartiles': {
            'q25': float(quality_scores.quantile(0.25)),
            'q75': float(quality_scores.quantile(0.75))
        }
    }
    
    # Quality labels
    if 'domain_quality' in quality_data.columns:
        quality_counts = quality_data['domain_quality'].value_counts()
        stats['quality_label_distribution'] = {
            label: {
                'count': int(count),
                'percentage': float(count / len(quality_data) * 100)
            }
            for label, count in quality_counts.items()
        }
    
    # Quality by model analysis
    model_col = None
    for col in ['model_name_raw', 'model_name_llm', 'model']:
        if col in quality_data.columns:
            model_col = col
            break
    
    if model_col:
        model_quality_stats = quality_data.groupby(model_col)['domain_quality_score'].agg([
            'count', 'mean', 'median', 'std'
        ]).round(3)
        
        stats['quality_by_model'] = {
            model: {
                'citations_with_quality': int(row['count']),
                'mean_quality': float(row['mean']),
                'median_quality': float(row['median']),
                'std_quality': float(row['std'])
            }
            for model, row in model_quality_stats.iterrows()
        }
    
    # Highest quality domains
    domain_quality = quality_data.groupby('domain')['domain_quality_score'].agg([
        'count', 'mean', 'median'
    ]).sort_values('mean', ascending=False)
    
    # Filter domains with multiple citations for reliability
    reliable_domains = domain_quality[domain_quality['count'] >= 5].head(10)
    if len(reliable_domains) > 0:
        stats['highest_quality_domains'] = {
            domain: {
                'citations': int(row['count']),
                'mean_quality': float(row['mean']),
                'median_quality': float(row['median'])
            }
            for domain, row in reliable_domains.iterrows()
        }
    
    return stats


def analyze_joint_bias_quality(data):
    """Analyze joint bias and quality patterns."""
    logger.info("Analyzing joint bias and quality patterns...")
    
    stats = {}
    
    # Filter for citations with both bias and quality scores
    joint_data = data[
        data['political_leaning_score'].notna() & 
        data['domain_quality_score'].notna()
    ].copy()
    
    if len(joint_data) == 0:
        logger.warning("No citations with both bias and quality scores")
        return stats
    
    logger.info(f"Analyzing {len(joint_data):,} citations with both scores")
    
    stats['joint_coverage'] = {
        'citations_with_both_scores': len(joint_data),
        'percentage_of_total': float(len(joint_data) / len(data) * 100),
        'unique_domains_with_both': joint_data['domain'].nunique()
    }
    
    # Correlation analysis
    correlation = joint_data['political_leaning_score'].corr(joint_data['domain_quality_score'])
    stats['bias_quality_correlation'] = {
        'correlation_coefficient': float(correlation),
        'correlation_interpretation': 'weak' if abs(correlation) < 0.3 else 'moderate' if abs(correlation) < 0.7 else 'strong'
    }
    
    # Quadrant analysis (bias vs quality)
    bias_median = joint_data['political_leaning_score'].median()
    quality_median = joint_data['domain_quality_score'].median()
    
    quadrants = {
        'high_quality_left_leaning': len(joint_data[
            (joint_data['political_leaning_score'] < bias_median) & 
            (joint_data['domain_quality_score'] > quality_median)
        ]),
        'high_quality_right_leaning': len(joint_data[
            (joint_data['political_leaning_score'] > bias_median) & 
            (joint_data['domain_quality_score'] > quality_median)
        ]),
        'low_quality_left_leaning': len(joint_data[
            (joint_data['political_leaning_score'] < bias_median) & 
            (joint_data['domain_quality_score'] < quality_median)
        ]),
        'low_quality_right_leaning': len(joint_data[
            (joint_data['political_leaning_score'] > bias_median) & 
            (joint_data['domain_quality_score'] < quality_median)
        ])
    }
    
    stats['bias_quality_quadrants'] = {
        key: {
            'count': count,
            'percentage': float(count / len(joint_data) * 100)
        }
        for key, count in quadrants.items()
    }
    
    return stats


def generate_markdown_report(all_stats, output_path):
    """Generate comprehensive markdown report."""
    logger.info("Generating markdown report...")
    
    report_lines = [
        "# News Citations Statistics Report",
        "",
        f"*Generated on: {datetime.now().isoformat()}*",
        "",
        "## Executive Summary",
        ""
    ]
    
    # Executive summary
    if 'domain_patterns' in all_stats:
        domain_stats = all_stats['domain_patterns']['domain_overview']
        report_lines.extend([
            f"- **Total News Citations**: {domain_stats['total_citations']:,}",
            f"- **Unique News Domains**: {domain_stats['unique_domains']:,}",
            f"- **Average Citations per Domain**: {domain_stats['avg_citations_per_domain']:.1f}",
            ""
        ])
    
    # Temporal patterns
    if 'temporal_patterns' in all_stats:
        temporal = all_stats['temporal_patterns']
        if 'date_range' in temporal:
            report_lines.extend([
                "## Temporal Patterns",
                "",
                f"- **Date Range**: {temporal['date_range']['earliest']} to {temporal['date_range']['latest']}",
                f"- **Span**: {temporal['date_range']['span_days']} days",
                ""
            ])
            
            if 'daily_patterns' in temporal:
                daily = temporal['daily_patterns']
                report_lines.extend([
                    f"- **Average Citations per Day**: {daily['avg_citations_per_day']:.1f}",
                    f"- **Peak Day**: {daily['peak_day']} ({daily['peak_citations']:,} citations)",
                    f"- **Total Active Days**: {daily['total_active_days']:,}",
                    ""
                ])
    
    # Domain analysis
    if 'domain_patterns' in all_stats:
        domain = all_stats['domain_patterns']
        
        report_lines.extend([
            "## Domain Analysis",
            "",
            f"### Domain Overview",
            f"- **Unique Domains**: {domain['domain_overview']['unique_domains']:,}",
            f"- **Average Citations per Domain**: {domain['domain_overview']['avg_citations_per_domain']:.1f}",
            f"- **Median Citations per Domain**: {domain['domain_overview']['median_citations_per_domain']:.1f}",
            ""
        ])
        
        if 'top_domains' in domain:
            top = domain['top_domains']
            report_lines.extend([
                f"### Top News Domains",
                f"- **Top 10 Coverage**: {top['top_10_coverage']:.1f}% of all news citations",
                f"- **Top 20 Coverage**: {top['top_20_coverage']:.1f}% of all news citations",
                "",
                "**Top 10 Most Cited News Domains:**",
                ""
            ])
            
            for i, (domain_name, stats) in enumerate(list(top['top_20_domains'].items())[:10], 1):
                report_lines.append(f"{i}. **{domain_name}**: {stats['citations']:,} citations ({stats['percentage']:.1f}%)")
            report_lines.append("")
    
    # Model comparison
    if 'model_comparison' in all_stats:
        model = all_stats['model_comparison']
        
        report_lines.extend([
            "## Model Comparison",
            "",
            f"### News Citations by Model",
            f"- **Unique Models**: {model['model_overview']['unique_models']}",
            f"- **Most Citing Model**: {model['model_overview']['most_citing_model']}",
            ""
        ])
        
        # Top models by citations
        sorted_models = sorted(
            model['citations_by_model'].items(),
            key=lambda x: x[1]['citations'],
            reverse=True
        )
        
        for model_name, stats in sorted_models[:10]:
            report_lines.append(f"- **{model_name}**: {stats['citations']:,} citations ({stats['percentage']:.1f}%)")
        report_lines.append("")
    
    # Political bias analysis
    if 'political_bias' in all_stats:
        bias = all_stats['political_bias']
        
        if 'bias_score_distribution' in bias:
            bias_dist = bias['bias_score_distribution']
            report_lines.extend([
                "## Political Bias Analysis",
                "",
                f"### Bias Score Coverage",
                f"- **Citations with Bias Scores**: {bias_dist['total_with_bias_scores']:,} ({bias_dist['coverage_percentage']:.1f}%)",
                f"- **Mean Bias Score**: {bias_dist['mean_score']:.3f}",
                f"- **Median Bias Score**: {bias_dist['median_score']:.3f}",
                f"- **Score Range**: {bias_dist['min_score']:.3f} to {bias_dist['max_score']:.3f}",
                ""
            ])
            
        if 'political_leaning_distribution' in bias:
            leaning_dist = bias['political_leaning_distribution']
            report_lines.extend([
                "### Political Leaning Distribution",
                ""
            ])
            
            for leaning, stats in sorted(leaning_dist.items(), key=lambda x: x[1]['count'], reverse=True):
                report_lines.append(f"- **{leaning}**: {stats['count']:,} citations ({stats['percentage']:.1f}%)")
            report_lines.append("")
    
    # Quality analysis
    if 'quality_patterns' in all_stats:
        quality = all_stats['quality_patterns']
        
        if 'quality_score_distribution' in quality:
            quality_dist = quality['quality_score_distribution']
            report_lines.extend([
                "## Source Quality Analysis",
                "",
                f"### Quality Score Coverage",
                f"- **Citations with Quality Scores**: {quality_dist['total_with_quality_scores']:,} ({quality_dist['coverage_percentage']:.1f}%)",
                f"- **Mean Quality Score**: {quality_dist['mean_score']:.3f}",
                f"- **Median Quality Score**: {quality_dist['median_score']:.3f}",
                f"- **Score Range**: {quality_dist['min_score']:.3f} to {quality_dist['max_score']:.3f}",
                ""
            ])
    
    # Joint analysis
    if 'joint_analysis' in all_stats:
        joint = all_stats['joint_analysis']
        
        if 'joint_coverage' in joint:
            coverage = joint['joint_coverage']
            report_lines.extend([
                "## Joint Bias & Quality Analysis",
                "",
                f"### Coverage",
                f"- **Citations with Both Scores**: {coverage['citations_with_both_scores']:,} ({coverage['percentage_of_total']:.1f}%)",
                f"- **Domains with Both Scores**: {coverage['unique_domains_with_both']:,}",
                ""
            ])
            
        if 'bias_quality_correlation' in joint:
            corr = joint['bias_quality_correlation']
            report_lines.extend([
                f"### Correlation Analysis",
                f"- **Bias-Quality Correlation**: {corr['correlation_coefficient']:.3f} ({corr['correlation_interpretation']})",
                ""
            ])
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))


def main():
    """Main function for news statistics generation."""
    # Get paths from Snakemake
    input_path = snakemake.input.news_citations
    stats_output_path = snakemake.output.statistics_json
    report_output_path = snakemake.output.statistics_report
    
    # Create output directory
    output_dir = Path(stats_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load news citations
    data = load_news_citations(input_path)
    
    # Generate all analyses
    all_stats = {
        'generation_timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_citations': len(data),
            'columns': list(data.columns)
        },
        'temporal_patterns': analyze_temporal_patterns(data),
        'domain_patterns': analyze_domain_patterns(data),
        'model_comparison': analyze_model_comparison(data),
        'political_bias': analyze_political_bias_patterns(data),
        'quality_patterns': analyze_quality_patterns(data),
        'joint_analysis': analyze_joint_bias_quality(data)
    }
    
    # Save statistics as JSON
    logger.info(f"Saving statistics to {stats_output_path}")
    with open(stats_output_path, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    
    # Generate markdown report
    generate_markdown_report(all_stats, report_output_path)
    logger.info(f"Saved markdown report to {report_output_path}")
    
    # Print summary
    logger.info("=== NEWS CITATION STATISTICS SUMMARY ===")
    logger.info(f"Total news citations analyzed: {len(data):,}")
    if 'domain_patterns' in all_stats:
        logger.info(f"Unique domains: {all_stats['domain_patterns']['domain_overview']['unique_domains']:,}")
    if 'political_bias' in all_stats and 'bias_score_distribution' in all_stats['political_bias']:
        bias_coverage = all_stats['political_bias']['bias_score_distribution']['coverage_percentage']
        logger.info(f"Political bias coverage: {bias_coverage:.1f}%")
    if 'quality_patterns' in all_stats and 'quality_score_distribution' in all_stats['quality_patterns']:
        quality_coverage = all_stats['quality_patterns']['quality_score_distribution']['coverage_percentage']
        logger.info(f"Quality score coverage: {quality_coverage:.1f}%")
    
    logger.info("✅ News citation statistics generation completed!")


if __name__ == "__main__":
    main()