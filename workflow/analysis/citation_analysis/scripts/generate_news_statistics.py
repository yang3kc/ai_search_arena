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


def load_all_data(input_files):
    """Load all required datasets."""
    logger.info("Loading all datasets...")
    
    # Load news citations
    logger.info(f"Loading news citations from {input_files.news_citations}")
    news_citations = pd.read_parquet(input_files.news_citations)
    logger.info(f"Loaded {len(news_citations):,} news citations")
    
    # Load threads
    logger.info(f"Loading threads from {input_files.threads}")
    threads = pd.read_parquet(input_files.threads)
    logger.info(f"Loaded {len(threads):,} threads")
    
    # Load questions
    logger.info(f"Loading questions from {input_files.questions}")
    questions = pd.read_parquet(input_files.questions)
    logger.info(f"Loaded {len(questions):,} questions")
    
    # Load responses
    logger.info(f"Loading responses from {input_files.responses}")
    responses = pd.read_parquet(input_files.responses)
    logger.info(f"Loaded {len(responses):,} responses")
    
    return {
        'news_citations': news_citations,
        'threads': threads,
        'questions': questions,
        'responses': responses
    }


def analyze_thread_patterns(threads, news_citations):
    """Analyze thread-level patterns for news citations."""
    logger.info("Analyzing thread patterns...")
    
    stats = {}
    
    # Get threads that have news citations - focus only on these
    news_thread_ids = set(news_citations['thread_id'].unique())
    news_threads = threads[threads['thread_id'].isin(news_thread_ids)].copy()
    
    stats['thread_overview'] = {
        'news_threads_analyzed': len(news_threads),
        'unique_news_thread_ids': len(news_thread_ids)
    }
    
    # Winner analysis for news threads
    if 'winner' in news_threads.columns:
        winner_dist = news_threads['winner'].value_counts()
        threads_with_winner = len(news_threads[news_threads['winner'].notna()])
        stats['winner_analysis'] = {
            'threads_with_winner': threads_with_winner,
            'winner_distribution': {
                winner: {
                    'count': int(count),
                    'percentage': float(count / threads_with_winner * 100) if threads_with_winner > 0 else 0
                }
                for winner, count in winner_dist.items()
            } if len(winner_dist) > 0 else {}
        }
    
    # Intent analysis for news threads
    if 'primary_intent' in news_threads.columns:
        intent_dist = news_threads['primary_intent'].value_counts()
        total_threads = len(news_threads)
        stats['intent_analysis'] = {
            'primary_intent_distribution': {
                intent: {
                    'count': int(count),
                    'percentage': float(count / total_threads * 100)
                }
                for intent, count in intent_dist.items()
            }
        }
    
    # Turn analysis
    if 'total_turns' in news_threads.columns:
        turn_stats = news_threads['total_turns']
        stats['conversation_length'] = {
            'avg_turns': float(turn_stats.mean()),
            'median_turns': float(turn_stats.median()),
            'max_turns': int(turn_stats.max()),
            'min_turns': int(turn_stats.min())
        }
    
    
    return stats


def analyze_question_patterns(questions, news_citations):
    """Analyze question-level patterns for news citations."""
    logger.info("Analyzing question patterns...")
    
    stats = {}
    
    # Get questions that lead to news citations - focus only on these
    news_question_ids = set(news_citations['question_id'].unique())
    news_questions = questions[questions['question_id'].isin(news_question_ids)].copy()
    
    stats['question_overview'] = {
        'news_questions_analyzed': len(news_questions),
        'unique_news_question_ids': len(news_question_ids)
    }
    
    # Turn number analysis
    if 'turn_number' in news_questions.columns:
        turn_dist = news_questions['turn_number'].value_counts().sort_index()
        total_questions = len(news_questions)
        stats['turn_analysis'] = {
            'questions_by_turn': {
                turn: {
                    'count': int(count),
                    'percentage': float(count / total_questions * 100)
                }
                for turn, count in turn_dist.items()
            },
            'avg_turn_for_news': float(news_questions['turn_number'].mean()),
            'first_turn_news_percentage': float((news_questions['turn_number'] == 1).sum() / len(news_questions) * 100)
        }
    
    # Question text analysis (if available)
    if 'question_text' in news_questions.columns:
        question_lengths = news_questions['question_text'].str.len()
        stats['question_text_analysis'] = {
            'avg_question_length': float(question_lengths.mean()),
            'median_question_length': float(question_lengths.median()),
            'max_question_length': int(question_lengths.max()),
            'min_question_length': int(question_lengths.min())
        }
    
    return stats


def analyze_response_patterns(responses, news_citations):
    """Analyze response-level patterns for news citations."""
    logger.info("Analyzing response patterns...")
    
    stats = {}
    
    # Get responses that include news citations - focus only on these
    news_response_ids = set(news_citations['response_id'].unique())
    news_responses = responses[responses['response_id'].isin(news_response_ids)].copy()
    
    stats['response_overview'] = {
        'news_responses_analyzed': len(news_responses),
        'unique_news_response_ids': len(news_response_ids)
    }
    
    # Model analysis for news responses
    model_col = None
    for col in ['model_name_raw', 'model_name_llm', 'model']:
        if col in news_responses.columns:
            model_col = col
            break
    
    if model_col:
        model_dist = news_responses[model_col].value_counts()
        total_responses = len(news_responses)
        stats['model_analysis'] = {
            'responses_by_model': {
                model: {
                    'count': int(count),
                    'percentage': float(count / total_responses * 100)
                }
                for model, count in model_dist.items()
            },
            'unique_models_with_news': len(model_dist)
        }
    
    # Model side analysis
    if 'model_side' in news_responses.columns:
        side_dist = news_responses['model_side'].value_counts()
        total_responses = len(news_responses)
        stats['model_side_analysis'] = {
            'responses_by_side': {
                side: {
                    'count': int(count),
                    'percentage': float(count / total_responses * 100)
                }
                for side, count in side_dist.items()
            }
        }
    
    # Citation format analysis
    if 'citation_format' in news_citations.columns:
        format_dist = news_citations['citation_format'].value_counts()
        total_citations = len(news_citations)
        stats['citation_format_analysis'] = {
            'citations_by_format': {
                format_type: {
                    'count': int(count),
                    'percentage': float(count / total_citations * 100)
                }
                for format_type, count in format_dist.items()
            }
        }
    
    # News citations per response analysis
    citations_per_response = news_citations.groupby('response_id').size()
    stats['citations_per_response'] = {
        'avg_news_citations_per_response': float(citations_per_response.mean()),
        'median_news_citations_per_response': float(citations_per_response.median()),
        'max_news_citations_per_response': int(citations_per_response.max()),
        'min_news_citations_per_response': int(citations_per_response.min())
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
    
    # Domain-level bias coverage analysis
    unique_domains = data['domain'].nunique()
    domains_with_bias = bias_data['domain'].nunique()
    stats['domain_bias_coverage'] = {
        'total_unique_domains': unique_domains,
        'domains_with_bias_scores': domains_with_bias,
        'domain_coverage_percentage': float(domains_with_bias / unique_domains * 100)
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
    
    # Domain-level quality coverage analysis
    unique_domains = data['domain'].nunique()
    domains_with_quality = quality_data['domain'].nunique()
    stats['domain_quality_coverage'] = {
        'total_unique_domains': unique_domains,
        'domains_with_quality_scores': domains_with_quality,
        'domain_coverage_percentage': float(domains_with_quality / unique_domains * 100)
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


def analyze_news_relationships(news_citations):
    """Analyze relationships between news citations and their parent entities."""
    logger.info("Analyzing news citation relationships...")
    
    stats = {}
    
    # Citations per thread
    citations_per_thread = news_citations.groupby('thread_id').size()
    stats['citations_per_thread'] = {
        'avg_citations_per_thread': float(citations_per_thread.mean()),
        'median_citations_per_thread': float(citations_per_thread.median()),
        'max_citations_per_thread': int(citations_per_thread.max()),
        'min_citations_per_thread': int(citations_per_thread.min())
    }
    
    # Citations per question
    if 'question_id' in news_citations.columns:
        citations_per_question = news_citations.groupby('question_id').size()
        stats['citations_per_question'] = {
            'avg_citations_per_question': float(citations_per_question.mean()),
            'median_citations_per_question': float(citations_per_question.median()),
            'max_citations_per_question': int(citations_per_question.max()),
            'min_citations_per_question': int(citations_per_question.min())
        }
    
    # Citations per response (already calculated elsewhere but including for completeness)
    citations_per_response = news_citations.groupby('response_id').size()
    stats['citations_per_response'] = {
        'avg_citations_per_response': float(citations_per_response.mean()),
        'median_citations_per_response': float(citations_per_response.median()),
        'max_citations_per_response': int(citations_per_response.max()),
        'min_citations_per_response': int(citations_per_response.min())
    }
    
    # Thread-level statistics
    if 'question_id' in news_citations.columns:
        thread_stats = news_citations.groupby('thread_id').agg({
            'response_id': 'nunique',  # responses per thread
            'question_id': 'nunique',  # questions per thread
            'citation_id': 'count'  # citations per thread
        })
        
        stats['thread_level_stats'] = {
            'avg_responses_per_thread': float(thread_stats['response_id'].mean()),
            'avg_questions_per_thread': float(thread_stats['question_id'].mean()),
            'median_responses_per_thread': float(thread_stats['response_id'].median()),
            'median_questions_per_thread': float(thread_stats['question_id'].median())
        }
    else:
        thread_stats = news_citations.groupby('thread_id').agg({
            'response_id': 'nunique',  # responses per thread
            'citation_id': 'count'  # citations per thread
        })
        
        stats['thread_level_stats'] = {
            'avg_responses_per_thread': float(thread_stats['response_id'].mean()),
            'median_responses_per_thread': float(thread_stats['response_id'].median())
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
    
    # Domain-level joint coverage
    unique_domains = data['domain'].nunique()
    domains_with_both = joint_data['domain'].nunique()
    
    stats['joint_coverage'] = {
        'citations_with_both_scores': len(joint_data),
        'percentage_of_total': float(len(joint_data) / len(data) * 100),
        'unique_domains_with_both': domains_with_both,
        'domain_joint_coverage_percentage': float(domains_with_both / unique_domains * 100)
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
    if 'dataset_info' in all_stats:
        dataset_info = all_stats['dataset_info']
        report_lines.extend([
            f"- **Total News Citations**: {dataset_info['total_news_citations']:,}",
            ""
        ])
        
    # Add counts of news-related entities
    if 'thread_patterns' in all_stats and 'thread_overview' in all_stats['thread_patterns']:
        thread_overview = all_stats['thread_patterns']['thread_overview']
        report_lines.extend([
            f"- **News-Related Threads**: {thread_overview['news_threads_analyzed']:,}",
        ])
        
    if 'question_patterns' in all_stats and 'question_overview' in all_stats['question_patterns']:
        question_overview = all_stats['question_patterns']['question_overview']
        report_lines.extend([
            f"- **News-Related Questions**: {question_overview['news_questions_analyzed']:,}",
        ])
        
    if 'response_patterns' in all_stats and 'response_overview' in all_stats['response_patterns']:
        response_overview = all_stats['response_patterns']['response_overview']
        report_lines.extend([
            f"- **News-Related Responses**: {response_overview['news_responses_analyzed']:,}",
            ""
        ])
        
    if 'domain_patterns' in all_stats:
        domain_stats = all_stats['domain_patterns']['domain_overview']
        report_lines.extend([
            f"- **Unique News Domains**: {domain_stats['unique_domains']:,}",
            f"- **Average Citations per Domain**: {domain_stats['avg_citations_per_domain']:.1f}",
            ""
        ])
    
    # Thread analysis
    if 'thread_patterns' in all_stats:
        thread = all_stats['thread_patterns']
        
        report_lines.extend([
            "## News-Related Thread Analysis",
            ""
        ])
        
        if 'thread_overview' in thread:
            overview = thread['thread_overview']
            report_lines.extend([
                f"### Thread Overview",
                f"- **News-Related Threads Analyzed**: {overview['news_threads_analyzed']:,}",
                f"- **Unique Thread IDs**: {overview['unique_news_thread_ids']:,}",
                ""
            ])
        
        if 'winner_analysis' in thread:
            winner = thread['winner_analysis']
            report_lines.extend([
                f"### Winner Analysis (News Threads)",
                f"- **Threads with Winners**: {winner['threads_with_winner']:,}",
                ""
            ])
            
            if winner['winner_distribution']:
                report_lines.append("**Winner Distribution:**")
                for side, stats in sorted(winner['winner_distribution'].items(), key=lambda x: x[1]['count'], reverse=True):
                    report_lines.append(f"  - {side}: {stats['count']:,} threads ({stats['percentage']:.1f}%)")
                report_lines.append("")
        
        if 'intent_analysis' in thread:
            intent = thread['intent_analysis']
            if intent['primary_intent_distribution']:
                report_lines.extend([
                    "### Intent Analysis (News Threads)",
                    "**Primary Intent Distribution:**",
                ])
                for intent_type, stats in sorted(intent['primary_intent_distribution'].items(), key=lambda x: x[1]['count'], reverse=True):
                    report_lines.append(f"  - {intent_type}: {stats['count']:,} threads ({stats['percentage']:.1f}%)")
                report_lines.append("")
        
        if 'conversation_length' in thread:
            length = thread['conversation_length']
            report_lines.extend([
                f"### Conversation Length (News Threads)",
                f"- **Average Turns**: {length['avg_turns']:.1f}",
                f"- **Median Turns**: {length['median_turns']:.1f}",
                f"- **Range**: {length['min_turns']} to {length['max_turns']} turns",
                ""
            ])
    
    # Question analysis
    if 'question_patterns' in all_stats:
        question = all_stats['question_patterns']
        
        report_lines.extend([
            "## News-Related Question Analysis",
            ""
        ])
        
        if 'question_overview' in question:
            overview = question['question_overview']
            report_lines.extend([
                f"### Question Overview",
                f"- **News-Related Questions Analyzed**: {overview['news_questions_analyzed']:,}",
                f"- **Unique Question IDs**: {overview['unique_news_question_ids']:,}",
                ""
            ])
        
        if 'turn_analysis' in question:
            turn = question['turn_analysis']
            report_lines.extend([
                f"### Turn Analysis",
                f"- **Average Turn for News**: {turn['avg_turn_for_news']:.1f}",
                f"- **First Turn News Percentage**: {turn['first_turn_news_percentage']:.1f}%",
                ""
            ])
            
            if turn['questions_by_turn']:
                report_lines.append("**Questions by Turn Number:**")
                for turn_num, stats in sorted(turn['questions_by_turn'].items()):
                    report_lines.append(f"  - Turn {turn_num}: {stats['count']:,} questions ({stats['percentage']:.1f}%)")
                report_lines.append("")
        
        if 'question_text_analysis' in question:
            text = question['question_text_analysis']
            report_lines.extend([
                f"### Question Text Analysis",
                f"- **Average Length**: {text['avg_question_length']:.0f} characters",
                f"- **Median Length**: {text['median_question_length']:.0f} characters",
                f"- **Range**: {text['min_question_length']} to {text['max_question_length']} characters",
                ""
            ])
    
    # Response analysis
    if 'response_patterns' in all_stats:
        response = all_stats['response_patterns']
        
        report_lines.extend([
            "## News-Related Response Analysis",
            ""
        ])
        
        if 'response_overview' in response:
            overview = response['response_overview']
            report_lines.extend([
                f"### Response Overview",
                f"- **News-Related Responses Analyzed**: {overview['news_responses_analyzed']:,}",
                f"- **Unique Response IDs**: {overview['unique_news_response_ids']:,}",
                ""
            ])
        
        if 'citations_per_response' in response:
            cpr = response['citations_per_response']
            report_lines.extend([
                f"### News Citations per Response",
                f"- **Average**: {cpr['avg_news_citations_per_response']:.1f}",
                f"- **Median**: {cpr['median_news_citations_per_response']:.1f}",
                f"- **Range**: {cpr['min_news_citations_per_response']} to {cpr['max_news_citations_per_response']} citations",
                ""
            ])
        
        if 'model_side_analysis' in response:
            side = response['model_side_analysis']
            if side['responses_by_side']:
                report_lines.extend([
                    f"### Model Side Distribution",
                    ""
                ])
                for side_name, stats in sorted(side['responses_by_side'].items(), key=lambda x: x[1]['count'], reverse=True):
                    report_lines.append(f"- **{side_name}**: {stats['count']:,} responses ({stats['percentage']:.1f}%)")
                report_lines.append("")
        
        if 'citation_format_analysis' in response:
            format_analysis = response['citation_format_analysis']
            if format_analysis['citations_by_format']:
                report_lines.extend([
                    f"### Citation Format Distribution",
                    ""
                ])
                for format_type, stats in sorted(format_analysis['citations_by_format'].items(), key=lambda x: x[1]['count'], reverse=True):
                    report_lines.append(f"- **{format_type}**: {stats['count']:,} citations ({stats['percentage']:.1f}%)")
                report_lines.append("")
    
    # News relationships analysis
    if 'news_relationships' in all_stats:
        relationships = all_stats['news_relationships']
        
        report_lines.extend([
            "## News Citation Relationships",
            ""
        ])
        
        if 'citations_per_thread' in relationships:
            cpt = relationships['citations_per_thread']
            report_lines.extend([
                f"### Citations per Thread",
                f"- **Average**: {cpt['avg_citations_per_thread']:.1f}",
                f"- **Median**: {cpt['median_citations_per_thread']:.1f}",
                f"- **Range**: {cpt['min_citations_per_thread']} to {cpt['max_citations_per_thread']} citations",
                ""
            ])
        
        if 'citations_per_question' in relationships:
            cpq = relationships['citations_per_question']
            report_lines.extend([
                f"### Citations per Question",
                f"- **Average**: {cpq['avg_citations_per_question']:.1f}",
                f"- **Median**: {cpq['median_citations_per_question']:.1f}",
                f"- **Range**: {cpq['min_citations_per_question']} to {cpq['max_citations_per_question']} citations",
                ""
            ])
        
        if 'thread_level_stats' in relationships:
            tls = relationships['thread_level_stats']
            report_lines.extend([
                f"### Thread-Level Aggregation",
                f"- **Average Responses per Thread**: {tls['avg_responses_per_thread']:.1f}",
                f"- **Median Responses per Thread**: {tls['median_responses_per_thread']:.1f}",
            ])
            
            if 'avg_questions_per_thread' in tls:
                report_lines.extend([
                    f"- **Average Questions per Thread**: {tls['avg_questions_per_thread']:.1f}",
                    f"- **Median Questions per Thread**: {tls['median_questions_per_thread']:.1f}",
                ])
            report_lines.append("")
    
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
            
        if 'domain_bias_coverage' in bias:
            domain_bias = bias['domain_bias_coverage']
            report_lines.extend([
                f"### Domain-Level Bias Coverage",
                f"- **Total Unique News Domains**: {domain_bias['total_unique_domains']:,}",
                f"- **Domains with Bias Scores**: {domain_bias['domains_with_bias_scores']:,} ({domain_bias['domain_coverage_percentage']:.1f}%)",
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
            
        if 'domain_quality_coverage' in quality:
            domain_quality = quality['domain_quality_coverage']
            report_lines.extend([
                f"### Domain-Level Quality Coverage",
                f"- **Total Unique News Domains**: {domain_quality['total_unique_domains']:,}",
                f"- **Domains with Quality Scores**: {domain_quality['domains_with_quality_scores']:,} ({domain_quality['domain_coverage_percentage']:.1f}%)",
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
                f"- **Domains with Both Scores**: {coverage['unique_domains_with_both']:,} ({coverage['domain_joint_coverage_percentage']:.1f}%)",
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
    input_files = snakemake.input
    report_output_path = snakemake.output[0]
    
    # Create output directory
    output_dir = Path(report_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    all_data = load_all_data(input_files)
    news_citations = all_data['news_citations']
    
    # Generate all analyses
    all_stats = {
        'generation_timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_news_citations': len(news_citations),
            'news_citations_columns': list(news_citations.columns)
        },
        'thread_patterns': analyze_thread_patterns(all_data['threads'], news_citations),
        'question_patterns': analyze_question_patterns(all_data['questions'], news_citations),
        'response_patterns': analyze_response_patterns(all_data['responses'], news_citations),
        'news_relationships': analyze_news_relationships(news_citations),
        'domain_patterns': analyze_domain_patterns(news_citations),
        'model_comparison': analyze_model_comparison(news_citations),
        'political_bias': analyze_political_bias_patterns(news_citations),
        'quality_patterns': analyze_quality_patterns(news_citations),
        'joint_analysis': analyze_joint_bias_quality(news_citations)
    }
    
    # Generate markdown report only
    generate_markdown_report(all_stats, report_output_path)
    logger.info(f"Saved markdown report to {report_output_path}")
    
    # Print summary
    logger.info("=== NEWS CITATION STATISTICS SUMMARY ===")
    logger.info(f"Total news citations analyzed: {len(news_citations):,}")
    logger.info(f"Total threads: {len(all_data['threads']):,}")
    logger.info(f"Total questions: {len(all_data['questions']):,}")
    logger.info(f"Total responses: {len(all_data['responses']):,}")
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