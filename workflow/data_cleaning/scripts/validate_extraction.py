#!/usr/bin/env python3
"""
Data validation script for search arena extraction pipeline.
Validates data quality, referential integrity, and extraction completeness
across all extracted tables.
"""

import pandas as pd
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_extraction():
    """Validate data quality and integrity across all extracted tables."""

    # Get paths from Snakemake
    threads_file = snakemake.input.threads
    questions_file = snakemake.input.questions
    responses_file = snakemake.input.responses
    citations_file = snakemake.input.citations
    output_file = snakemake.output[0]

    logger.info("Loading all extracted tables for validation...")

    # Load all tables
    try:
        threads_df = pd.read_parquet(threads_file)
        questions_df = pd.read_parquet(questions_file)
        responses_df = pd.read_parquet(responses_file)
        citations_df = pd.read_parquet(citations_file)
        
        # Try to load enriched citations if available
        enriched_citations_file = citations_file.replace('citations.parquet', 'citations_enriched.parquet')
        try:
            enriched_citations_df = pd.read_parquet(enriched_citations_file)
            logger.info(f"Loaded enriched citations: {len(enriched_citations_df)} rows")
        except FileNotFoundError:
            enriched_citations_df = None
            logger.info("Enriched citations not found - skipping enrichment validation")
        
        logger.info(f"Loaded threads: {len(threads_df)} rows")
        logger.info(f"Loaded questions: {len(questions_df)} rows")
        logger.info(f"Loaded responses: {len(responses_df)} rows")
        logger.info(f"Loaded citations: {len(citations_df)} rows")
    except Exception as e:
        logger.error(f"Failed to load tables: {e}")
        return

    # Start validation report
    report_lines = []
    report_lines.append("=== DATA EXTRACTION VALIDATION REPORT ===\n")
    report_lines.append(f"Generated: {datetime.now()}\n")
    report_lines.append(f"Validation timestamp: {pd.Timestamp.now()}\n\n")

    # Track validation results
    validation_results = {
        "passed": 0,
        "failed": 0,
        "warnings": 0
    }

    def log_result(test_name, passed, message, warning=False):
        """Log validation result and track counts."""
        if passed:
            status = "✓ PASS"
            validation_results["passed"] += 1
            logger.info(f"{status}: {test_name}")
        elif warning:
            status = "⚠ WARN"
            validation_results["warnings"] += 1
            logger.warning(f"{status}: {test_name} - {message}")
        else:
            status = "✗ FAIL"
            validation_results["failed"] += 1
            logger.error(f"{status}: {test_name} - {message}")
        
        report_lines.append(f"{status}: {test_name}\n")
        if message:
            report_lines.append(f"  {message}\n")
        report_lines.append("\n")

    # ============================================================================
    # 1. BASIC DATA INTEGRITY CHECKS
    # ============================================================================
    logger.info("\n=== BASIC DATA INTEGRITY CHECKS ===")
    report_lines.append("=== BASIC DATA INTEGRITY CHECKS ===\n\n")

    # Check for empty tables
    log_result("Threads table not empty", len(threads_df) > 0, 
               f"Expected >0, got {len(threads_df)}")
    log_result("Questions table not empty", len(questions_df) > 0, 
               f"Expected >0, got {len(questions_df)}")
    log_result("Responses table not empty", len(responses_df) > 0, 
               f"Expected >0, got {len(responses_df)}")
    log_result("Citations table not empty", len(citations_df) > 0, 
               f"Expected >0, got {len(citations_df)}")

    # Check expected row counts
    expected_threads = 24069
    log_result("Threads count matches expected", len(threads_df) == expected_threads,
               f"Expected {expected_threads}, got {len(threads_df)}")

    # Responses should be 2x questions
    expected_responses = len(questions_df) * 2
    log_result("Responses count is 2x questions", len(responses_df) == expected_responses,
               f"Expected {expected_responses} (2x{len(questions_df)}), got {len(responses_df)}")

    # ============================================================================
    # 2. UNIQUE KEY VALIDATION
    # ============================================================================
    logger.info("\n=== UNIQUE KEY VALIDATION ===")
    report_lines.append("=== UNIQUE KEY VALIDATION ===\n\n")

    # Check primary key uniqueness
    log_result("Thread IDs are unique", threads_df['thread_id'].nunique() == len(threads_df),
               f"Expected {len(threads_df)} unique, got {threads_df['thread_id'].nunique()}")
    
    log_result("Question IDs are unique", questions_df['question_id'].nunique() == len(questions_df),
               f"Expected {len(questions_df)} unique, got {questions_df['question_id'].nunique()}")
    
    log_result("Response IDs are unique", responses_df['response_id'].nunique() == len(responses_df),
               f"Expected {len(responses_df)} unique, got {responses_df['response_id'].nunique()}")
    
    log_result("Citation IDs are unique", citations_df['citation_id'].nunique() == len(citations_df),
               f"Expected {len(citations_df)} unique, got {citations_df['citation_id'].nunique()}")

    # ============================================================================
    # 3. REFERENTIAL INTEGRITY CHECKS
    # ============================================================================
    logger.info("\n=== REFERENTIAL INTEGRITY CHECKS ===")
    report_lines.append("=== REFERENTIAL INTEGRITY CHECKS ===\n\n")

    # Questions -> Threads foreign key validation
    valid_thread_ids = set(threads_df['thread_id'])
    question_thread_ids = set(questions_df['thread_id'])
    invalid_question_threads = question_thread_ids - valid_thread_ids
    log_result("All question thread_ids reference valid threads", 
               len(invalid_question_threads) == 0,
               f"Found {len(invalid_question_threads)} invalid thread references")

    # Responses -> Questions foreign key validation
    valid_question_ids = set(questions_df['question_id'])
    response_question_ids = set(responses_df['question_id'])
    invalid_response_questions = response_question_ids - valid_question_ids
    log_result("All response question_ids reference valid questions",
               len(invalid_response_questions) == 0,
               f"Found {len(invalid_response_questions)} invalid question references")

    # Citations -> Responses foreign key validation
    valid_response_ids = set(responses_df['response_id'])
    citation_response_ids = set(citations_df['response_id'])
    invalid_citation_responses = citation_response_ids - valid_response_ids
    log_result("All citation response_ids reference valid responses",
               len(invalid_citation_responses) == 0,
               f"Found {len(invalid_citation_responses)} invalid response references")

    # ============================================================================
    # 4. DATA CONSISTENCY CHECKS
    # ============================================================================
    logger.info("\n=== DATA CONSISTENCY CHECKS ===")
    report_lines.append("=== DATA CONSISTENCY CHECKS ===\n\n")

    # Check model side distribution in responses
    model_side_counts = responses_df['model_side'].value_counts()
    side_a_count = model_side_counts.get('a', 0)
    side_b_count = model_side_counts.get('b', 0)
    
    log_result("Model sides are balanced", abs(side_a_count - side_b_count) <= 10,
               f"Side A: {side_a_count}, Side B: {side_b_count}, difference: {abs(side_a_count - side_b_count)}")

    # Check turn number consistency
    questions_by_thread = questions_df.groupby('thread_id')['turn_number'].max()
    threads_total_turns = threads_df.set_index('thread_id')['total_turns']
    
    # Compare max turn numbers with total_turns
    turn_mismatches = 0
    for thread_id in questions_by_thread.index:
        if thread_id in threads_total_turns.index:
            max_turn = questions_by_thread[thread_id]
            total_turns = threads_total_turns[thread_id]
            if max_turn != total_turns:
                turn_mismatches += 1
    
    log_result("Turn numbers consistent with total_turns", turn_mismatches == 0,
               f"Found {turn_mismatches} threads with inconsistent turn counts", 
               warning=turn_mismatches < len(threads_df) * 0.1)  # Warning if <10%

    # ============================================================================
    # 5. DATA COMPLETENESS CHECKS
    # ============================================================================
    logger.info("\n=== DATA COMPLETENESS CHECKS ===")
    report_lines.append("=== DATA COMPLETENESS CHECKS ===\n\n")

    # Check for missing critical fields
    required_fields = {
        'threads': ['thread_id', 'timestamp', 'total_turns'],
        'questions': ['question_id', 'thread_id', 'turn_number', 'user_query'],
        'responses': ['response_id', 'question_id', 'model_name_llm', 'model_name_raw', 'response_text'],
        'citations': ['citation_id', 'response_id', 'url', 'domain_full', 'domain']
    }

    tables = {
        'threads': threads_df,
        'questions': questions_df,
        'responses': responses_df,
        'citations': citations_df
    }

    for table_name, required_cols in required_fields.items():
        df = tables[table_name]
        for col in required_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                log_result(f"{table_name}.{col} has no nulls", null_count == 0,
                          f"Found {null_count} null values ({null_count/len(df)*100:.1f}%)",
                          warning=null_count < len(df) * 0.05)  # Warning if <5%
            else:
                log_result(f"{table_name}.{col} column exists", False,
                          f"Required column '{col}' missing from {table_name} table")

    # ============================================================================
    # 6. CITATION-SPECIFIC VALIDATIONS
    # ============================================================================
    logger.info("\n=== CITATION-SPECIFIC VALIDATIONS ===")
    report_lines.append("=== CITATION-SPECIFIC VALIDATIONS ===\n\n")

    # URL validation
    if 'url_valid' in citations_df.columns:
        valid_url_count = citations_df['url_valid'].sum()
        total_citations = len(citations_df)
        valid_url_pct = valid_url_count / total_citations * 100
        
        log_result("High URL validity rate", valid_url_pct >= 95.0,
                  f"Valid URLs: {valid_url_count}/{total_citations} ({valid_url_pct:.1f}%)",
                  warning=valid_url_pct >= 90.0)

    # Domain extraction completeness
    if 'domain' in citations_df.columns:
        domain_null_count = citations_df['domain'].isnull().sum()
        domain_completeness = (len(citations_df) - domain_null_count) / len(citations_df) * 100
        
        log_result("High domain extraction completeness", domain_completeness >= 95.0,
                  f"Domains extracted: {len(citations_df) - domain_null_count}/{len(citations_df)} ({domain_completeness:.1f}%)",
                  warning=domain_completeness >= 90.0)

    # Citation number distribution
    if 'citation_number' in citations_df.columns:
        citation_nums = citations_df['citation_number'].dropna()
        if len(citation_nums) > 0:
            max_citation_num = citation_nums.max()
            log_result("Reasonable citation numbers", max_citation_num <= 50,
                      f"Max citation number: {max_citation_num}",
                      warning=max_citation_num <= 100)

    # ============================================================================
    # 7. STATISTICAL SUMMARY
    # ============================================================================
    logger.info("\n=== STATISTICAL SUMMARY ===")
    report_lines.append("=== STATISTICAL SUMMARY ===\n\n")

    # Calculate key statistics
    avg_questions_per_thread = len(questions_df) / len(threads_df)
    avg_responses_per_question = len(responses_df) / len(questions_df)
    avg_citations_per_response = len(citations_df) / len(responses_df)

    stats = [
        f"Average questions per thread: {avg_questions_per_thread:.2f}",
        f"Average responses per question: {avg_responses_per_question:.2f}",
        f"Average citations per response: {avg_citations_per_response:.2f}",
        f"Total threads: {len(threads_df):,}",
        f"Total questions: {len(questions_df):,}",
        f"Total responses: {len(responses_df):,}",
        f"Total citations: {len(citations_df):,}"
    ]

    for stat in stats:
        logger.info(stat)
        report_lines.append(f"{stat}\n")

    # ============================================================================
    # 8. ENRICHED CITATIONS VALIDATION (if available)
    # ============================================================================
    if enriched_citations_df is not None:
        logger.info("\n=== ENRICHED CITATIONS VALIDATION ===")
        report_lines.append("\n=== ENRICHED CITATIONS VALIDATION ===\n")
        
        # Check row count consistency
        enriched_count = len(enriched_citations_df)
        original_count = len(citations_df)
        
        log_result("Enriched citations row count matches original", 
                  enriched_count == original_count,
                  f"Enriched: {enriched_count}, Original: {original_count}")
        
        # Check for political leaning data
        if 'political_leaning_score' in enriched_citations_df.columns:
            pol_coverage = enriched_citations_df['political_leaning_score'].notna().sum()
            pol_pct = pol_coverage / enriched_count * 100
            log_result("Political leaning data coverage", pol_pct >= 30.0,
                      f"Political leaning available: {pol_coverage}/{enriched_count} ({pol_pct:.1f}%)",
                      warning=pol_pct >= 20.0)
            
            # Check categorical leaning variable
            if 'political_leaning' in enriched_citations_df.columns:
                leaning_counts = enriched_citations_df['political_leaning'].value_counts()
                left_count = leaning_counts.get('left_leaning', 0)
                right_count = leaning_counts.get('right_leaning', 0)
                unknown_count = leaning_counts.get('unknown_leaning', 0)
                log_result("Categorical leaning variable created", 'political_leaning' in enriched_citations_df.columns,
                          f"Left: {left_count:,}, Right: {right_count:,}, Unknown: {unknown_count:,}")
        
        # Check for domain quality data
        if 'domain_quality_score' in enriched_citations_df.columns:
            qual_coverage = enriched_citations_df['domain_quality_score'].notna().sum()
            qual_pct = qual_coverage / enriched_count * 100
            log_result("Domain quality data coverage", qual_pct >= 20.0,
                      f"Domain quality available: {qual_coverage}/{enriched_count} ({qual_pct:.1f}%)",
                      warning=qual_pct >= 15.0)
            
            # Check categorical quality variable
            if 'domain_quality' in enriched_citations_df.columns:
                quality_counts = enriched_citations_df['domain_quality'].value_counts()
                high_count = quality_counts.get('high_quality', 0)
                low_count = quality_counts.get('low_quality', 0)
                unknown_count = quality_counts.get('unknown_quality', 0)
                log_result("Categorical quality variable created", 'domain_quality' in enriched_citations_df.columns,
                          f"High: {high_count:,}, Low: {low_count:,}, Unknown: {unknown_count:,}")
        
        # Check for combined coverage
        if 'political_leaning_score' in enriched_citations_df.columns and 'domain_quality_score' in enriched_citations_df.columns:
            combined_coverage = ((enriched_citations_df['political_leaning_score'].notna()) & 
                               (enriched_citations_df['domain_quality_score'].notna())).sum()
            combined_pct = combined_coverage / enriched_count * 100
            log_result("Combined metrics coverage", combined_pct >= 15.0,
                      f"Both metrics available: {combined_coverage}/{enriched_count} ({combined_pct:.1f}%)",
                      warning=combined_pct >= 10.0)
        
        # Enrichment summary stats
        enrichment_stats = [
            f"Enriched citations: {enriched_count:,} rows, {len(enriched_citations_df.columns)} columns"
        ]
        
        if 'political_leaning_score' in enriched_citations_df.columns:
            pol_stats = enriched_citations_df['political_leaning_score'].describe()
            enrichment_stats.append(f"Political leaning range: {pol_stats['min']:.3f} to {pol_stats['max']:.3f}")
        
        if 'domain_quality_score' in enriched_citations_df.columns:
            qual_stats = enriched_citations_df['domain_quality_score'].describe()
            enrichment_stats.append(f"Domain quality range: {qual_stats['min']:.3f} to {qual_stats['max']:.3f}")
        
        for stat in enrichment_stats:
            logger.info(stat)
            report_lines.append(f"{stat}\n")

    # ============================================================================
    # 9. VALIDATION SUMMARY
    # ============================================================================
    logger.info("\n=== VALIDATION SUMMARY ===")
    report_lines.append("\n=== VALIDATION SUMMARY ===\n")

    total_tests = validation_results["passed"] + validation_results["failed"] + validation_results["warnings"]
    
    summary = [
        f"Total tests run: {total_tests}",
        f"Tests passed: {validation_results['passed']} ({validation_results['passed']/total_tests*100:.1f}%)",
        f"Tests failed: {validation_results['failed']} ({validation_results['failed']/total_tests*100:.1f}%)",
        f"Warnings: {validation_results['warnings']} ({validation_results['warnings']/total_tests*100:.1f}%)"
    ]

    for line in summary:
        logger.info(line)
        report_lines.append(f"{line}\n")

    # Overall assessment
    if validation_results["failed"] == 0:
        if validation_results["warnings"] == 0:
            overall_status = "✓ EXCELLENT - All validations passed"
            logger.info(overall_status)
        else:
            overall_status = f"⚠ GOOD - All critical validations passed with {validation_results['warnings']} warnings"
            logger.warning(overall_status)
    else:
        overall_status = f"✗ ISSUES FOUND - {validation_results['failed']} validations failed"
        logger.error(overall_status)

    report_lines.append(f"\nOverall Status: {overall_status}\n")

    # Write validation report
    logger.info(f"Writing validation report to {output_file}")
    try:
        with open(output_file, 'w') as f:
            f.writelines(report_lines)
        logger.info("✓ Validation report saved successfully")
    except Exception as e:
        logger.error(f"✗ Failed to save validation report: {e}")
        return

    # Final status
    if validation_results["failed"] > 0:
        logger.error("VALIDATION FAILED - Critical issues found in data extraction")
        return False
    else:
        logger.info("VALIDATION COMPLETED - Data extraction quality verified")
        return True


if __name__ == "__main__":
    validate_extraction()