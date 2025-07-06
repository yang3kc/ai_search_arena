#!/usr/bin/env python3
"""
Generate comprehensive data summary report for research paper.

This script analyzes the extracted and enriched data to produce statistics
that can be included in academic papers, including dataset size, coverage
metrics, and data quality indicators.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data_tables(input_files):
    """Load all the main data tables from Snakemake input files."""
    logger.info("Loading data tables...")

    tables = {}
    table_mapping = {
        "threads": input_files.threads,
        "questions": input_files.questions,
        "responses": input_files.responses,
        "citations": input_files.citations,
        "citations_enriched": input_files.citations_enriched,
        "domains": input_files.domains,
        "domains_enriched": input_files.domains_enriched,
    }

    for table_name, filepath in table_mapping.items():
        if Path(filepath).exists():
            tables[table_name] = pd.read_parquet(filepath)
            logger.info(f"Loaded {table_name}: {len(tables[table_name]):,} rows")
        else:
            logger.warning(f"File not found: {filepath}")
            tables[table_name] = None

    return tables


def analyze_dataset_overview(tables):
    """Generate high-level dataset statistics."""
    stats = {}

    if tables["threads"] is not None:
        threads = tables["threads"]
        stats["total_conversations"] = len(threads)
        stats["total_conversations_with_judgement"] = len(
            threads[threads["winner"].notna()]
        )

        # Date range analysis
        timestamp_col = "timestamp" if "timestamp" in threads.columns else "tstamp"
        if timestamp_col in threads.columns:
            # Convert timestamps to datetime
            threads_with_dates = threads.copy()
            threads_with_dates["datetime"] = pd.to_datetime(
                threads_with_dates[timestamp_col]
            )

            earliest_date = threads_with_dates["datetime"].min()
            latest_date = threads_with_dates["datetime"].max()
            span_days = (latest_date - earliest_date).days

            # Calculate additional temporal statistics
            threads_per_day = threads_with_dates.groupby(
                threads_with_dates["datetime"].dt.date
            ).size()
            threads_per_month = threads_with_dates.groupby(
                threads_with_dates["datetime"].dt.to_period("M")
            ).size()
            threads_per_week = threads_with_dates.groupby(
                threads_with_dates["datetime"].dt.to_period("W")
            ).size()

            stats["date_range"] = {
                "earliest": earliest_date.strftime("%Y-%m-%d %H:%M:%S"),
                "latest": latest_date.strftime("%Y-%m-%d %H:%M:%S"),
                "earliest_date_only": earliest_date.strftime("%Y-%m-%d"),
                "latest_date_only": latest_date.strftime("%Y-%m-%d"),
                "span_days": span_days,
                "span_weeks": span_days // 7,
                "span_months": round(span_days / 30.4, 1),
                "avg_conversations_per_day": float(threads_per_day.mean()),
                "max_conversations_per_day": int(threads_per_day.max()),
                "min_conversations_per_day": int(threads_per_day.min()),
                "avg_conversations_per_week": float(threads_per_week.mean()),
                "avg_conversations_per_month": float(threads_per_month.mean()),
                "total_active_days": len(threads_per_day),
            }

    if tables["questions"] is not None:
        stats["total_questions"] = len(tables["questions"])

    if tables["responses"] is not None:
        responses = tables["responses"]
        stats["total_responses"] = len(responses)

        # Model analysis
        model_col = (
            "model_name_raw" if "model_name_raw" in responses.columns else "model"
        )
        if model_col in responses.columns:
            model_counts = responses[model_col].value_counts()
            stats["models"] = {
                "unique_models": len(model_counts),
                "model_distribution": model_counts.to_dict(),
            }

    if tables["questions"] is not None and tables["responses"] is not None:
        stats["responses_per_question"] = len(tables["responses"]) / len(
            tables["questions"]
        )

    if tables["citations"] is not None:
        citations = tables["citations"]
        stats["total_citations"] = len(citations)

        # Citation per response stats
        cites_per_response = citations.groupby("response_id").size()
        stats["citations_per_response"] = {
            "mean": float(cites_per_response.mean()),
            "median": float(cites_per_response.median()),
            "max": int(cites_per_response.max()),
            "responses_with_citations": len(cites_per_response),
        }

    return stats


def analyze_domain_coverage(tables):
    """Analyze domain and enrichment coverage."""
    stats = {}

    if tables["domains"] is not None:
        domains = tables["domains"]
        stats["unique_domains"] = len(domains)

    if tables["domains_enriched"] is not None:
        enriched = tables["domains_enriched"]

        # Political leaning coverage
        if "political_leaning_score" in enriched.columns:
            political_coverage = enriched["political_leaning_score"].notna().sum()
            stats["political_leaning_coverage"] = {
                "domains_with_scores": int(political_coverage),
                "coverage_rate": float(political_coverage / len(enriched)),
            }

        # Quality ratings coverage
        if "domain_quality_score" in enriched.columns:
            quality_coverage = enriched["domain_quality_score"].notna().sum()
            stats["quality_ratings_coverage"] = {
                "domains_with_ratings": int(quality_coverage),
                "coverage_rate": float(quality_coverage / len(enriched)),
            }

        # Domain classification coverage
        if "domain_classification" in enriched.columns:
            classification_counts = enriched["domain_classification"].value_counts()
            stats["domain_classification"] = {
                "classification_counts": classification_counts.to_dict(),
                "total_classified": int(
                    (enriched["domain_classification"] != "unclassified").sum()
                ),
                "classification_rate": float(
                    (enriched["domain_classification"] != "unclassified").sum()
                    / len(enriched)
                ),
            }

    # Add enriched citation coverage analysis
    if tables["citations_enriched"] is not None:
        citations_enriched = tables["citations_enriched"]

        # Political leaning coverage in actual citations
        if "political_leaning_score" in citations_enriched.columns:
            political_coverage_cites = (
                citations_enriched["political_leaning_score"].notna().sum()
            )
            stats["citation_political_leaning_coverage"] = {
                "citations_with_scores": int(political_coverage_cites),
                "coverage_rate": float(
                    political_coverage_cites / len(citations_enriched)
                ),
            }

        # Quality ratings coverage in actual citations
        if "domain_quality_score" in citations_enriched.columns:
            quality_coverage_cites = (
                citations_enriched["domain_quality_score"].notna().sum()
            )
            stats["citation_quality_ratings_coverage"] = {
                "citations_with_ratings": int(quality_coverage_cites),
                "coverage_rate": float(
                    quality_coverage_cites / len(citations_enriched)
                ),
            }

        # Domain classification coverage in actual citations
        if "domain_classification" in citations_enriched.columns:
            citation_classification_counts = citations_enriched[
                "domain_classification"
            ].value_counts()
            stats["citation_domain_classification"] = {
                "classification_counts": citation_classification_counts.to_dict(),
                "total_classified": int(
                    (
                        citations_enriched["domain_classification"] != "unclassified"
                    ).sum()
                ),
                "classification_rate": float(
                    (
                        citations_enriched["domain_classification"] != "unclassified"
                    ).sum()
                    / len(citations_enriched)
                ),
            }

    return stats


def analyze_citation_patterns(tables):
    """Analyze citation and source patterns."""
    stats = {}

    if tables["citations_enriched"] is not None:
        citations = tables["citations_enriched"]

        # URL validity
        if "url" in citations.columns:
            valid_urls = citations["url"].notna().sum()
            stats["url_quality"] = {
                "valid_urls": int(valid_urls),
                "validity_rate": float(valid_urls / len(citations)),
            }

    return stats


def analyze_model_comparison(tables):
    """Analyze model-specific patterns."""
    stats = {}

    if tables["responses"] is not None and tables["citations_enriched"] is not None:
        responses = tables["responses"]
        citations = tables["citations_enriched"]

        # Merge to get model info for each citation
        model_col = (
            "model_name_llm" if "model_name_llm" in responses.columns else "model"
        )
        model_citations = citations.merge(
            responses[["response_id", model_col]], on="response_id", how="left"
        )

        # Citations per model
        model_cite_counts = model_citations.groupby(model_col).size()
        stats["citations_by_model"] = model_cite_counts.to_dict()

        # News citation rates by model
        if "domain_classification" in model_citations.columns:
            model_citations["is_news"] = (
                model_citations["domain_classification"] == "news"
            )
            news_by_model = model_citations.groupby(model_col)["is_news"].agg(
                ["sum", "count", "mean"]
            )
            stats["news_citation_rates_by_model"] = {
                model: {
                    "news_citations": int(row["sum"]),
                    "total_citations": int(row["count"]),
                    "news_rate": float(row["mean"]),
                }
                for model, row in news_by_model.iterrows()
            }

    return stats


def generate_summary_report(output_path, input_files):
    """Generate comprehensive data summary report."""
    logger.info("Starting data summary generation...")

    # Load all data
    tables = load_data_tables(input_files)

    # Generate all analyses
    summary = {
        "generation_timestamp": datetime.now().isoformat(),
        "dataset_overview": analyze_dataset_overview(tables),
        "domain_coverage": analyze_domain_coverage(tables),
        "citation_patterns": analyze_citation_patterns(tables),
        "model_comparison": analyze_model_comparison(tables),
    }

    # Generate markdown report only
    markdown_path = Path(output_path) / "data_summary_report.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    generate_markdown_report(summary, markdown_path)

    logger.info(f"Markdown report saved to: {markdown_path}")

    return summary


def generate_markdown_report(summary, output_path):
    """Generate human-readable markdown report."""

    report_lines = [
        "# AI Search Arena Dataset Summary Report",
        "",
        f"*Generated on: {summary['generation_timestamp']}*",
        "",
        "## Dataset Overview",
        "",
    ]

    overview = summary["dataset_overview"]
    if "total_conversations" in overview:
        report_lines.extend(
            [
                f"- **Total Conversations**: {overview['total_conversations']:,}",
                f"- **Total Conversations with Judgement**: {overview['total_conversations_with_judgement']:,}",
                f"- **Total Questions**: {overview.get('total_questions', 'N/A'):,}",
                f"- **Total Responses**: {overview.get('total_responses', 'N/A'):,}",
                f"- **Responses per Question**: {overview.get('responses_per_question', 'N/A'):.2f}",
                f"- **Total Citations**: {overview.get('total_citations', 'N/A'):,}",
                "",
            ]
        )

    if "date_range" in overview:
        date_info = overview["date_range"]
        report_lines.extend(
            [
                "### Temporal Coverage",
                f"- **Date Range**: {date_info['earliest_date_only']} to {date_info['latest_date_only']}",
                f"- **Full Timestamp Range**: {date_info['earliest']} to {date_info['latest']}",
                f"- **Collection Period**: {date_info['span_days']} days ({date_info['span_weeks']} weeks, {date_info['span_months']} months)",
                f"- **Active Days**: {date_info['total_active_days']} out of {date_info['span_days']} total days",
                "",
                "**Activity Statistics:**",
                f"- **Average Conversations per Day**: {date_info['avg_conversations_per_day']:.1f}",
                f"- **Peak Daily Activity**: {date_info['max_conversations_per_day']} conversations",
                f"- **Minimum Daily Activity**: {date_info['min_conversations_per_day']} conversations",
                f"- **Average Conversations per Week**: {date_info['avg_conversations_per_week']:.0f}",
                f"- **Average Conversations per Month**: {date_info['avg_conversations_per_month']:.0f}",
                "",
            ]
        )

    if "models" in overview:
        model_info = overview["models"]
        report_lines.extend(
            [
                "### Model Distribution",
                f"- **Unique Models**: {model_info['unique_models']}",
                "",
            ]
        )
        for model, count in sorted(
            model_info["model_distribution"].items(), key=lambda x: x[1], reverse=True
        ):
            report_lines.append(f"  - {model}: {count:,} responses")
        report_lines.append("")

    # Citations analysis
    if "citations_per_response" in overview:
        cite_stats = overview["citations_per_response"]
        report_lines.extend(
            [
                "### Citation Statistics",
                f"- **Average Citations per Response**: {cite_stats['mean']:.2f}",
                f"- **Median Citations per Response**: {cite_stats['median']:.1f}",
                f"- **Maximum Citations per Response**: {cite_stats['max']}",
                f"- **Responses with Citations**: {cite_stats['responses_with_citations']:,}",
                "",
            ]
        )

    # Domain coverage
    domain_coverage = summary["domain_coverage"]
    if domain_coverage:
        report_lines.extend(
            [
                "## Domain Coverage",
                f"- **Unique Domains**: {domain_coverage.get('unique_domains', 'N/A'):,}"
                if domain_coverage.get("unique_domains") is not None
                else "- **Unique Domains**: N/A",
                "",
            ]
        )

        if "political_leaning_coverage" in domain_coverage:
            pol_cov = domain_coverage["political_leaning_coverage"]
            report_lines.extend(
                [
                    "### Political Leaning Data",
                    f"- **Domains with Political Scores**: {pol_cov['domains_with_scores']:,} ({pol_cov['coverage_rate']:.1%})",
                    "",
                ]
            )

        if "quality_ratings_coverage" in domain_coverage:
            qual_cov = domain_coverage["quality_ratings_coverage"]
            report_lines.extend(
                [
                    "### Quality Ratings Data",
                    f"- **Domains with Quality Ratings**: {qual_cov['domains_with_ratings']:,} ({qual_cov['coverage_rate']:.1%})",
                    "",
                ]
            )

        if "domain_classification" in domain_coverage:
            class_info = domain_coverage["domain_classification"]
            report_lines.extend(
                [
                    "### Domain Classification",
                    f"- **Total Classified Domains**: {class_info['total_classified']:,} ({class_info['classification_rate']:.1%})",
                    "",
                    "**Domain Types:**",
                ]
            )
            total_domains = sum(class_info["classification_counts"].values())
            for domain_type, count in sorted(
                class_info["classification_counts"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                if (
                    domain_type != "unclassified"
                ):  # Skip unclassified in detailed breakdown
                    percentage = (count / total_domains) * 100
                    report_lines.append(
                        f"  - {domain_type}: {count:,} domains ({percentage:.1f}%)"
                    )
            report_lines.append("")

        # Add citation-level coverage statistics
        if "citation_political_leaning_coverage" in domain_coverage:
            cite_pol_cov = domain_coverage["citation_political_leaning_coverage"]
            report_lines.extend(
                [
                    "### Citation-Level Coverage",
                    f"- **Citations with Political Scores**: {cite_pol_cov['citations_with_scores']:,} ({cite_pol_cov['coverage_rate']:.1%})",
                ]
            )

        if "citation_quality_ratings_coverage" in domain_coverage:
            cite_qual_cov = domain_coverage["citation_quality_ratings_coverage"]
            report_lines.append(
                f"- **Citations with Quality Ratings**: {cite_qual_cov['citations_with_ratings']:,} ({cite_qual_cov['coverage_rate']:.1%})"
            )

        if "citation_domain_classification" in domain_coverage:
            cite_class_info = domain_coverage["citation_domain_classification"]
            report_lines.extend(
                [
                    f"- **Classified Citations**: {cite_class_info['total_classified']:,} ({cite_class_info['classification_rate']:.1%})",
                    "",
                    "**Citation Types:**",
                ]
            )
            total_citations = sum(cite_class_info["classification_counts"].values())
            for domain_type, count in sorted(
                cite_class_info["classification_counts"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                if (
                    domain_type != "unclassified"
                ):  # Skip unclassified in detailed breakdown
                    percentage = (count / total_citations) * 100
                    report_lines.append(
                        f"  - {domain_type}: {count:,} citations ({percentage:.1f}%)"
                    )

    # Model comparison
    model_comparison = summary["model_comparison"]
    if model_comparison and "citations_by_model" in model_comparison:
        report_lines.extend(["## Model Comparison", "", "### Citations by Model", ""])

        for model, count in sorted(
            model_comparison["citations_by_model"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            report_lines.append(f"- {model}: {count:,} citations")
        report_lines.append("")

        if "news_citation_rates_by_model" in model_comparison:
            report_lines.extend(["### News Citation Rates by Model", ""])
            news_rates = model_comparison["news_citation_rates_by_model"]
            for model, stats in sorted(
                news_rates.items(), key=lambda x: x[1]["news_rate"], reverse=True
            ):
                report_lines.append(
                    f"- {model}: {stats['news_citations']:,}/{stats['total_citations']:,} ({stats['news_rate']:.1%})"
                )
            report_lines.append("")

    # Write the report
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))


def main():
    """Main function for Snakemake execution."""

    # Get paths from snakemake object if available, otherwise use defaults
    try:
        # When run by Snakemake
        output_path = Path(snakemake.output[0]).parent
        input_files = snakemake.input
    except NameError:
        # For standalone execution
        class MockInput:
            def __init__(self):
                base_dir = "../../data/intermediate/cleaned_arena_data"
                self.threads = f"{base_dir}/threads.parquet"
                self.questions = f"{base_dir}/questions.parquet"
                self.responses = f"{base_dir}/responses.parquet"
                self.citations = f"{base_dir}/citations.parquet"
                self.citations_enriched = f"{base_dir}/citations_enriched.parquet"
                self.domains = f"{base_dir}/domains.parquet"
                self.domains_enriched = f"{base_dir}/domains_enriched.parquet"

        output_path = Path("../../data/output")
        input_files = MockInput()

    # Generate the summary report
    summary = generate_summary_report(output_path, input_files)

    logger.info("Data summary report generation completed successfully!")

    # Print key statistics for immediate feedback
    overview = summary["dataset_overview"]
    logger.info("=== KEY STATISTICS ===")
    logger.info(f"Total Conversations: {overview.get('total_conversations', 'N/A'):,}")
    logger.info(f"Total Citations: {overview.get('total_citations', 'N/A'):,}")

    domain_coverage = summary["domain_coverage"]
    logger.info(f"Unique Domains: {domain_coverage.get('unique_domains', 'N/A'):,}")

    if "political_leaning_coverage" in domain_coverage:
        pol_cov = domain_coverage["political_leaning_coverage"]
        logger.info(f"Political Coverage: {pol_cov['coverage_rate']:.1%}")


if __name__ == "__main__":
    main()
