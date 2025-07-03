#!/usr/bin/env python3
"""
Citations extraction script for search arena data.
Extracts web search citations from system metadata web_search_trace and creates the citations table
according to the defined schema.
"""

import pandas as pd
import logging
import re
from urllib.parse import urlparse
import tldextract

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_domain_full(url):
    """Extract full domain from URL with error handling (keeps subdomains)."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix if present
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return None


def extract_domain(url):
    """Extract base domain from URL using tldextract (without subdomains)."""
    try:
        extracted = tldextract.extract(url)
        # Combine domain and suffix (e.g., 'example' + 'co.uk' = 'example.co.uk')
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
        elif extracted.domain:
            return extracted.domain.lower()
        return None
    except Exception:
        return None


def is_valid_url(url):
    """Check if URL is well-formed."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def extract_citation_number(citation_ref):
    """Extract citation number from reference like '[1]', '[2]', etc."""
    try:
        match = re.search(r"\[(\d+)\]", str(citation_ref))
        if match:
            return int(match.group(1))
        return None
    except Exception:
        return None


def extract_citations():
    """Extract citations from search arena data web search traces."""

    # Get paths from Snakemake
    input_data = snakemake.input.data
    responses_file = snakemake.input.responses
    output_file = snakemake.output[0]
    config = snakemake.config

    logger.info(f"Loading data from {input_data}")
    logger.info(f"Loading responses table from {responses_file}")

    # Load the raw data and responses table
    try:
        df = pd.read_parquet(input_data)
        responses_df = pd.read_parquet(responses_file)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        logger.info(f"Loaded {len(responses_df)} response records")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    logger.info("Extracting citations from web search traces...")

    # Create citations table according to schema
    citations_data = []
    total_citations = 0
    missing_trace_count = 0
    invalid_url_count = 0
    citation_parsing_errors = 0

    for idx, row in df.iterrows():
        # Get thread_id and corresponding responses for this thread
        thread_id = f"{config['thread_id_prefix']}{idx:08d}"
        thread_responses = responses_df[responses_df["thread_id"] == thread_id]

        if len(thread_responses) == 0:
            logger.warning(f"No responses found for thread {thread_id}, skipping")
            continue

        # Extract system metadata for both models
        system_a_metadata = row.get("system_a_metadata", {})
        system_b_metadata = row.get("system_b_metadata", {})

        if not isinstance(system_a_metadata, dict) or not isinstance(
            system_b_metadata, dict
        ):
            missing_trace_count += 1
            continue

        # Get web search traces for both models
        web_search_trace_a = system_a_metadata.get("web_search_trace", [])
        web_search_trace_b = system_b_metadata.get("web_search_trace", [])

        # Process citations for model A responses
        model_a_responses = thread_responses[thread_responses["model_side"] == "a"]
        if len(web_search_trace_a) > 0 and len(model_a_responses) > 0:
            citations_extracted_a = process_web_search_trace(
                web_search_trace_a, model_a_responses, config, total_citations
            )
            citations_data.extend(citations_extracted_a["citations"])
            total_citations = citations_extracted_a["total_count"]
            invalid_url_count += citations_extracted_a["invalid_urls"]
            citation_parsing_errors += citations_extracted_a["parsing_errors"]

        # Process citations for model B responses
        model_b_responses = thread_responses[thread_responses["model_side"] == "b"]
        if len(web_search_trace_b) > 0 and len(model_b_responses) > 0:
            citations_extracted_b = process_web_search_trace(
                web_search_trace_b, model_b_responses, config, total_citations
            )
            citations_data.extend(citations_extracted_b["citations"])
            total_citations = citations_extracted_b["total_count"]
            invalid_url_count += citations_extracted_b["invalid_urls"]
            citation_parsing_errors += citations_extracted_b["parsing_errors"]

        # Log progress every 1000 rows
        if (idx + 1) % 1000 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} threads")

    # Create DataFrame
    citations_df = pd.DataFrame(citations_data)

    # Data validation
    logger.info("Validating citations data...")

    if len(citations_df) == 0:
        logger.warning("No citations extracted!")
        # Create empty DataFrame with correct schema
        citations_df = pd.DataFrame(
            columns=[
                "citation_id",
                "response_id",
                "citation_number",
                "url",
                "domain_full",
                "domain",
                "url_valid",
                "citation_order",
            ]
        )
    else:
        # Check for required fields
        required_fields = ["citation_id", "response_id", "url"]
        missing_required = []
        for field in required_fields:
            if citations_df[field].isna().any():
                missing_count = citations_df[field].isna().sum()
                missing_required.append(f"{field}: {missing_count} missing")

        if missing_required:
            logger.warning(f"Missing required fields: {', '.join(missing_required)}")

        # Validate citation_id uniqueness
        if citations_df["citation_id"].nunique() != len(citations_df):
            logger.error("Citation IDs are not unique!")
            return

        # Validate response_id references
        valid_response_ids = set(responses_df["response_id"])
        invalid_response_ids = set(citations_df["response_id"]) - valid_response_ids
        if invalid_response_ids:
            logger.error(
                f"Invalid response_id references found: {len(invalid_response_ids)}"
            )
            return

    # Log statistics
    logger.info("\n=== CITATION EXTRACTION STATISTICS ===")
    logger.info(f"Total citations extracted: {len(citations_df)}")
    if len(citations_df) > 0:
        logger.info(
            f"Total responses with citations: {citations_df['response_id'].nunique()}"
        )
        logger.info(
            f"Average citations per response: {len(citations_df) / citations_df['response_id'].nunique():.2f}"
        )
        logger.info(
            f"Missing web search traces: {missing_trace_count}/{len(df)} ({missing_trace_count / len(df) * 100:.1f}%)"
        )
        logger.info(
            f"Invalid URLs found: {invalid_url_count}/{len(citations_df)} ({invalid_url_count / len(citations_df) * 100:.1f}%)"
        )
        logger.info(f"Citation parsing errors: {citation_parsing_errors}")

        # Citation number distribution
        citation_num_dist = citations_df["citation_number"].value_counts().sort_index()
        logger.info("Citation number distribution (top 10):")
        for num, count in citation_num_dist.head(10).items():
            logger.info(
                f"  [{num}]: {count} citations ({count / len(citations_df) * 100:.1f}%)"
            )

        # Top domains (full with subdomains)
        domain_full_counts = citations_df["domain_full"].value_counts()
        logger.info("Top cited domains (full with subdomains) (top 10):")
        for domain, count in domain_full_counts.head(10).items():
            if domain:  # Skip None domains
                logger.info(
                    f"  {domain}: {count} citations ({count / len(citations_df) * 100:.1f}%)"
                )

        # Top domains (base without subdomains)
        domain_counts = citations_df["domain"].value_counts()
        logger.info("Top cited domains (base without subdomains) (top 10):")
        for domain, count in domain_counts.head(10).items():
            if domain:  # Skip None domains
                logger.info(
                    f"  {domain}: {count} citations ({count / len(citations_df) * 100:.1f}%)"
                )

        # URL validity
        valid_urls = citations_df["url_valid"].sum()
        logger.info(
            f"URL validity: {valid_urls}/{len(citations_df)} ({valid_urls / len(citations_df) * 100:.1f}%) valid"
        )

    # Save to parquet
    logger.info(f"Saving citations table to {output_file}")
    try:
        citations_df.to_parquet(output_file, index=False)
        logger.info(f"Successfully saved {len(citations_df)} citation records")
    except Exception as e:
        logger.error(f"Failed to save citations table: {e}")
        return

    # Validation check - reload and verify
    try:
        validation_df = pd.read_parquet(output_file)
        if len(validation_df) == len(citations_df):
            logger.info("✓ Citations table validation successful")
        else:
            logger.error(
                f"✗ Citations table validation failed: expected {len(citations_df)}, got {len(validation_df)}"
            )
    except Exception as e:
        logger.error(f"✗ Citations table validation failed: {e}")


def process_web_search_trace(web_search_trace, responses, config, total_citations):
    """Process web search trace for a set of responses."""
    citations = []
    invalid_urls = 0
    parsing_errors = 0

    # web_search_trace is typically organized by turns
    # Each turn can have multiple citation arrays
    for turn_idx, turn_trace in enumerate(web_search_trace):
        if not hasattr(turn_trace, "__iter__"):
            continue

        # Find the corresponding response for this turn
        turn_number = turn_idx + 1
        turn_responses = responses[responses["turn_number"] == turn_number]

        if len(turn_responses) == 0:
            continue

        response_id = turn_responses.iloc[0]["response_id"]

        # Extract citations from this turn's trace
        citation_order = 0

        # turn_trace can be nested arrays of citations
        for citation_array in turn_trace:
            if not hasattr(citation_array, "__iter__"):
                continue

            # Each citation should be in format ['[1]', 'https://example.com']
            try:
                if len(citation_array) >= 2:
                    citation_ref = str(citation_array[0])
                    citation_url = str(citation_array[1])

                    # Extract citation number
                    citation_number = extract_citation_number(citation_ref)

                    # Validate and process URL
                    url_valid = is_valid_url(citation_url)
                    if not url_valid:
                        invalid_urls += 1

                    domain_full = extract_domain_full(citation_url)
                    domain = extract_domain(citation_url)

                    # Create citation record
                    citation_id = f"{config['citation_id_prefix']}{total_citations:08d}"
                    citation_record = {
                        "citation_id": citation_id,
                        "response_id": response_id,
                        "citation_number": citation_number,
                        "url": citation_url,
                        "domain_full": domain_full,
                        "domain": domain,
                        "url_valid": url_valid,
                        "citation_order": citation_order,
                    }

                    citations.append(citation_record)
                    total_citations += 1
                    citation_order += 1

            except Exception as e:
                parsing_errors += 1
                if parsing_errors <= 10:  # Log first few errors
                    logger.warning(f"Citation parsing error: {e}")

    return {
        "citations": citations,
        "total_count": total_citations,
        "invalid_urls": invalid_urls,
        "parsing_errors": parsing_errors,
    }


if __name__ == "__main__":
    extract_citations()
