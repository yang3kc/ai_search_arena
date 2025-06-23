#!/usr/bin/env python3
"""
Domain classification enrichment module.

This module enriches domains with classification labels:
1. Manual classifications from domain_classification_manual.csv
2. News identification from list_of_news_domains.csv
3. Gov/edu identification from domain endings
4. Unclassified for remaining domains
"""

import pandas as pd


def load_manual_classification_data(manual_classification_path):
    """Load manual domain classification data."""
    print(f"Loading manual domain classifications from {manual_classification_path}")

    try:
        manual_data = pd.read_csv(manual_classification_path)
        print(f"Loaded {len(manual_data):,} manual domain classifications")

        # Remove any empty rows
        manual_data = manual_data.dropna(subset=["domain"])
        print(f"Valid entries: {len(manual_data):,}")

        # Show classification distribution
        print("Manual classification distribution:")
        for classification, count in (
            manual_data["classification"].value_counts().items()
        ):
            print(f"  {classification}: {count:,}")

        return manual_data

    except Exception as e:
        print(f"Error loading manual classification data: {e}")
        return pd.DataFrame(columns=["domain", "classification"])


def load_news_domains_data(news_domains_path):
    """Load news domains list."""
    print(f"Loading news domains from {news_domains_path}")

    try:
        news_data = pd.read_csv(news_domains_path)
        print(f"Loaded {len(news_data):,} news domains")

        # Remove any empty rows
        news_data = news_data.dropna(subset=["domain"])
        print(f"Valid news domains: {len(news_data):,}")

        # Create set for fast lookup
        news_domains_set = set(news_data["domain"].str.lower())

        return news_domains_set

    except Exception as e:
        print(f"Error loading news domains data: {e}")
        return set()


def classify_gov_edu_domains(domain):
    """Classify domains as gov_edu based on their endings."""
    if not isinstance(domain, str):
        return False

    domain_lower = domain.lower()

    # Common government and educational domain endings
    gov_edu_endings = [
        ".gov",
        ".edu",
        ".gov.",
        ".edu.",
        ".mil",
        ".ac.uk",
        ".ac.",
        ".edu.au",
        ".gov.au",
        ".gov.uk",
        ".edu.cn",
        ".gov.cn",
        ".edu.de",
        ".gov.de",
        ".edu.fr",
        ".gouv.fr",
        ".edu.ca",
        ".gc.ca",
        ".gov.ca",
    ]

    return any(domain_lower.endswith(ending) for ending in gov_edu_endings)


def classify_domain(domain, manual_classifications, news_domains_set):
    """
    Classify a single domain using the 4-step hierarchy:
    1. Manual classification first
    2. News domain identification
    3. Gov/edu identification
    4. Unclassified
    """
    if not isinstance(domain, str):
        return "unclassified"

    domain_lower = domain.lower()

    # Step 1: Check manual classifications
    if domain_lower in manual_classifications:
        return manual_classifications[domain_lower]

    # Step 2: Check news domains
    if domain_lower in news_domains_set:
        return "news"

    # Step 3: Check gov/edu domains
    if classify_gov_edu_domains(domain):
        return "gov_edu"

    # Step 4: Default to unclassified
    return "unclassified"


def enrich_with_domain_classification(domains, manual_data, news_domains_set):
    """Enrich domains dataset with classification labels."""
    print(f"Enriching {len(domains):,} domains with classification labels...")

    # Create lookup dictionary for manual classifications
    manual_classifications = {}
    if not manual_data.empty:
        manual_classifications = dict(
            zip(manual_data["domain"].str.lower(), manual_data["classification"])
        )

    print(f"Using {len(manual_classifications):,} manual classifications")
    print(f"Using {len(news_domains_set):,} news domains")

    # Apply classification to all domains
    domains["domain_classification"] = domains["domain"].apply(
        lambda x: classify_domain(x, manual_classifications, news_domains_set)
    )

    # Generate classification statistics
    print("\n=== DOMAIN CLASSIFICATION RESULTS ===")

    total_domains = len(domains)
    total_citations = domains["citation_count"].sum()

    print(f"Total domains classified: {total_domains:,}")
    print(f"Total citations represented: {total_citations:,}")

    # Classification distribution by domain count
    print(f"\nClassification distribution (by domain count):")
    class_counts = domains["domain_classification"].value_counts()
    for classification, count in class_counts.items():
        pct = count / total_domains * 100
        print(f"  {classification}: {count:,} domains ({pct:.1f}%)")

    # Classification distribution by citation count
    print(f"\nClassification distribution (by citation count):")
    for classification in class_counts.index:
        citations = domains[domains["domain_classification"] == classification][
            "citation_count"
        ].sum()
        pct = citations / total_citations * 100
        print(f"  {classification}: {citations:,} citations ({pct:.1f}%)")

    # Show examples for each classification
    print(f"\nTop domains by classification (by citation count):")
    for classification in [
        "news",
        "social_media",
        "tech",
        "academic",
        "gov_edu",
        "wiki",
        "community_blog",
        "search_engine",
        "other",
        "unclassified",
    ]:
        class_domains = domains[domains["domain_classification"] == classification]
        if len(class_domains) > 0:
            top_domain = class_domains.nlargest(1, "citation_count")
            if len(top_domain) > 0:
                domain_name = top_domain.iloc[0]["domain"]
                citation_count = top_domain.iloc[0]["citation_count"]
                print(
                    f"  {classification}: {domain_name} ({citation_count:,} citations)"
                )

    return domains


def main():
    """Main function for domain classification enrichment (if run standalone)."""
    print("Domain classification enrichment module")
    print("This module is designed to be imported by enrich_domains_combined.py")


if __name__ == "__main__":
    main()
