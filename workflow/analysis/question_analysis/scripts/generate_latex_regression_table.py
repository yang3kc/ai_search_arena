#!/usr/bin/env python3
"""
Generate LaTeX regression coefficients table for paper.

This script creates a publication-ready LaTeX table showing regression coefficients
for key news source citation outcomes, with statistical significance markers.
"""

import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_topic_label_mapping():
    """Get mapping from topic variable names to human-readable labels."""
    return {
        "topic_0": "Topic: Guide to Selecting and Using AI Models",
        "topic_1": "Topic: Stock Prices and Market Volatility Today",
        "topic_2": "Topic: Diet, Nutrients, and Health-Related Medical Claims",
        "topic_3": "Topic: Latest News Updates Around the World",
        "topic_4": "Topic: FIFA World Cup Europe vs South America Finals",
        "topic_5": "Topic: Biographical Details of Internet Creators and Engineers",
        "topic_6": "Topic: Fictional Character Battle Analysis and Comparisons",
        "topic_7": "Topic: Cross Creek book summary and analysis",
        "topic_8": "Topic: Identifying and Sharing Song Lyrics from Quotes",
        "topic_9": "Topic: Dark Alternate Tails as Villain in Sonic",
        "topic_-1": "Topic: Outlier/Noise",
        "topic_0_prob": "Topic: AI models and technology",
        "topic_1_prob": "Topic: stock prices and market",
        "topic_2_prob": "Topic: diet, nutrients, and health",
        "topic_3_prob": "Topic: news updates",
        "topic_4_prob": "Topic: sports and entertainment",
        "topic_5_prob": "Topic: biography and personal stories",
        "topic_6_prob": "Topic: fictional character battle analysis",
        "topic_7_prob": "Topic: online content and book",
        "topic_8_prob": "Topic: music and lyrics",
        "topic_9_prob": "Topic: comics and games",
    }


def format_feature_name(feature_name):
    """Format feature names for display, using topic labels where applicable."""
    topic_mapping = get_topic_label_mapping()

    if feature_name in topic_mapping:
        return topic_mapping[feature_name]

    # Handle other feature formatting
    if feature_name.startswith("embedding_pc_"):
        pc_num = feature_name.split("_")[-1]
        return f"Embedding: PC {pc_num}"
    elif feature_name.startswith("client_country_"):
        country = feature_name.replace("client_country_", "")
        if country == "nan":
            return "Client country/region: unknown"
        if country == "Other":
            return "Client country/region: other"
        return f"Client country/region: {country}"
    elif feature_name.startswith("model_family_"):
        family = feature_name.replace("model_family_", "")
        return f"Model family: {family.title()}"
    elif feature_name.startswith("primary_intent_"):
        intent = feature_name.replace("primary_intent_", "")
        return f"Intent: {intent.lower()}"
    elif feature_name == "question_length_words_log":
        return "Question length (words, log Z-score)"
    elif feature_name == "response_word_count_log":
        return "Response length (words, log Z-score)"
    elif feature_name == "turn_number":
        return "Turn number"
    elif feature_name == "total_turns":
        return "Total turns"
    elif feature_name == "num_citations":
        return "Number of citations"
    elif feature_name == "proportion_news":
        return "News sources percentage"
    else:
        # Default formatting: replace underscores with spaces and title case
        return feature_name.replace("_", " ").title()


def create_latex_regression_table(results, outcomes_to_include):
    """Create a LaTeX table with regression coefficients for multiple outcomes."""

    # Get all unique features across the specified outcomes
    all_features = set()
    outcome_data = {}

    for result in results["regression_results"]:
        outcome = result["outcome"]
        if outcome in outcomes_to_include:
            outcome_data[outcome] = result

            # Add intercept
            all_features.add("Intercept")

            # Add significant features (excluding embeddings)
            for feature in result["coefficients"]["features"]:
                all_features.add(feature["feature"])

    # Sort features for consistent ordering (intercept first)
    sorted_features = ["Intercept"] + sorted(
        [f for f in all_features if f != "Intercept"]
    )

    # Build coefficient data for each outcome
    coef_data = {}
    for outcome in outcomes_to_include:
        if outcome in outcome_data:
            result = outcome_data[outcome]
            coef_data[outcome] = {}

            # Add intercept
            intercept = result["coefficients"]["intercept"]
            coef_data[outcome]["Intercept"] = {
                "coefficient": intercept["coefficient"],
                "p_value": intercept["p_value"],
                "significant": intercept["p_value"] < 0.05,
            }

            # Add features
            for feature in result["coefficients"]["features"]:
                if feature["feature"] in sorted_features:
                    coef_data[outcome][feature["feature"]] = {
                        "coefficient": feature["coefficient"],
                        "p_value": feature["p_value"],
                        "significant": feature["significant"],
                    }

    # Format outcome names for column headers
    def format_outcome_name(outcome):
        if outcome == "news_proportion_left_leaning":
            return "\% left-leaning news"
        elif outcome == "news_proportion_right_leaning":
            return "\% right-leaning news"
        elif outcome == "news_proportion_center_leaning":
            return "\% center-leaning news"
        elif outcome == "news_proportion_high_quality":
            return "\% high-quality news"
        elif outcome == "news_proportion_low_quality":
            return "\% low-quality news"
        elif outcome == "news_proportion_unknown_quality":
            return "\% unknown-quality news"
        else:
            return outcome.replace("_", " ").title()

    # Start building LaTeX table
    latex_content = []

    # Create comprehensive caption with all information
    sample_size = list(outcome_data.values())[0]["n_samples"]
    caption_text = (
        "Regression Coefficients for News Source Citation Patterns. "
        "Coefficients show the relationship between features and news source citation patterns. "
        "Statistical significance: *** p < 0.001, ** p < 0.01, * p < 0.05. "
        f"Sample size: {sample_size:,} observations. "
        "Only statistically significant features are shown."
    )

    # Table header
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append(f"\\caption{{{caption_text}}}")
    latex_content.append("\\label{tab:regression_coefficients}")

    # Column specification - adjust width for readability
    n_outcomes = len(outcomes_to_include)
    if n_outcomes == 4:
        latex_content.append("\\begin{tabular}{p{4cm}cccc}")
    else:
        col_spec = "l" + "c" * n_outcomes
        latex_content.append(f"\\begin{{tabular}}{{{col_spec}}}")

    latex_content.append("\\toprule")

    # Header row with outcome names
    header_row = "Feature"
    for outcome in outcomes_to_include:
        if outcome in outcome_data:
            header_row += f" & {format_outcome_name(outcome)}"
    header_row += " \\\\"
    latex_content.append(header_row)
    latex_content.append("\\midrule")

    # Add coefficient rows
    for feature in sorted_features:
        # Format feature name
        formatted_feature = format_feature_name(feature)

        # Check if this feature appears in any outcome
        appears_in_outcomes = [
            outcome
            for outcome in outcomes_to_include
            if outcome in coef_data and feature in coef_data[outcome]
        ]

        if (
            appears_in_outcomes
        ):  # Only include if feature appears in at least one outcome
            # Escape LaTeX special characters
            row = (
                formatted_feature.replace("&", "\\&")
                .replace("%", "\\%")
                .replace("$", "\\$")
            )

            for outcome in outcomes_to_include:
                if outcome in coef_data and feature in coef_data[outcome]:
                    coef_info = coef_data[outcome][feature]
                    coef_val = coef_info["coefficient"]
                    p_val = coef_info["p_value"]

                    # Format coefficient with significance stars
                    if p_val < 0.001:
                        stars = "***"
                    elif p_val < 0.01:
                        stars = "**"
                    elif p_val < 0.05:
                        stars = "*"
                    else:
                        stars = ""

                    # Format coefficient value
                    coef_str = f"{coef_val:.2f}{stars}"
                    row += f" & {coef_str}"
                else:
                    row += " & "  # Empty cell if feature not in this outcome

            row += " \\\\"
            latex_content.append(row)

    # Add R-squared row at the bottom
    latex_content.append("\\midrule")
    r2_row = "R-squared"
    for outcome in outcomes_to_include:
        if outcome in outcome_data:
            r2_value = outcome_data[outcome]["model_performance"]["r2"]
            r2_row += f" & {r2_value:.2f}"
        else:
            r2_row += " & "
    r2_row += " \\\\"
    latex_content.append(r2_row)

    # Table footer
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")

    return "\n".join(latex_content)


def load_regression_results(results_path):
    """Load regression results from JSON file."""
    logger.info(f"Loading regression results from {results_path}")

    with open(results_path, "r") as f:
        results = json.load(f)

    logger.info(f"Loaded results for {len(results['regression_results'])} outcomes")
    return results


def main():
    """Main function for LaTeX table generation."""
    try:
        # Get input/output paths from Snakemake
        results_path = snakemake.input[0]  # type: ignore
        output_path = snakemake.output[0]  # type: ignore

    except NameError:
        # Fallback for running outside Snakemake
        logger.info("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent

        results_path = (
            base_dir / "data/output/question_analysis/regression_results.json"
        )
        output_path = (
            base_dir / "data/output/question_analysis/regression_coefficients_table.tex"
        )

    # Load regression results
    results = load_regression_results(results_path)

    # Define key outcomes for the table
    key_outcomes = [
        "news_proportion_left_leaning",
        "news_proportion_right_leaning",
        "news_proportion_high_quality",
        "news_proportion_low_quality",
    ]

    # Create LaTeX table
    logger.info("Generating LaTeX regression coefficients table...")
    latex_table = create_latex_regression_table(results, key_outcomes)

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save LaTeX table
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    logger.info(f"LaTeX table saved to {output_path}")

    # Print summary
    logger.info("=== LATEX TABLE GENERATION SUMMARY ===")
    logger.info(f"Outcomes included: {len(key_outcomes)}")
    logger.info(
        f"Available outcomes in data: {[r['outcome'] for r in results['regression_results']]}"
    )
    logger.info(f"Output file: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Show preview
    logger.info("\n=== TABLE PREVIEW ===")
    lines = latex_table.split("\n")
    for line in lines[:20]:  # Show first 20 lines
        logger.info(line)
    if len(lines) > 20:
        logger.info("... (table continues)")

    logger.info("\n✅ LaTeX table generation completed!")
    logger.info(
        f"You can include this table in your paper with: \\input{{{output_path}}}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
