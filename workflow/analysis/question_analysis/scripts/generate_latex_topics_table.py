#!/usr/bin/env python3
"""
Generate LaTeX topics table for paper.

This script creates a publication-ready LaTeX table showing the topics discovered
in the question analysis, including KeyBERT keywords and OpenAI labels.
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


def create_latex_topics_table(topic_info):
    """Create a LaTeX table with topic information."""

    # Start building LaTeX table
    latex_content = []

    # Create comprehensive caption
    n_topics = topic_info["metadata"]["n_topics"]
    n_questions = topic_info["metadata"]["n_questions"]

    caption_text = (
        "Question Topics Discovered Through Clustering Analysis. "
        f"Topics were identified from {n_questions:,} questions using BERTopic clustering with UMAP dimensionality reduction. "
        "KeyBERT keywords represent the most relevant terms extracted using KeyBERT algorithm. "
        "Topic labels provide human-readable descriptions of the discovered topics. "
        f"Total of {n_topics - 1} main topics were discovered (excluding outliers)."
    )

    # Table header
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append(f"\\caption{{{caption_text}}}")
    latex_content.append("\\label{tab:question_topics}")

    # Column specification - use p{} for text wrapping
    latex_content.append("\\begin{tabular}{cp{5cm}p{7cm}}")
    latex_content.append("\\toprule")

    # Header row
    header_row = "Topic & KeyBERT Keywords & Label \\\\"
    latex_content.append(header_row)
    latex_content.append("\\midrule")

    # Sort topics by topic number (but put -1 at the end)
    topics = topic_info["topics"]
    regular_topics = [t for t in topics if t["topic"] != -1]
    outlier_topics = [t for t in topics if t["topic"] == -1]

    # Sort regular topics by topic number
    regular_topics.sort(key=lambda x: x["topic"])
    sorted_topics = regular_topics + outlier_topics

    topic_labels = {
        "topic_0": "AI models and technology",
        "topic_1": "stock prices and market",
        "topic_2": "diet, nutrients, and health",
        "topic_3": "news updates",
        "topic_4": "sports and entertainment",
        "topic_5": "biography and personal stories",
        "topic_6": "fictional character battle analysis",
        "topic_7": "online content and book",
        "topic_8": "music and lyrics",
        "topic_9": "comics and games",
    }

    # Add topic rows
    for topic in sorted_topics:
        topic_id = topic["topic"]
        if topic_id == -1:
            continue

        keybert_keywords = topic["keybert_keywords"]
        topic_label = topic_labels[f"topic_{topic_id}"]

        topic_display = f"{topic_id}"

        # Format KeyBERT keywords (take first 8 for space)
        keywords_text = ", ".join(keybert_keywords)

        # Escape LaTeX special characters
        def escape_latex(text):
            return (
                text.replace("&", "\\&")
                .replace("%", "\\%")
                .replace("$", "\\$")
                .replace("#", "\\#")
                .replace("_", "\\_")
                .replace("{", "\\{")
                .replace("}", "\\}")
                .replace("^", "\\textasciicircum{}")
                .replace("~", "\\textasciitilde{}")
            )

        keywords_text = escape_latex(keywords_text)
        topic_label = escape_latex(topic_label)

        # Create row
        row = f"{topic_display} & {keywords_text} & {topic_label} \\\\"
        latex_content.append(row)

    # Table footer
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")

    return "\n".join(latex_content)


def load_topic_info(topic_info_path):
    """Load topic information from JSON file."""
    logger.info(f"Loading topic information from {topic_info_path}")

    with open(topic_info_path, "r") as f:
        topic_info = json.load(f)

    logger.info(f"Loaded information for {len(topic_info['topics'])} topics")
    return topic_info


def main():
    """Main function for LaTeX topics table generation."""
    try:
        # Get input/output paths from Snakemake
        topic_info_path = snakemake.input[0]  # type: ignore
        output_path = snakemake.output[0]  # type: ignore

    except NameError:
        # Fallback for running outside Snakemake
        logger.info("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent

        topic_info_path = (
            base_dir / "data/intermediate/question_analysis/topic_info.json"
        )
        output_path = base_dir / "data/output/question_analysis/topics_table.tex"

    # Load topic information
    topic_info = load_topic_info(topic_info_path)

    # Create LaTeX table
    logger.info("Generating LaTeX topics table...")
    latex_table = create_latex_topics_table(topic_info)

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save LaTeX table
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    logger.info(f"LaTeX topics table saved to {output_path}")

    # Print summary
    logger.info("=== LATEX TOPICS TABLE GENERATION SUMMARY ===")
    logger.info(f"Topics included: {len(topic_info['topics'])}")
    logger.info(f"Total questions: {topic_info['metadata']['n_questions']:,}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Show preview
    logger.info("\n=== TABLE PREVIEW ===")
    lines = latex_table.split("\\n")
    for line in lines[:15]:  # Show first 15 lines
        logger.info(line)
    if len(lines) > 15:
        logger.info("... (table continues)")

    logger.info("\n✅ LaTeX topics table generation completed!")
    logger.info(
        f"You can include this table in your paper with: \\input{{{output_path}}}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
