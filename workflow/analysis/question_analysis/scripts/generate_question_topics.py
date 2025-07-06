#!/usr/bin/env python3
"""
Generate topic modeling for questions using BERTopic.

This script creates topic models from user questions using BERTopic,
extracting both topic assignments and topic probability distributions.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_question_data(questions_path, embeddings_path):
    """Load question data and embeddings."""
    logger.info(f"Loading questions from {questions_path}")
    questions_df = pd.read_parquet(questions_path)

    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings_df = pd.read_parquet(embeddings_path)

    # Merge questions with embeddings
    question_data = embeddings_df.merge(
        questions_df[["question_id", "user_query"]], on="question_id"
    )

    logger.info(f"Loaded {len(question_data):,} questions with embeddings")
    return question_data


def extract_embeddings_and_queries(question_data):
    """Extract embeddings and queries from the merged data."""
    logger.info("Extracting embeddings and queries...")

    # Get embedding columns
    embedding_cols = [
        col for col in question_data.columns if col.startswith("embedding_dim_")
    ]

    # Extract embeddings as numpy array
    embeddings = question_data[embedding_cols].values

    # Extract queries as numpy array
    queries = question_data["user_query"].values

    logger.info(
        f"Extracted {len(queries):,} queries with {len(embedding_cols)} embedding dimensions"
    )
    return embeddings, queries, question_data["question_id"].values


def create_topic_model():
    """Create and configure the BERTopic model."""
    logger.info("Creating BERTopic model...")

    # Configure embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Configure UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )

    # Configure HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=150,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Configure vectorizer for term extraction
    vectorizer_model = CountVectorizer(
        stop_words="english", min_df=2, ngram_range=(1, 2)
    )

    # Configure representation models
    keybert_model = KeyBERTInspired()
    mmr_model = MaximalMarginalRelevance(diversity=0.3)

    representation_model = {
        "KeyBERT": keybert_model,
        "MMR": mmr_model,
    }

    # Add OpenAI representation model if available
    if OPENAI_AVAILABLE:
        try:
            # OpenAI prompt for generating topic labels
            openai_prompt = """
            I have a topic that contains the following documents:
            [DOCUMENTS]
            The topic is described by the following keywords: [KEYWORDS]

            Based on the information above, extract a short but highly descriptive topic label of at most 8 words. Make sure it is in the following format:
            topic: <topic label>
            """

            client = openai.OpenAI()
            openai_model = OpenAI(
                client,
                model="gpt-4.1",
                exponential_backoff=True,
                chat=True,
                prompt=openai_prompt,
            )
            representation_model["OpenAI"] = openai_model
            logger.info("OpenAI representation model added successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI model: {e}")
            logger.info("Continuing without OpenAI representation model")
    else:
        logger.info("OpenAI not available, using KeyBERT and MMR only")

    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=10,
        calculate_probabilities=True,
        verbose=True,
    )

    logger.info("BERTopic model created successfully")
    return topic_model


def fit_topic_model(topic_model, queries, embeddings):
    """Fit the topic model to the data."""
    logger.info("Fitting topic model to queries...")

    # Fit the model
    topics, probs = topic_model.fit_transform(queries, embeddings)

    logger.info(f"Topic modeling completed. Found {len(set(topics))} topics")

    # Get topic info
    topic_info = topic_model.get_topic_info()
    logger.info(f"Topic distribution:\n{topic_info[['Topic', 'Count']].head(10)}")

    return topics, probs, topic_info


def reduce_outliers(topic_model, queries, topics, embeddings):
    """Reduce outliers using embeddings strategy."""
    logger.info("Reducing outliers...")

    # Get original outlier count
    original_outliers = np.sum(np.array(topics) == -1)
    logger.info(f"Original outliers: {original_outliers:,}")

    # Reduce outliers
    new_topics = topic_model.reduce_outliers(
        queries, topics, strategy="embeddings", embeddings=embeddings
    )

    # Update topics in the model
    topic_model.update_topics(queries, topics=new_topics)

    # Get new outlier count
    new_outliers = np.sum(np.array(new_topics) == -1)
    logger.info(f"Outliers after reduction: {new_outliers:,}")
    logger.info(f"Outliers reduced by: {original_outliers - new_outliers:,}")

    return new_topics


def generate_topic_distributions(topic_model, queries):
    """Generate topic probability distributions for each query."""
    logger.info("Generating topic probability distributions...")

    # Approximate topic distributions
    topic_distributions, _ = topic_model.approximate_distribution(queries)

    logger.info(
        f"Generated topic distributions with shape: {topic_distributions.shape}"
    )

    # Check distribution properties
    non_zero_counts = np.sum(topic_distributions > 0, axis=1)
    logger.info(f"Average topics per query: {np.mean(non_zero_counts):.2f}")
    logger.info(f"Queries with single dominant topic: {np.sum(non_zero_counts == 1):,}")

    return topic_distributions


def save_topic_results(
    question_ids,
    topics,
    topic_distributions,
    topic_info,
    topics_output_path,
    probabilities_output_path,
    info_output_path,
):
    """Save topic modeling results."""
    logger.info("Saving topic modeling results...")

    # Create question topics DataFrame
    question_topics = pd.DataFrame({"question_id": question_ids, "topic": topics})

    # Save topic assignments
    topics_output_path = Path(topics_output_path)
    topics_output_path.parent.mkdir(parents=True, exist_ok=True)
    question_topics.to_parquet(topics_output_path, index=False)
    logger.info(f"Saved topic assignments to {topics_output_path}")

    # Create topic probabilities DataFrame
    n_topics = topic_distributions.shape[1]
    topic_prob_cols = [f"topic_{i}_prob" for i in range(n_topics)]

    topic_probabilities = pd.DataFrame(topic_distributions, columns=topic_prob_cols)
    topic_probabilities["question_id"] = question_ids

    # Reorder columns to put question_id first
    columns = ["question_id"] + topic_prob_cols
    topic_probabilities = topic_probabilities[columns]

    # Save topic probabilities
    probabilities_output_path = Path(probabilities_output_path)
    probabilities_output_path.parent.mkdir(parents=True, exist_ok=True)
    topic_probabilities.to_parquet(probabilities_output_path, index=False)
    logger.info(f"Saved topic probabilities to {probabilities_output_path}")

    # Save topic info as JSON
    info_output_path = Path(info_output_path)
    info_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert topic info to JSON-serializable format
    topic_info_dict = {
        "metadata": {
            "n_topics": len(topic_info),
            "n_questions": len(question_ids),
            "openai_used": "OpenAI" in topic_info.columns,
            "model_config": {
                "min_cluster_size": 150,
                "umap_n_components": 5,
                "umap_n_neighbors": 15,
                "top_n_words": 10,
            },
        },
        "topics": [],
    }

    for _, row in topic_info.iterrows():
        topic_dict = {
            "topic": int(row["Topic"]),
            "count": int(row["Count"]),
            "name": row["Name"],
            "representation": row["Representation"][:10],  # Top 10 words
            "keybert_keywords": row["KeyBERT"][:10] if "KeyBERT" in row else [],
            "mmr_keywords": row["MMR"][:10] if "MMR" in row else [],
            "openai_label": row["OpenAI"][0]
            if "OpenAI" in row and len(row["OpenAI"]) > 0
            else None,
        }
        topic_info_dict["topics"].append(topic_dict)

    with open(info_output_path, "w") as f:
        json.dump(topic_info_dict, f, indent=2)

    logger.info(f"Saved topic info to {info_output_path}")


def validate_results(question_topics, topic_probabilities, topic_info):
    """Validate the topic modeling results."""
    logger.info("Validating topic modeling results...")

    # Check topic assignments
    topic_counts = question_topics["topic"].value_counts()
    logger.info("Topic assignment distribution:")
    logger.info(
        f"  Most common topic: {topic_counts.index[0]} ({topic_counts.iloc[0]:,} questions)"
    )
    logger.info(
        f"  Least common topic: {topic_counts.index[-1]} ({topic_counts.iloc[-1]:,} questions)"
    )
    logger.info(f"  Outliers (topic -1): {topic_counts.get(-1, 0):,} questions")

    # Check probability distributions
    prob_cols = [col for col in topic_probabilities.columns if col.endswith("_prob")]
    row_sums = topic_probabilities[prob_cols].sum(axis=1)

    logger.info("Topic probability validation:")
    logger.info(f"  Mean row sum: {row_sums.mean():.4f}")
    logger.info(f"  Rows summing to 1.0: {np.sum(np.isclose(row_sums, 1.0)):,}")
    logger.info(f"  Rows summing to 0.0: {np.sum(np.isclose(row_sums, 0.0)):,}")

    # Check topic info
    logger.info("Topic info validation:")
    logger.info(f"  Number of topics: {len(topic_info)}")
    logger.info(f"  Total questions across topics: {topic_info['Count'].sum():,}")

    logger.info("✅ Topic modeling validation completed")


def main():
    """Main function for topic modeling."""
    try:
        # Get input/output paths from Snakemake
        questions_path = snakemake.input.questions  # type: ignore
        embeddings_path = snakemake.input.embeddings  # type: ignore
        topics_output_path = snakemake.output.topics  # type: ignore
        probabilities_output_path = snakemake.output.probabilities  # type: ignore
        info_output_path = snakemake.output.info  # type: ignore

    except NameError:
        # Fallback for running outside Snakemake
        logger.info("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent

        questions_path = (
            base_dir / "data/intermediate/question_analysis/english_questions.parquet"
        )
        embeddings_path = (
            base_dir / "data/intermediate/question_analysis/question_embeddings.parquet"
        )
        topics_output_path = (
            base_dir / "data/intermediate/question_analysis/question_topics.parquet"
        )
        probabilities_output_path = (
            base_dir
            / "data/intermediate/question_analysis/question_topic_probabilities.parquet"
        )
        info_output_path = (
            base_dir / "data/intermediate/question_analysis/topic_info.json"
        )

    # Load data
    question_data = load_question_data(questions_path, embeddings_path)

    # Extract embeddings and queries
    embeddings, queries, question_ids = extract_embeddings_and_queries(question_data)

    # Create topic model
    topic_model = create_topic_model()

    # Fit topic model
    topics, probs, topic_info = fit_topic_model(topic_model, queries, embeddings)

    # Reduce outliers
    topics = reduce_outliers(topic_model, queries, topics, embeddings)

    # Generate topic distributions
    topic_distributions = generate_topic_distributions(topic_model, queries)

    # Create results DataFrames
    question_topics = pd.DataFrame({"question_id": question_ids, "topic": topics})

    n_topics = topic_distributions.shape[1]
    topic_prob_cols = [f"topic_{i}_prob" for i in range(n_topics)]
    topic_probabilities = pd.DataFrame(topic_distributions, columns=topic_prob_cols)
    topic_probabilities["question_id"] = question_ids
    topic_probabilities = topic_probabilities[["question_id"] + topic_prob_cols]

    # Get updated topic info
    topic_info = topic_model.get_topic_info()

    # Validate results
    validate_results(question_topics, topic_probabilities, topic_info)

    # Save results
    save_topic_results(
        question_ids,
        topics,
        topic_distributions,
        topic_info,
        topics_output_path,
        probabilities_output_path,
        info_output_path,
    )

    logger.info("✅ Topic modeling completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
