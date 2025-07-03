#!/usr/bin/env python3
"""
Generate Question Embeddings Script for Question Analysis Pipeline.

This script generates semantic embeddings for English questions using 
sentence transformers for downstream analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings

# Suppress sentence transformers warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QuestionEmbeddingGenerator:
    """Generate semantic embeddings for questions using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.embedding_dim = None
        
    def load_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        self.model = SentenceTransformer(self.model_name, device=device)
        
        # Get embedding dimension
        sample_embedding = self.model.encode(["test"], show_progress_bar=False)
        self.embedding_dim = sample_embedding.shape[1]
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
    def generate_embeddings(self, questions: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of questions.
        
        Args:
            questions: List of question strings
            
        Returns:
            numpy array of embeddings (n_questions, embedding_dim)
        """
        if self.model is None:
            self.load_model()
            
        logger.info(f"Generating embeddings for {len(questions):,} questions")
        
        # Generate embeddings in batches with progress bar
        embeddings = self.model.encode(
            questions,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for better similarity computation
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def validate_embeddings(self, embeddings: np.ndarray, questions: List[str]):
        """
        Validate the generated embeddings.
        
        Args:
            embeddings: Generated embeddings array
            questions: Original questions list
        """
        logger.info("Validating embeddings...")
        
        # Check dimensions
        assert embeddings.shape[0] == len(questions), "Embedding count mismatch"
        assert embeddings.shape[1] == self.embedding_dim, "Embedding dimension mismatch"
        
        # Check for NaN or infinite values
        nan_count = np.isnan(embeddings).sum()
        inf_count = np.isinf(embeddings).sum()
        
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in embeddings")
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values in embeddings")
            
        # Check embedding norms (should be ~1.0 after normalization)
        norms = np.linalg.norm(embeddings, axis=1)
        logger.info(f"Embedding norms - Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")
        
        # Sample similarity check
        if len(questions) >= 2:
            # Compute similarity between first two questions
            sim = np.dot(embeddings[0], embeddings[1])
            logger.info(f"Sample similarity between questions 0 and 1: {sim:.4f}")
            
        logger.info("✅ Embedding validation completed")


def preprocess_questions(questions_df: pd.DataFrame, test_mode: bool = False) -> pd.DataFrame:
    """
    Preprocess questions for embedding generation.
    
    Args:
        questions_df: DataFrame with questions
        test_mode: If True, only process first 100 questions for testing
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Preprocessing questions for embedding generation...")
    
    df = questions_df.copy()
    
    # Test mode - limit to first 100 questions
    if test_mode:
        df = df.head(100).copy()
        logger.info(f"Test mode: limiting to {len(df)} questions")
    
    # Ensure user_query is string type
    df["user_query"] = df["user_query"].astype(str)
    
    # Remove any remaining null/empty questions
    initial_count = len(df)
    df = df[df["user_query"].str.strip() != ""].copy()
    df = df[df["user_query"] != "nan"].copy()
    logger.info(f"Removed {initial_count - len(df):,} empty/null questions")
    
    # Clean questions (basic text cleaning)
    df["cleaned_query"] = df["user_query"].str.strip()
    
    # Remove excessive whitespace
    df["cleaned_query"] = df["cleaned_query"].str.replace(r'\s+', ' ', regex=True)
    
    logger.info(f"Preprocessed {len(df):,} questions for embedding generation")
    return df


def create_embedding_dataframe(
    questions_df: pd.DataFrame, 
    embeddings: np.ndarray, 
    embedding_dim: int
) -> pd.DataFrame:
    """
    Create DataFrame with question metadata and embeddings.
    
    Args:
        questions_df: Original questions DataFrame
        embeddings: Generated embeddings array
        embedding_dim: Dimension of embeddings
        
    Returns:
        DataFrame with embeddings and metadata
    """
    logger.info("Creating embedding DataFrame...")
    
    # Create base DataFrame with question metadata
    embedding_df = questions_df[["question_id", "thread_id", "turn_number", "user_query"]].copy()
    
    # Add embedding columns
    embedding_cols = [f"embedding_dim_{i}" for i in range(embedding_dim)]
    embedding_data = pd.DataFrame(embeddings, columns=embedding_cols, index=embedding_df.index)
    
    # Combine metadata and embeddings
    result_df = pd.concat([embedding_df, embedding_data], axis=1)
    
    # Add metadata about embeddings
    result_df["embedding_model"] = "all-MiniLM-L6-v2"
    result_df["embedding_dim"] = embedding_dim
    result_df["query_length"] = result_df["user_query"].str.len()
    
    logger.info(f"Created embedding DataFrame with shape: {result_df.shape}")
    return result_df


def validate_output(df: pd.DataFrame, embedding_dim: int):
    """Validate the final output DataFrame."""
    logger.info("Validating output DataFrame...")
    
    # Check required columns
    required_cols = ["question_id", "thread_id", "user_query"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check embedding columns
    embedding_cols = [f"embedding_dim_{i}" for i in range(embedding_dim)]
    missing_embedding_cols = [col for col in embedding_cols if col not in df.columns]
    if missing_embedding_cols:
        raise ValueError(f"Missing embedding columns: {missing_embedding_cols}")
    
    # Check for duplicates
    duplicates = df["question_id"].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate question IDs")
    
    # Check data types
    for col in embedding_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Embedding column {col} is not numeric")
    
    # Sample output
    logger.info("Sample embedding output:")
    sample_row = df.iloc[0]
    logger.info(f"Question ID: {sample_row['question_id']}")
    logger.info(f"Query: {sample_row['user_query'][:100]}...")
    first_5_values = [f"{sample_row[f'embedding_dim_{i}']:.4f}" for i in range(5)]
    logger.info(f"First 5 embedding values: {first_5_values}")
    
    logger.info("✅ Output validation completed")


def main():
    """Main function for question embedding generation."""
    try:
        # Get input/output paths from Snakemake
        input_path = snakemake.input[0]
        output_path = snakemake.output[0]
        
        # Get config parameters
        config = snakemake.config
        model_name = config.get("embedding", {}).get("model_name", "all-MiniLM-L6-v2")
        batch_size = config.get("embedding", {}).get("batch_size", 32)
        
    except NameError:
        # Fallback for running outside Snakemake (for testing)
        logger.info("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        input_path = base_dir / "data/intermediate/question_analysis/english_questions.parquet"
        output_path = base_dir / "data/intermediate/question_analysis/question_embeddings.parquet"
        model_name = "all-MiniLM-L6-v2"
        batch_size = 32
    
    # Load questions
    logger.info(f"Loading questions from {input_path}")
    questions_df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(questions_df):,} questions")
    
    # Check for test mode (check if TEST_MODE environment variable is set)
    import os
    test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
    
    # Preprocess questions
    preprocessed_df = preprocess_questions(questions_df, test_mode=test_mode)
    
    # Initialize embedding generator
    generator = QuestionEmbeddingGenerator(model_name=model_name, batch_size=batch_size)
    
    # Generate embeddings
    questions_list = preprocessed_df["cleaned_query"].tolist()
    embeddings = generator.generate_embeddings(questions_list)
    
    # Validate embeddings
    generator.validate_embeddings(embeddings, questions_list)
    
    # Create output DataFrame
    embedding_df = create_embedding_dataframe(preprocessed_df, embeddings, generator.embedding_dim)
    
    # Validate output
    validate_output(embedding_df, generator.embedding_dim)
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    embedding_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(embedding_df):,} question embeddings to {output_path}")
    
    # Summary statistics
    logger.info("=== QUESTION EMBEDDING GENERATION SUMMARY ===")
    logger.info(f"Input questions: {len(questions_df):,}")
    logger.info(f"Processed questions: {len(embedding_df):,}")
    logger.info(f"Model used: {model_name}")
    logger.info(f"Embedding dimension: {generator.embedding_dim}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Average query length: {embedding_df['query_length'].mean():.1f} characters")
    logger.info(f"Output file size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    logger.info("✅ Question embedding generation completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())