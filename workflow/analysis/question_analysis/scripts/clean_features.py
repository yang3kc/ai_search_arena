#!/usr/bin/env python3
"""
Feature Cleaning and Preprocessing Script for Question Analysis Pipeline.

This script prepares the integrated analysis data for regression modeling by:
- Converting categorical variables to dummy variables
- Transforming long-tail distributed variables (response length, question length)
- Standardizing/normalizing features as needed
- Creating interaction terms if necessary
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_integrated_data(input_path):
    """Load the integrated analysis data."""
    logger.info(f"Loading integrated data from {input_path}")
    data = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(data):,} rows with {len(data.columns)} columns")
    return data


def identify_categorical_variables(data):
    """Identify categorical variables that need dummy encoding."""
    logger.info("Identifying categorical variables...")

    categorical_vars = {}

    # Explicit categorical variables
    explicit_categoricals = [
        "client_country",
        "model_name_raw",
        "model_side",
        "winner",
        "primary_intent",
    ]

    for var in explicit_categoricals:
        if var in data.columns:
            unique_count = data[var].nunique()
            logger.info(f"Categorical variable '{var}': {unique_count} unique values")
            categorical_vars[var] = unique_count

    return categorical_vars


def create_dummy_variables(data, categorical_vars, max_categories=10):
    """Create dummy variables for categorical features."""
    logger.info("Creating dummy variables for categorical features...")

    dummy_data = data.copy()
    dummy_columns_created = []

    for var, unique_count in categorical_vars.items():
        if var not in data.columns:
            continue

        # Handle high cardinality variables (like client_country)
        if unique_count > max_categories:
            logger.info(
                f"High cardinality variable '{var}' ({unique_count} categories)"
            )

            if var == "client_country":
                # Keep top N countries, group others as "Other", preserve NaN
                top_countries = data[var].value_counts().head(max_categories).index
                dummy_data[f"{var}_grouped"] = data[var].apply(
                    lambda x: x if pd.isna(x) or x in top_countries else "Other"
                )

                # Create dummies for grouped variable
                dummies = pd.get_dummies(
                    dummy_data[f"{var}_grouped"], prefix=var, dummy_na=True
                )
                dummy_data = pd.concat([dummy_data, dummies], axis=1)
                dummy_columns_created.extend(dummies.columns.tolist())

                # Drop original and intermediate columns
                dummy_data = dummy_data.drop(columns=[var, f"{var}_grouped"])

            elif var == "model_name_raw":
                # Extract model family and create dummies
                def extract_model_family(model_name):
                    if pd.isna(model_name):
                        return "unknown"
                    model_name = str(model_name).lower()

                    if "gpt" in model_name:
                        return "openai"
                    elif "claude" in model_name:
                        return "anthropic"
                    elif "gemini" in model_name:
                        return "google"
                    elif "sonar" in model_name:
                        return "perplexity"
                    elif "llama" in model_name:
                        return "meta"
                    else:
                        return "other"

                dummy_data["model_family"] = data[var].apply(extract_model_family)
                dummies = pd.get_dummies(
                    dummy_data["model_family"], prefix="model_family", dummy_na=True
                )
                dummy_data = pd.concat([dummy_data, dummies], axis=1)
                dummy_columns_created.extend(dummies.columns.tolist())

                # Drop original columns
                dummy_data = dummy_data.drop(columns=[var, "model_family"])

        else:
            # Standard dummy encoding for low cardinality variables
            dummies = pd.get_dummies(data[var], prefix=var, dummy_na=True)
            dummy_data = pd.concat([dummy_data, dummies], axis=1)
            dummy_columns_created.extend(dummies.columns.tolist())

            # Drop original column
            dummy_data = dummy_data.drop(columns=[var])

    logger.info(f"Created {len(dummy_columns_created)} dummy variables")
    return dummy_data, dummy_columns_created


def transform_length_variables(data):
    """Transform length variables with long-tail distributions."""
    logger.info("Transforming length variables with long-tail distributions...")

    transformed_data = data.copy()
    length_vars = [
        "question_length_chars",
        "question_length_words",
        "response_length",
        "response_word_count",
    ]

    transformation_summary = {}

    for var in length_vars:
        if var not in data.columns:
            continue

        # Check if variable has long tail (high skewness)
        skewness = data[var].skew()
        logger.info(f"Variable '{var}' skewness: {skewness:.3f}")

        if abs(skewness) > 1.0:  # Threshold for considering transformation
            # Apply log(1+x) transformation to handle zeros and reduce skewness
            log_transformed = np.log1p(data[var])

            # Standardize to z-scores (mean=0, std=1)
            transformed_data[f"{var}_log"] = (
                log_transformed - log_transformed.mean()
            ) / log_transformed.std()

            new_skewness = transformed_data[f"{var}_log"].skew()
            final_mean = transformed_data[f"{var}_log"].mean()
            final_std = transformed_data[f"{var}_log"].std()

            logger.info(
                f"  -> Log-transformed and standardized '{var}' skewness: {new_skewness:.3f}, mean: {final_mean:.3f}, std: {final_std:.3f}"
            )
            transformation_summary[var] = {
                "original_skewness": skewness,
                "transformed_skewness": new_skewness,
                "transformation": "log1p + z-score",
                "final_mean": final_mean,
                "final_std": final_std,
            }

            # Optionally remove original variable
            # transformed_data = transformed_data.drop(columns=[var])
        else:
            logger.info(f"  -> '{var}' skewness acceptable, no transformation needed")

    return transformed_data, transformation_summary


def standardize_embeddings(data):
    """Standardize embedding dimensions."""
    logger.info("Standardizing embedding dimensions...")

    embedding_cols = [col for col in data.columns if col.startswith("embedding_dim_")]

    if not embedding_cols:
        logger.warning("No embedding columns found")
        return data

    standardized_data = data.copy()

    # Standardize embeddings (mean=0, std=1)
    scaler = StandardScaler()
    standardized_data[embedding_cols] = scaler.fit_transform(data[embedding_cols])

    logger.info(f"Standardized {len(embedding_cols)} embedding dimensions")

    return standardized_data


def apply_pca_to_embeddings(data, n_components=20):
    """Apply PCA to embedding dimensions while keeping original embeddings."""
    logger.info("Applying PCA to embedding dimensions...")

    embedding_cols = [col for col in data.columns if col.startswith("embedding_dim_")]

    if not embedding_cols:
        logger.warning("No embedding columns found for PCA")
        return data

    if len(embedding_cols) <= n_components:
        logger.warning(
            f"Number of embedding dimensions ({len(embedding_cols)}) <= n_components ({n_components}), skipping PCA"
        )
        return data

    pca_data = data.copy()

    # Apply PCA to standardized embeddings
    pca = PCA(n_components=n_components, random_state=42)
    embedding_data = data[embedding_cols]
    embedding_pca = pca.fit_transform(embedding_data)

    # Create DataFrame with PCA components
    pca_cols = [f"embedding_pc_{i}" for i in range(n_components)]
    pca_df = pd.DataFrame(embedding_pca, columns=pca_cols, index=data.index)

    # Add PCA components to the dataset (keeping original embeddings)
    pca_data = pd.concat([pca_data, pca_df], axis=1)

    logger.info(f"Applied PCA to {len(embedding_cols)} embedding dimensions")
    logger.info(f"Created {n_components} PCA components")
    logger.info(
        f"PCA explained variance ratio (first 5): {pca.explained_variance_ratio_[:5]}"
    )
    logger.info(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    return pca_data


def handle_missing_values(data):
    """Handle remaining missing values in the dataset."""
    logger.info("Handling missing values...")

    cleaned_data = data.copy()

    # Check for missing values
    missing_summary = cleaned_data.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]

    if len(missing_cols) > 0:
        logger.info("Missing value summary:")
        for col, count in missing_cols.items():
            pct = (count / len(cleaned_data)) * 100
            logger.info(f"  {col}: {count:,} ({pct:.1f}%)")

        # Strategy for handling missing values
        for col in missing_cols.index:
            if col.startswith("embedding_dim_"):
                # Fill embedding dimensions with 0
                cleaned_data[col] = cleaned_data[col].fillna(0)
            elif col in ["turn_number", "total_turns"]:
                # Fill turn numbers with 1 (first turn)
                cleaned_data[col] = cleaned_data[col].fillna(1)
            elif "proportion_" in col or "news_proportion_" in col:
                # Fill proportion columns with 0
                cleaned_data[col] = cleaned_data[col].fillna(0)
            elif col == "num_citations":
                # Fill citation count with 0
                cleaned_data[col] = cleaned_data[col].fillna(0)
            else:
                # For other numeric columns, use median
                if cleaned_data[col].dtype in ["float64", "int64"]:
                    median_val = cleaned_data[col].median()
                    cleaned_data[col] = cleaned_data[col].fillna(median_val)
                    logger.info(f"  Filled {col} with median: {median_val}")

    logger.info(f"Missing values after cleaning: {cleaned_data.isnull().sum().sum()}")

    return cleaned_data


def validate_cleaned_data(data):
    """Validate the cleaned dataset."""
    logger.info("Validating cleaned dataset...")

    # Check for remaining missing values
    missing_count = data.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Still have {missing_count} missing values")

    # Check for infinite values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(data[numeric_cols]).sum().sum()
    if inf_count > 0:
        logger.warning(f"Found {inf_count} infinite values")
        # Replace infinite values with NaN then fill
        data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    # Data type summary
    logger.info("Data type summary:")
    dtype_summary = data.dtypes.value_counts()
    for dtype, count in dtype_summary.items():
        logger.info(f"  {dtype}: {count} columns")

    # Dataset dimensions
    logger.info(f"Final dataset dimensions: {data.shape}")

    # Feature categories
    embedding_cols = [col for col in data.columns if col.startswith("embedding_dim_")]
    pca_cols = [col for col in data.columns if col.startswith("embedding_pc_")]
    dummy_cols = [
        col
        for col in data.columns
        if any(
            prefix in col
            for prefix in [
                "client_country_",
                "model_family_",
                "model_side_",
                "winner_",
                "primary_intent_",
                "secondary_intent_",
            ]
        )
    ]
    length_cols = [col for col in data.columns if col.endswith("_log")]
    citation_cols = [
        col
        for col in data.columns
        if col.startswith(("proportion_", "news_proportion_", "num_citations"))
    ]

    logger.info("Feature summary:")
    logger.info(f"  Original embedding features: {len(embedding_cols)}")
    logger.info(f"  PCA embedding features: {len(pca_cols)}")
    logger.info(f"  Dummy variables: {len(dummy_cols)}")
    logger.info(f"  Transformed length variables: {len(length_cols)}")
    logger.info(f"  Citation pattern features: {len(citation_cols)}")

    return data


def main():
    """Main function for feature cleaning and preprocessing."""
    try:
        # Get input/output paths from Snakemake
        input_path = snakemake.input[0]  # type: ignore
        output_path = snakemake.output[0]  # type: ignore

    except NameError:
        # Fallback for running outside Snakemake (for testing)
        logger.info("Running outside Snakemake - using default paths")
        base_dir = Path(__file__).parent.parent.parent.parent.parent

        input_path = (
            base_dir
            / "data/intermediate/question_analysis/integrated_analysis_data.parquet"
        )
        output_path = (
            base_dir / "data/intermediate/question_analysis/cleaned_features.parquet"
        )

    # Load integrated data
    data = load_integrated_data(input_path)

    # Step 1: Identify categorical variables
    categorical_vars = identify_categorical_variables(data)

    # Step 2: Create dummy variables
    data_with_dummies, dummy_columns = create_dummy_variables(data, categorical_vars)

    # Step 3: Transform length variables
    data_transformed, transformation_summary = transform_length_variables(
        data_with_dummies
    )

    # Step 4: Standardize embeddings
    data_standardized = standardize_embeddings(data_transformed)

    # Step 5: Apply PCA to embeddings
    data_with_pca = apply_pca_to_embeddings(data_standardized)

    # Step 6: Handle missing values
    data_cleaned = handle_missing_values(data_with_pca)

    # Step 7: Validate cleaned data
    final_data = validate_cleaned_data(data_cleaned)

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save cleaned dataset
    final_data.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(final_data):,} cleaned records to {output_path}")

    # Summary statistics
    logger.info("=== FEATURE CLEANING SUMMARY ===")
    logger.info(f"Input dataset: {len(data):,} rows, {len(data.columns)} columns")
    logger.info(
        f"Output dataset: {len(final_data):,} rows, {len(final_data.columns)} columns"
    )
    logger.info(f"Features added: {len(final_data.columns) - len(data.columns)}")
    logger.info(f"Output file size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Transformation summary
    if transformation_summary:
        logger.info("Length variable transformations:")
        for var, summary in transformation_summary.items():
            if "final_mean" in summary:
                logger.info(
                    f"  {var}: {summary['transformation']} "
                    f"(skewness: {summary['original_skewness']:.3f} → "
                    f"{summary['transformed_skewness']:.3f}, "
                    f"mean: {summary['final_mean']:.3f}, "
                    f"std: {summary['final_std']:.3f})"
                )
            else:
                logger.info(
                    f"  {var}: {summary['transformation']} "
                    f"(skewness: {summary['original_skewness']:.3f} → "
                    f"{summary['transformed_skewness']:.3f})"
                )

    logger.info("✅ Feature cleaning completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
