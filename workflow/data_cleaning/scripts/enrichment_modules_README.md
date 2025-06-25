# Citation Enrichment Modules

This directory contains modular scripts for enriching citations with various signals. The design uses a 3-phase approach for efficiency and modularity.

## Architecture

### 3-Phase Enrichment Pipeline

1. **Domain Extraction** (`extract_domains.py`)
   - Extracts unique domains from citations with frequencies
   - Creates intermediate `domains.parquet` file
   - Avoids repeated loading of large citations dataset

2. **Domain Enrichment** (`enrich_domains_combined.py`)
   - Enriches unique domains with all signals
   - Much more efficient than enriching full citations
   - Creates `domains_enriched.parquet`

3. **Citation Merging** (`merge_enriched_citations.py`)
   - Merges enriched domains back to citations
   - Creates final `citations_enriched.parquet`

### Individual Signal Scripts
Each signal has its own dedicated script that follows a standard pattern:

- **`enrich_political_leaning.py`** - Adds political leaning scores and categories
- **`enrich_domain_quality.py`** - Adds domain quality scores and categories

### Legacy Scripts
- **`enrich_citations.py`** - Original monolithic script
- **`enrich_citations_combined.py`** - Direct citation enrichment (inefficient)

## Standard Signal Script Pattern

Each signal script follows this structure:

```python
def load_[signal]_data(filepath):
    """Load and process [signal] data."""
    # Load data from source
    # Clean and process
    # Create categorical variables if needed
    # Return processed dataframe

def enrich_with_[signal](citations_df, signal_data):
    """Merge citations with [signal] data."""
    # Merge citations with signal data
    # Report coverage statistics
    # Return enriched dataframe

def main():
    """Main function for standalone execution."""
    # Can be run independently for testing
```

## Adding New Signals

To add a new signal:

1. **Create signal script** (`enrich_new_signal.py`):
   ```python
   def load_new_signal_data(filepath):
       # Load and process new signal data
       pass
   
   def enrich_with_new_signal(citations_df, signal_data):
       # Merge new signal with citations
       pass
   ```

2. **Update orchestrator** (`enrich_citations_combined.py`):
   ```python
   import enrich_new_signal
   
   # In main():
   new_signal_data = enrich_new_signal.load_new_signal_data(new_signal_path)
   enriched = enrich_new_signal.enrich_with_new_signal(enriched, new_signal_data)
   ```

3. **Update Snakefile** if new input files are needed:
   ```python
   rule enrich_citations:
       input:
           citations=f"{INTERMEDIATE_DIR}/citations.parquet",
           political_leaning=f"{RAW_DATA_DIR}/DomainDemo_political_leaning.csv.gz",
           domain_ratings=f"{RAW_DATA_DIR}/lin_domain_ratings.csv.gz",
           new_signal=f"{RAW_DATA_DIR}/new_signal_data.csv"  # Add new input
   ```

## Benefits

- **Efficiency**: Only loads large citations dataset once, enriches smaller domains dataset
- **Modularity**: Each signal can be developed, tested, and maintained independently
- **Reusability**: Signal scripts can be used standalone for testing or analysis
- **Scalability**: Easy to add new signals without modifying existing code
- **Maintainability**: Clear separation of concerns and standardized patterns
- **Testing**: Individual signals can be unit tested separately
- **Performance**: 3-phase approach scales better with dataset size

## Current Signals

### Political Leaning Signal
- **Source**: `DomainDemo_political_leaning.csv.gz`
- **Continuous**: `political_leaning_score` (from `leaning_score_users`)
- **Categorical**: `political_leaning` (left_leaning/right_leaning/unknown)
- **Threshold**: 0 (negative = left_leaning, positive/zero = right_leaning)

### Domain Quality Signal  
- **Source**: `lin_domain_ratings.csv.gz`
- **Continuous**: `domain_quality_score` (from `pc1`)
- **Categorical**: `domain_quality` (high_quality/low_quality/unknown)
- **Threshold**: 0.7 (≥0.7 = high_quality, <0.7 = low_quality)

## Usage

The 3-phase pipeline is executed by Snakemake:
```bash
# Run the complete enrichment pipeline
snakemake enrich_citations --cores 1

# Or run individual phases
snakemake extract_domains --cores 1     # Phase 1: Extract domains
snakemake enrich_domains --cores 1      # Phase 2: Enrich domains
snakemake enrich_citations --cores 1    # Phase 3: Merge back to citations
```

Individual signal scripts can also be run standalone for testing (requires manual path setup).

## Pipeline Flow

```
citations.parquet 
    ↓ extract_domains.py
domains.parquet (unique domains + frequencies)
    ↓ enrich_domains_combined.py (calls individual signal modules)
domains_enriched.parquet (domains + all signals)
    ↓ merge_enriched_citations.py
citations_enriched.parquet (final enriched citations)
```