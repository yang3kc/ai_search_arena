"""
Citation Style Analysis - Focused on understanding source citation effects

This module analyzes how different citation sources and styles affect user preferences
in AI search conversations. Unlike the full leaderboard system, this focuses specifically
on style coefficients and their confidence intervals using SQL-style DataFrames.
"""

import pandas as pd
import numpy as np
import scipy.optimize as opt
from typing import List, Tuple, Optional, Dict
import warnings


class CitationStyleAnalyzer:
    """
    Analyzes citation style effects on user preferences using Bradley-Terry models
    with style controls, designed to work with normalized SQL-style data.
    """
    
    def __init__(self, anchor_model: str = "gpt-4o-search-preview", anchor_rating: float = 1000.0):
        self.anchor_model = anchor_model
        self.anchor_rating = anchor_rating
        
    def load_data(self, threads_path: str, responses_path: str, citations_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load normalized data tables."""
        threads_df = pd.read_parquet(threads_path)
        responses_df = pd.read_parquet(responses_path)
        citations_df = pd.read_parquet(citations_path)
        return threads_df, responses_df, citations_df
    
    def compute_citation_features(self, citations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute citation features per response using SQL-style operations.
        Returns DataFrame with response_id and citation features.
        """
        # First categorize domains
        citations_with_categories = self._categorize_domains(citations_df.copy())
        
        # Domain categories for analysis
        domain_categories = [
            'youtube', 'gov_edu', 'wiki', 'us_news', 'foreign_news',
            'social_media', 'community_blog', 'tech_coding', 'map', 
            'academic_journal', 'other'
        ]
        
        # Basic citation counts per response
        basic_counts = citations_with_categories.groupby('response_id').agg({
            'citation_id': 'count'
        }).rename(columns={'citation_id': 'num_citations'})
        
        # Domain-specific counts
        domain_counts = citations_with_categories.groupby(['response_id', 'domain_category']).size().unstack(fill_value=0)
        
        # Ensure all domain categories are present
        for domain in domain_categories:
            if domain not in domain_counts.columns:
                domain_counts[domain] = 0
                
        # Rename columns to match expected format
        domain_counts.columns = [f'cites_{col}' for col in domain_counts.columns]
        
        # Combine basic counts with domain counts
        features_df = basic_counts.join(domain_counts, how='left').fillna(0)
        
        # Add binary indicators for having citations from each domain
        for domain in domain_categories:
            features_df[f'has_{domain}'] = (features_df[f'cites_{domain}'] > 0).astype(int)
        
        # Reset index to make response_id a column
        features_df = features_df.reset_index()
        
        return features_df
    
    def _categorize_domains(self, citations_df: pd.DataFrame) -> pd.DataFrame:
        """Categorize domains into analysis categories."""
        def categorize_domain(domain):
            domain = domain.lower()
            
            # YouTube
            if 'youtube' in domain or 'youtu.be' in domain:
                return 'youtube'
            
            # Government and Educational
            if any(suffix in domain for suffix in ['.gov', '.edu', '.mil']):
                return 'gov_edu'
            
            # Wikipedia
            if 'wikipedia' in domain or 'wikimedia' in domain:
                return 'wiki'
            
            # US News
            us_news_domains = ['cnn.com', 'nytimes.com', 'washingtonpost.com', 'wsj.com', 
                             'usatoday.com', 'foxnews.com', 'npr.org', 'abc.com', 'cbs.com', 
                             'nbc.com', 'politico.com', 'huffpost.com', 'time.com', 'newsweek.com']
            if any(news_domain in domain for news_domain in us_news_domains):
                return 'us_news'
            
            # Foreign News
            foreign_news_domains = ['bbc.com', 'reuters.com', 'theguardian.com', 'aljazeera.com',
                                  'france24.com', 'dw.com', 'rt.com', 'tass.com']
            if any(news_domain in domain for news_domain in foreign_news_domains):
                return 'foreign_news'
            
            # Social Media
            social_domains = ['twitter.com', 'facebook.com', 'instagram.com', 'linkedin.com',
                            'tiktok.com', 'snapchat.com', 'pinterest.com', 'reddit.com']
            if any(social_domain in domain for social_domain in social_domains):
                return 'social_media'
            
            # Community/Blog
            blog_domains = ['medium.com', 'wordpress.com', 'blogspot.com', 'tumblr.com',
                          'quora.com', 'stackoverflow.com', 'stackexchange.com']
            if any(blog_domain in domain for blog_domain in blog_domains):
                return 'community_blog'
            
            # Tech/Coding
            tech_domains = ['github.com', 'gitlab.com', 'bitbucket.com', 'docker.com',
                          'kubernetes.io', 'apache.org', 'python.org', 'nodejs.org']
            if any(tech_domain in domain for tech_domain in tech_domains):
                return 'tech_coding'
            
            # Maps
            if any(map_term in domain for map_term in ['maps.google', 'openstreetmap', 'mapquest']):
                return 'map'
            
            # Academic Journals (common publishers)
            academic_domains = ['springer.com', 'elsevier.com', 'nature.com', 'science.org',
                              'ieee.org', 'acm.org', 'wiley.com', 'taylor', 'sage']
            if any(academic_domain in domain for academic_domain in academic_domains):
                return 'academic_journal'
            
            # Default to other
            return 'other'
        
        citations_df['domain_category'] = citations_df['domain'].apply(categorize_domain)
        return citations_df
    
    def create_battle_data(self, threads_df: pd.DataFrame, responses_df: pd.DataFrame, 
                          citation_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create battle-style data from normalized tables.
        Each row represents a conversation with model_a vs model_b comparison.
        """
        # Join responses with citation features
        responses_with_features = responses_df.merge(
            citation_features_df, on='response_id', how='left'
        ).fillna(0)  # Fill NaN for responses without citations
        
        # Create A/B pairs from threads
        battle_data = []
        
        for _, thread in threads_df.iterrows():
            thread_responses = responses_with_features[
                responses_with_features['thread_id'] == thread['thread_id']
            ]
            
            if len(thread_responses) < 2:
                continue  # Skip single-model conversations
                
            # Group by question to get model pairs
            for question_id in thread_responses['question_id'].unique():
                question_responses = thread_responses[
                    thread_responses['question_id'] == question_id
                ]
                
                if len(question_responses) != 2:
                    continue  # Skip if not exactly 2 responses
                    
                responses_list = question_responses.to_dict('records')
                model_a_response = responses_list[0]
                model_b_response = responses_list[1]
                
                # Create battle row
                battle_row = {
                    'thread_id': thread['thread_id'],
                    'question_id': question_id,
                    'model_a': model_a_response['model_name_llm'],
                    'model_b': model_b_response['model_name_llm'],
                    'winner': thread['winner']  # Thread-level winner
                }
                
                # Add features for both models
                feature_cols = [col for col in citation_features_df.columns if col != 'response_id']
                for col in feature_cols:
                    battle_row[f'{col}_a'] = model_a_response.get(col, 0)
                    battle_row[f'{col}_b'] = model_b_response.get(col, 0)
                    
                battle_data.append(battle_row)
                
        return pd.DataFrame(battle_data)
    
    def compute_style_coefficients(self, battle_df: pd.DataFrame, 
                                 style_features: List[str],
                                 num_bootstrap: int = 100) -> Dict:
        """
        Compute style coefficients and confidence intervals for citation features.
        
        Args:
            battle_df: Battle data with model pairs and features
            style_features: List of feature names to control for (without _a/_b suffix)
            num_bootstrap: Number of bootstrap samples for confidence intervals
            
        Returns:
            Dictionary with coefficients, confidence intervals, and statistics
        """
        # Filter out ties for cleaner analysis
        battle_clean = battle_df[~battle_df['winner'].isin(['tie', 'tie (bothbad)'])].copy()
        
        if len(battle_clean) == 0:
            raise ValueError("No valid battles found after filtering ties")
            
        # Prepare feature matrix for style control
        style_columns = []
        for feature in style_features:
            if f'{feature}_a' in battle_clean.columns and f'{feature}_b' in battle_clean.columns:
                style_columns.extend([f'{feature}_a', f'{feature}_b'])
            else:
                warnings.warn(f"Feature {feature} not found in battle data")
                
        if not style_columns:
            raise ValueError("No valid style features found")
            
        # Compute base coefficients
        coefficients, log_likelihood = self._fit_style_model(battle_clean, style_columns)
        
        # Bootstrap for confidence intervals
        bootstrap_coeffs = []
        for _ in range(num_bootstrap):
            try:
                # Resample battles
                bootstrap_sample = battle_clean.sample(n=len(battle_clean), replace=True)
                boot_coeffs, _ = self._fit_style_model(bootstrap_sample, style_columns)
                bootstrap_coeffs.append(boot_coeffs)
            except:
                continue  # Skip failed bootstrap samples
                
        bootstrap_coeffs = np.array(bootstrap_coeffs)
        
        # Compute confidence intervals
        ci_lower = np.percentile(bootstrap_coeffs, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_coeffs, 97.5, axis=0)
        
        # Organize results
        results = {
            'features': style_features,
            'coefficients': dict(zip(style_features, coefficients)),
            'confidence_intervals': {
                feature: {'lower': ci_lower[i], 'upper': ci_upper[i]} 
                for i, feature in enumerate(style_features)
            },
            'log_likelihood': log_likelihood,
            'n_battles': len(battle_clean),
            'bootstrap_samples': len(bootstrap_coeffs)
        }
        
        return results
    
    def _fit_style_model(self, battle_df: pd.DataFrame, style_columns: List[str]) -> Tuple[np.ndarray, float]:
        """
        Fit Bradley-Terry model with style controls using MLE.
        Returns coefficients and log-likelihood.
        """
        # Prepare data
        models = sorted(set(battle_df['model_a']) | set(battle_df['model_b']))
        model_to_idx = {model: i for i, model in enumerate(models)}
        
        n_models = len(models)
        n_style_features = len(style_columns) // 2  # Divide by 2 since we have _a and _b versions
        
        # Create design matrices
        battles = []
        outcomes = []
        style_diffs = []
        
        for _, row in battle_df.iterrows():
            model_a_idx = model_to_idx[row['model_a']]
            model_b_idx = model_to_idx[row['model_b']]
            
            battles.append((model_a_idx, model_b_idx))
            outcomes.append(1 if row['winner'] == 'model_a' else 0)
            
            # Compute style differences (model_a - model_b)
            style_diff = []
            for i in range(0, len(style_columns), 2):  # Step by 2 to get pairs
                feature_a = style_columns[i]
                feature_b = style_columns[i + 1]
                diff = row[feature_a] - row[feature_b]
                style_diff.append(diff)
            style_diffs.append(style_diff)
            
        battles = np.array(battles)
        outcomes = np.array(outcomes)
        style_diffs = np.array(style_diffs)
        
        # Optimization function
        def neg_log_likelihood(params):
            # Split parameters into model ratings and style coefficients
            ratings = params[:n_models]
            style_coeffs = params[n_models:]
            
            # Compute win probabilities
            rating_diffs = ratings[battles[:, 0]] - ratings[battles[:, 1]]
            style_effects = np.sum(style_diffs * style_coeffs, axis=1)
            logits = rating_diffs + style_effects
            
            # Bradley-Terry probabilities
            probs = 1 / (1 + np.exp(-logits))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)  # Numerical stability
            
            # Log-likelihood
            ll = np.sum(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))
            return -ll
        
        # Initial parameters
        initial_params = np.zeros(n_models + n_style_features)
        
        # Test initial likelihood
        initial_ll = neg_log_likelihood(initial_params)
        if not np.isfinite(initial_ll):
            warnings.warn("Initial log-likelihood is not finite - may indicate data issues")
            return np.zeros(n_style_features), float('-inf')
        
        # Optimize with multiple methods
        methods_to_try = ['BFGS', 'L-BFGS-B', 'Nelder-Mead']
        
        for method in methods_to_try:
            try:
                if method == 'L-BFGS-B':
                    # Add bounds to prevent extreme values
                    bounds = [(-10, 10)] * (n_models + n_style_features)
                    result = opt.minimize(neg_log_likelihood, initial_params, method=method, bounds=bounds)
                else:
                    result = opt.minimize(neg_log_likelihood, initial_params, method=method)
                
                if result.success and np.isfinite(result.fun):
                    style_coefficients = result.x[n_models:]
                    log_likelihood = -result.fun
                    return style_coefficients, log_likelihood
                    
            except Exception as e:
                warnings.warn(f"Optimization method {method} failed: {e}")
                continue
        
        # If all methods fail, try a simpler approach
        warnings.warn("All optimization methods failed - using fallback")
        return np.zeros(n_style_features), float('-inf')


def analyze_citation_effects(data_dir: str, output_dir: str, 
                           features_to_analyze: List[str],
                           num_bootstrap: int = 1000) -> Dict:
    """
    Main function to analyze citation style effects.
    
    Args:
        data_dir: Path to directory with normalized data tables
        output_dir: Path to save results
        features_to_analyze: List of citation features to analyze
        num_bootstrap: Number of bootstrap samples
        
    Returns:
        Analysis results dictionary
    """
    analyzer = CitationStyleAnalyzer()
    
    # Load data
    threads_df, responses_df, citations_df = analyzer.load_data(
        f"{data_dir}/threads.parquet",
        f"{data_dir}/responses.parquet", 
        f"{data_dir}/citations.parquet"
    )
    
    # Compute citation features
    citation_features = analyzer.compute_citation_features(citations_df)
    
    # Create battle data
    battle_df = analyzer.create_battle_data(threads_df, responses_df, citation_features)
    
    # Analyze each feature set
    results = {}
    for feature_name in features_to_analyze:
        if isinstance(feature_name, str):
            features = [feature_name]
        else:
            features = feature_name
            
        try:
            result = analyzer.compute_style_coefficients(
                battle_df, features, num_bootstrap=num_bootstrap
            )
            results[str(features)] = result
        except Exception as e:
            print(f"Failed to analyze {features}: {e}")
            
    return results


if __name__ == "__main__":
    # Example usage
    data_dir = "../../data/intermediate/cleaned_arena_data"
    output_dir = "../../data/output/citation_analysis"
    
    # Define features to analyze
    domain_features = [
        'cites_youtube', 'cites_gov_edu', 'cites_wiki', 'cites_us_news',
        'cites_foreign_news', 'cites_social_media', 'cites_community_blog',
        'cites_tech_coding', 'cites_academic_journal'
    ]
    
    features_to_analyze = [
        'num_citations',  # Total citation count effect
        domain_features   # Domain-specific effects
    ]
    
    results = analyze_citation_effects(data_dir, output_dir, features_to_analyze)
    
    # Print results
    for feature_set, result in results.items():
        print(f"\n=== Analysis for {feature_set} ===")
        print(f"Battles analyzed: {result['n_battles']}")
        print(f"Bootstrap samples: {result['bootstrap_samples']}")
        
        for feature in result['features']:
            coeff = result['coefficients'][feature]
            ci = result['confidence_intervals'][feature]
            print(f"{feature}: {coeff:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")