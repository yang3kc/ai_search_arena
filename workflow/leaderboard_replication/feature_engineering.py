"""
Feature engineering functions to transform normalized data into leaderboard format.
"""

import pandas as pd


# Domain categorization mappings
DOMAIN_CATEGORIES = {
    'youtube': ['youtube.com', 'youtu.be'],
    'gov_edu': ['.gov', '.edu', 'whitehouse.gov', 'congress.gov', 'senate.gov'],
    'wiki': ['wikipedia.org', 'wikimedia.org', 'wikidata.org'],
    'us_news': [
        'cnn.com', 'foxnews.com', 'nytimes.com', 'washingtonpost.com', 
        'wsj.com', 'usatoday.com', 'nbcnews.com', 'cbsnews.com', 'abcnews.go.com',
        'npr.org', 'apnews.com', 'reuters.com', 'politico.com', 'bloomberg.com'
    ],
    'foreign_news': [
        'bbc.com', 'bbc.co.uk', 'theguardian.com', 'independent.co.uk',
        'telegraph.co.uk', 'dailymail.co.uk', 'aljazeera.com', 'france24.com',
        'dw.com', 'rt.com', 'xinhuanet.com', 'thetimes.co.uk'
    ],
    'social_media': [
        'facebook.com', 'twitter.com', 'x.com', 'instagram.com', 'linkedin.com',
        'tiktok.com', 'snapchat.com', 'pinterest.com', 'reddit.com'
    ],
    'community_blog': [
        'medium.com', 'substack.com', 'wordpress.com', 'blogspot.com',
        'tumblr.com', 'quora.com', 'dev.to'
    ],
    'tech_coding': [
        'github.com', 'stackoverflow.com', 'stackexchange.com', 'gitlab.com',
        'bitbucket.org', 'codepen.io', 'codesandbox.io', 'repl.it'
    ],
    'map': [
        'maps.google.com', 'google.com/maps', 'openstreetmap.org', 
        'mapquest.com', 'bing.com/maps'
    ],
    'academic_journal': [
        'pubmed.ncbi.nlm.nih.gov', 'scholar.google.com', 'arxiv.org',
        'jstor.org', 'springer.com', 'elsevier.com', 'wiley.com',
        'nature.com', 'science.org', 'plos.org'
    ]
}


def categorize_domain(domain: str) -> str:
    """Categorize a domain into one of the predefined categories."""
    if pd.isna(domain) or not domain:
        return 'other'
    
    domain = domain.lower().strip()
    
    for category, patterns in DOMAIN_CATEGORIES.items():
        for pattern in patterns:
            if pattern.startswith('.'):
                # TLD match (e.g., .gov, .edu)
                if domain.endswith(pattern):
                    return category
            elif domain == pattern or domain.endswith('.' + pattern):
                return category
    
    return 'other'


def compute_citation_metrics(citations_df: pd.DataFrame) -> pd.DataFrame:
    """Compute domain-specific citation metrics from citations table."""
    
    # Add domain categories
    citations_df = citations_df.copy()
    citations_df['domain_category'] = citations_df['domain'].apply(categorize_domain)
    
    # Group by response_id and compute metrics
    citation_metrics = []
    
    for response_id, group in citations_df.groupby('response_id'):
        metrics = {'response_id': response_id}
        
        # Total citations
        metrics['num_citations'] = len(group)
        
        # Domain category counts and indicators
        domain_counts = group['domain_category'].value_counts()
        
        for category in ['youtube', 'gov_edu', 'wiki', 'us_news', 'foreign_news',
                        'social_media', 'community_blog', 'tech_coding', 'map',
                        'academic_journal', 'other']:
            count = domain_counts.get(category, 0)
            metrics[f'num_cites_{category}'] = count
            metrics[f'cites_{category}'] = 1 if count > 0 else 0
        
        citation_metrics.append(metrics)
    
    return pd.DataFrame(citation_metrics)


def compute_response_metrics(responses_df: pd.DataFrame) -> pd.DataFrame:
    """Compute response-level metrics (length, etc.)."""
    
    responses_metrics = responses_df.copy()
    
    # Response length
    responses_metrics['response_length'] = responses_metrics['response_text'].str.len()
    
    # Standardized citations indicator
    responses_metrics['standardized_citations'] = (
        responses_metrics['citation_format'] == 'standardized'
    ).astype(int)
    
    return responses_metrics[['response_id', 'question_id', 'thread_id', 'turn_number',
                             'model_name_llm', 'model_side', 'response_length', 
                             'standardized_citations']]


def create_conversation_metadata(
    threads_df: pd.DataFrame,
    responses_df: pd.DataFrame, 
    citation_metrics_df: pd.DataFrame,
    response_metrics_df: pd.DataFrame
) -> pd.DataFrame:
    """Create conversation-level metadata matching preference dataset format."""
    
    # Merge response metrics with citation metrics
    full_response_data = response_metrics_df.merge(
        citation_metrics_df, 
        on='response_id', 
        how='left'
    ).fillna(0)
    
    # Group by thread_id and aggregate by model side
    conversation_data = []
    
    for thread_id, thread_responses in full_response_data.groupby('thread_id'):
        thread_info = threads_df[threads_df['thread_id'] == thread_id].iloc[0]
        
        conv_data = {
            'thread_id': thread_id,
            'winner': thread_info['winner'],
            'judge': thread_info['judge'],
            'timestamp': thread_info['timestamp'],
            'total_turns': thread_info['total_turns'],
            'primary_intent': thread_info['primary_intent'],
            'secondary_intent': thread_info['secondary_intent'],
        }
        
        # Get model sides A and B
        side_a_responses = thread_responses[thread_responses['model_side'] == 'a']
        side_b_responses = thread_responses[thread_responses['model_side'] == 'b']
        
        if len(side_a_responses) > 0:
            conv_data['model_a'] = side_a_responses['model_name_llm'].iloc[0]
            
            # Aggregate metrics for side A
            conv_data['response_length_a'] = side_a_responses['response_length'].mean()
            conv_data['num_citations_a'] = side_a_responses['num_citations'].mean()
            conv_data['standardized_citations_a'] = side_a_responses['standardized_citations'].iloc[0]
            
            # Domain citation metrics for side A
            for category in ['youtube', 'gov_edu', 'wiki', 'us_news', 'foreign_news',
                            'social_media', 'community_blog', 'tech_coding', 'map',
                            'academic_journal', 'other']:
                conv_data[f'cites_{category}_a'] = side_a_responses[f'cites_{category}'].max()
                conv_data[f'num_cites_{category}_a'] = side_a_responses[f'num_cites_{category}'].sum()
        
        if len(side_b_responses) > 0:
            conv_data['model_b'] = side_b_responses['model_name_llm'].iloc[0]
            
            # Aggregate metrics for side B
            conv_data['response_length_b'] = side_b_responses['response_length'].mean()
            conv_data['num_citations_b'] = side_b_responses['num_citations'].mean()
            conv_data['standardized_citations_b'] = side_b_responses['standardized_citations'].iloc[0]
            
            # Domain citation metrics for side B
            for category in ['youtube', 'gov_edu', 'wiki', 'us_news', 'foreign_news',
                            'social_media', 'community_blog', 'tech_coding', 'map',
                            'academic_journal', 'other']:
                conv_data[f'cites_{category}_b'] = side_b_responses[f'cites_{category}'].max()
                conv_data[f'num_cites_{category}_b'] = side_b_responses[f'num_cites_{category}'].sum()
        
        # Only include conversations with both sides
        if 'model_a' in conv_data and 'model_b' in conv_data:
            conversation_data.append(conv_data)
    
    return pd.DataFrame(conversation_data)


def create_leaderboard_format(
    threads_df: pd.DataFrame,
    responses_df: pd.DataFrame,
    citations_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Transform normalized data into leaderboard format matching preference dataset.
    
    Returns DataFrame with columns:
    - model_a, model_b, winner, judge, timestamp, etc.
    - conv_metadata dict with all citation and response metrics
    """
    
    print("Computing citation metrics...")
    citation_metrics = compute_citation_metrics(citations_df)
    
    print("Computing response metrics...")
    response_metrics = compute_response_metrics(responses_df)
    
    print("Creating conversation metadata...")
    conversation_df = create_conversation_metadata(
        threads_df, responses_df, citation_metrics, response_metrics
    )
    
    print("Formatting for leaderboard...")
    # Create conv_metadata dict for each row
    leaderboard_data = []
    
    for _, row in conversation_df.iterrows():
        conv_metadata = {}
        
        # Extract conv_metadata fields
        for col in row.index:
            if col.endswith('_a') or col.endswith('_b'):
                conv_metadata[col] = row[col]
        
        leaderboard_row = {
            'model_a': row['model_a'],
            'model_b': row['model_b'], 
            'winner': row['winner'],
            'judge': row['judge'],
            'timestamp': row['timestamp'],
            'total_turns': row['total_turns'],
            'primary_intent': row['primary_intent'],
            'secondary_intent': row['secondary_intent'],
            'conv_metadata': conv_metadata,
            'thread_id': row['thread_id']
        }
        
        leaderboard_data.append(leaderboard_row)
    
    return pd.DataFrame(leaderboard_data)


def test_feature_engineering():
    """Test feature engineering with cleaned data."""
    
    print("Loading cleaned data...")
    threads_df = pd.read_parquet('/Users/yangkc/working/llm/ai_search_arena/data/intermediate/cleaned_arena_data/threads.parquet')
    responses_df = pd.read_parquet('/Users/yangkc/working/llm/ai_search_arena/data/intermediate/cleaned_arena_data/responses.parquet')
    citations_df = pd.read_parquet('/Users/yangkc/working/llm/ai_search_arena/data/intermediate/cleaned_arena_data/citations.parquet')
    
    print(f"Loaded {len(threads_df)} threads, {len(responses_df)} responses, {len(citations_df)} citations")
    
    # Test on subset for speed
    sample_threads = threads_df.head(100)
    sample_thread_ids = set(sample_threads['thread_id'])
    sample_responses = responses_df[responses_df['thread_id'].isin(sample_thread_ids)]
    sample_citations = citations_df[citations_df['response_id'].isin(sample_responses['response_id'])]
    
    print(f"Testing with {len(sample_threads)} threads, {len(sample_responses)} responses, {len(sample_citations)} citations")
    
    # Test citation categorization
    print("\nTesting domain categorization...")
    test_domains = ['youtube.com', 'github.com', 'nytimes.com', 'wikipedia.org', 'unknown-domain.com']
    for domain in test_domains:
        category = categorize_domain(domain)
        print(f"  {domain} -> {category}")
    
    # Test full pipeline
    print("\nTesting full feature engineering pipeline...")
    leaderboard_df = create_leaderboard_format(sample_threads, sample_responses, sample_citations)
    
    print(f"Created leaderboard format with {len(leaderboard_df)} conversations")
    print("Sample conv_metadata keys:", list(leaderboard_df['conv_metadata'].iloc[0].keys())[:10])
    
    # Check if we have the expected structure
    sample_conv = leaderboard_df.iloc[0]
    print(f"Sample conversation: {sample_conv['model_a']} vs {sample_conv['model_b']} -> {sample_conv['winner']}")
    
    return leaderboard_df


if __name__ == "__main__":
    test_feature_engineering()