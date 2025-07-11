# Introduction

This directory contains the raw data for the project.

## Search arena data

### Full dataset
The file `search-arena-chat-24k.parquet` was downloaded from the following link:
https://huggingface.co/datasets/lmarena-ai/search-arena-24k

It contains the most comprehensive data for the search arena.
It's too large so we are not committing it to the repo.
Make sure to download it before running the code.

### Leaderboard data

The file `search-arena-v1-preference-7k.parquet` was downloaded from the following link: https://huggingface.co/datasets/lmarena-ai/search-arena-v1-7k

The file contains a subset of the full dataset with some additional information for each conversation.
The dataset was used to generate the public leaderboard.
We downloaded it in order to replicate the leaderboard and make sure the implementation is correct.

## DomainDemo political leaning data

`DomainDemo_political_leaning.csv.gz` contains the political leaning of the domains.
The `domain` contains the domain name.
The `leaning_score_users` contains the political leaning of the domain.

Please refer to the paper [DomainDemo: a dataset of domain-sharing activities among different demographic groups on Twitter](https://arxiv.org/abs/2501.09035) for more details.
The dataset can be downloaded from [here](https://github.com/LazerLab/DomainDemo).

## Domain credibility data

`lin_domain_ratings.csv.gz` contains the credibility ratings of the domains.
The `domain` contains the domain name.
The `pc1` contains the domain credibility score.

Please refer to the paper [High level of correspondence across different news domain quality rating sets](https://academic.oup.com/pnasnexus/article/2/9/pgad286/7258994) for more details.

## Domain classification data

### Manual classification

`domain_classification_manual.csv` contains the manual classification of some most popular domains from the citation data.
It contains two columns: `domain` and `classification`.

The classifications include:

- `social_media`: social media platforms (e.g. facebook, twitter, instagram, etc.)
- `wiki`: all types of encyclopedia (e.g. wikipedia, wikidata, etc.)
- `news`: news outlets
- `tech`: tech and coding platforms (e.g. github, stackoverflow, etc.)
- `community_blog`: community blogs (e.g. medium, substack, etc.)
- `gov_edu`: government and education institutions (those that end with `.gov` or `.edu`)
- `other`: other domains that don't fit into the other categories

### List of news sources

`list_of_news_domains.txt` contains a list of news domains.

The dataset was downloaded from https://github.com/yang3kc/list_of_news_domains.
