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

## Domain credibility data

`lin_domain_ratings.csv.gz` contains the credibility ratings of the domains.
The `domain` contains the domain name.
The `pc1` contains the domain credibility score.