# Introduction

This directory contains the raw data for the project.

## Search arena data

The file `search-arena-chat-24k.parquet` was downloaded from the following link:
https://huggingface.co/datasets/lmarena-ai/search-arena-24k

It's too large so we are not committing it to the repo.
Make sure to download it before running the code.

## DomainDemo political leaning data

`DomainDemo_political_leaning.csv.gz` contains the political leaning of the domains.
The `domain` contains the domain name.
The `leaning_score_users` contains the political leaning of the domain.

## Domain credibility data

`lin_domain_ratings.csv.gz` contains the credibility ratings of the domains.
The `domain` contains the domain name.
The `pc1` contains the domain credibility score.