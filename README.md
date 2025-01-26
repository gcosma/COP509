# COP509 -  Jewellery Reviews Analysis

## Overview
This repository contains a Jewellery product reviews dataset with associated analytical components for natural language processing tasks.

## Dataset Structure

### JewelleryReviewsLSA.csv
- 200 jewelleryry product reviews
- Columns:
  - ID: Unique identifier
  - Reviews: Review text
  - Ratings: Numerical rating

### vReviewsQueryRelevantID.csv
- Query1-8: Integer scores representing file IDs relevant to each query

### JewelleryReviewsSummarisationTargets.csv
- 5 entries mapping ratings to summaries
- Columns:
  - Ratings: Integer ratings
  - Targets: Summary text

### QueryText.csv.xlsx
- Query-related text data
- Complements query relevance scores

## Applications
- Sentiment analysis
- Text summarization
- Query relevance assessment
- Rating prediction
- Customer feedback analysis

## Technical Details
- File formats: CSV (UTF-8), Excel
- Dataset size:
  - Reviews: 200
  - Query relevance entries: 16
  - Summarization targets: 5

## Features
- Combined structured/unstructured data
- Pre-processed query relevance scores
- Rating-based summary targets

## Use Cases
Ideal for research in e-commerce review analysis, text summarization, and query relevance systems for jewelry/luxury goods.
