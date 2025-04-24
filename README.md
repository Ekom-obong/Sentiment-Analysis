# Sentiment-Analysis
This project analyzes user reviews of the Uber mobile app to extract sentiment scores using both rule-based and transformer-based Natural Language Processing (NLP) models. 
## Project Overview

This project performs sentiment analysis on Uber reviews using two approaches: VADER (Valence Aware Dictionary and sEntiment Reasoner) and a pre-trained RoBERTa model from the Hugging Face Transformers library. The analysis is conducted on a dataset of 12,000 Uber reviews, focusing on the textual content and corresponding star ratings. The goal is to compare the sentiment scores from both models and identify discrepancies between model predictions and user-assigned ratings.

## Dataset

The dataset (uber_reviews_without_reviewid.csv) contains 12,000 Uber reviews with the following columns:





userName: Unique identifier for the reviewer.



content: Text of the review.



score: Star rating (1 to 5).



thumbsUpCount: Number of thumbs-up votes.



reviewCreatedVersion: App version when the review was created.



at: Timestamp of the review.



replyContent: Reply content (if any).



repliedAt: Timestamp of the reply (if any).



appVersion: App version.

## Requirements

To run the Jupyter notebook, install the required Python packages:

pip install pandas numpy matplotlib seaborn nltk transformers scipy tqdm

Additionally, download the necessary NLTK data:

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

## Project Structure





uber_reviews_without_reviewid.csv: Input dataset.



Sentiment_Analysis.ipynb: Jupyter notebook containing the analysis.



## Methodology





Data Loading and Preprocessing:





Load the dataset using pandas.



Inspect the data structure and clean it by handling missing values and creating unique identifiers.



VADER Sentiment Analysis:





Use NLTK's SentimentIntensityAnalyzer to compute polarity scores (neg, neu, pos, compound) for each review.



Merge VADER scores with the original dataset.



RoBERTa Sentiment Analysis:





Use the cardiffnlp/twitter-roberta-base-sentiment model to compute sentiment scores (roberta_neg, roberta_neu, roberta_pos).



Handle exceptions for reviews that cause runtime errors during processing.



Merge RoBERTa scores with the dataset.



Visualization:





Plot bar charts to compare sentiment scores (pos, neu, neg) across star ratings.



Create a pairplot to visualize relationships between VADER and RoBERTa scores, colored by star rating.



Save plots to files (e.g., compound_score.png, sentiment_comparison.png).



## Discrepancy Analysis:





Identify reviews with high positive sentiment but low star ratings (e.g., 1 star) and vice versa.



Highlight examples where model predictions differ significantly from user ratings.

## Key Findings





VADER vs. RoBERTa: RoBERTa captures contextual nuances better than VADER, as it accounts for word relationships, while VADER relies on a lexicon-based approach.



Discrepancies: Some reviews show high positive sentiment scores (e.g., "Rider is very good") but receive 1-star ratings, indicating potential sarcasm or unmet expectations. Similarly, negative reviews (e.g., "One of the worst app") with 5-star ratings suggest possible errors or irony.



Visual Insights: The pairplot reveals that RoBERTa scores are more spread out, indicating higher sensitivity to context compared to VADER's more polarized scores.
