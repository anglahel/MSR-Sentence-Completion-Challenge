# MSR-Sentence-Completion-Challenge
## Petnica Summer Institute: Machine Learning Project (2019)
Next word recommendation model for sentence completion using Deep Structured Semantic Models.
Research was based on:
https://www.researchgate.net/publication/262289160_Learning_deep_structured_semantic_models_for_web_search_using_clickthrough_data  
trying to adopt same ideas applied to web search recommendations to sentence completion recommendations

Requirements:  
tensorflow>=1.7  
tf-hub  
nltk

## TODO: preprocess data with uniform distribution for sentences
data: https://drive.google.com/drive/folders/0B5eGOMdyHn2mWDYtQzlQeGNKa2s  
**Problem:** Current preprocessing method parses training data with distribution of sentance length far to the left (too short), while test data is distributed to the right
