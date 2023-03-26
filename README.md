This repository contains several classes to perform various tasks on a set of documents or images. Here is a brief description of each class:

- SimilarityMatrix: A class that creates a similarity matrix and a dataframe with the top n most similar documents per document from a corpus.
ThoughtGrouper: A class which groups similar sentences within each document in a corpus into thoughts and then flattens the results into a pandas DataFrame.
- WebscrapeAnalysisManager: A class for managing web-scraped data analysis, such as generating plots and analyzing dataframes. It includes a SentimentAnalyzer class that analyzes the sentiment of a series of documents.
- KeywordExtractor: A class which extracts keywords and phrases and candidate topic labels from a given set of documents.
- ImagePresentation: A class that takes a list of image file paths, uploads those images to Google Cloud Platform (GCP), obtains signed URLs for each image on GCP, and then uses those signed URLs to create a Google Slides presentation with each image on a different slide.
- HistogramPresentation: A class which takes a pandas DataFrame with columns to be converted to a series of histograms or a column chart.
This repository can be used for a variety of purposes, such as clustering similar documents, analyzing sentiment, extracting keywords and phrases, creating image presentations, and generating histograms.
