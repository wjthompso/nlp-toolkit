import pandas as pd
from bertopic import BERTopic
from textblob import TextBlob
import flair
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch import embedding, positive
try:
    from nlp_tools.keyword_extractor import KeywordExtractor
except:
    from keyword_extractor import KeywordExtractor
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import copy


class SentimentAnalyzer:
    """
    SentimentAnalyzer is a class that analyzes the sentiment of a series of documents.
    It supports different sentiment analysis classifiers and contains methods to generate plots
    to visualize the sentiment scores.

    Attributes:
        documents (list or pd.Series): A list or pandas Series of documents to be analyzed.
    """
    def __init__(self, documents) -> None:
        """
        Initializes the SentimentAnalyzer with a list or a pandas.Series of documents.

        Args:
            documents (list or pd.Series): A list or pandas Series of documents to be analyzed.
        """
        self.documents = documents

        if not isinstance(documents, list) and not isinstance(documents, pd.Series):
            raise Exception(
                f"You must pass in a list or a pandas.Series object. You passed in an object of type {type(documents)}")

        for idx, document in enumerate(self.documents):
            if type(document) != str:
                raise Exception(
                    f"All elements must be of type str. Element at idx = {idx} was of type {type(document)}")

        self.polarities = None
        self.last_sentiment_analyzer_used = None

    def get_polarity_scores(self, sentiment_classifier="Flair"):
        """
        Calculates polarity scores for the documents using either the TextBlob or Flair classifier.

        Args:
            sentiment_classifier (str, optional): The classifier to use. Must be either "Flair" or "TextBlob".
                Defaults to "Flair".
        """

        if sentiment_classifier not in ["Flair", "TextBlob"]:
            raise Exception(
                f"sentiment_classifier must be one of 'Flair', 'TextBlob' or ... Not {sentiment_classifier}")

        # Use the Flair library to calculate the polarity scores
        if sentiment_classifier == "Flair":
            polarities = []
            flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
            for document in self.documents:
                s = flair.data.Sentence(document)
                flair_sentiment.predict(s)
                total_sentiment = self._convert_flair_polarity_labels(
                    s.labels[0])
                polarities.append(total_sentiment)

            self.polarities = polarities
            self.last_sentiment_analyzer_used = "Flair"

        # Use the TextBlob library to calculate the polarity scores
        if sentiment_classifier == "TextBlob":
            polarities = []
            for document in self.documents:
                text_blob = TextBlob(document)
                polarity = text_blob.sentiment.polarity
                polarities.append(polarity)

            self.polarities = polarities
            self.last_sentiment_analyzer_used = "TextBlob"

    def _convert_flair_polarity_labels(self, text_label):
        """Convert the flair polarity labels to positive or negative floats.

        Example:

        'POSITIVE (0.98)' [str] -> 0.98 [int]
        'NEGATIVE (1.0)' [str] -> -1.0 [int]

        Args:
            text_label (str): A string representation of the Flair polarity label.

        Returns:
            float: The polarity score as a float.
        """
        if isinstance(text_label, flair.data.Label):
            text_label = str(text_label)

        polarity_score = float(
            re.search(r"(?<=\()[0-9.]*", text_label).group())

        if "NEGATIVE" in text_label.split(" ")[0]:
            polarity_score = (-1) * polarity_score

        return polarity_score

    def _convert_text_blob_polarity_scores(self, lower_neutral=0.2, upper_neutral=0.3):
        """
        Converts TextBlob polarity scores to categories: Negative, Neutral, and Positive.

        Args:
            lower_neutral (float, optional): The lower bound for the neutral category. Defaults to 0.2.
            upper_neutral (float, optional): The upper bound for the neutral category. Defaults to 0.3.
        """
        if self.last_sentiment_analyzer_used == "TextBlob":
            polarity_ranking = []
            for polarity_score in self.polarities:
                if polarity_score < 0.2:
                    polarity_ranking.append("Negative")
                if 0.2 <= polarity_score < 0.4:
                    polarity_ranking.append("Neutral")
                if polarity_score >= 0.4:
                    polarity_ranking.append("Positive")

            self.polarity_ranking = polarity_ranking

    def create_basic_plot(self):
        """
        Creates a basic histogram plot of the sentiment polarity scores.
        """
        if self.last_sentiment_analyzer_used == "TextBlob":
            plt.figure(figsize=(15, 12))
            sns.histplot(self.polarities, kde=True, color='darkblue', bins=15)
            plt.title('Sentiment Scores, Negative to Positive', fontsize=18)
            plt.xlabel(
                'Polarity Scores (-1: Negative, <0.2: Neutral, 1: Positive)', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.ylabel('Frequency', fontsize=16)

        if self.last_sentiment_analyzer_used == "Flair":
            plt.figure(figsize=(15, 12))
            sns.histplot(self.polarities, color='blue', bins=5)
            plt.title('Sentiment Scores, Negative to Positive', fontsize=18)
            plt.xlabel(
                'Polarity Scores (-1: Negative, <0.2: Neutral, 1: Positive)', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.ylabel('Frequency', fontsize=16)

    def create_color_differentiated_polarity_plot(self,
                                                  lower_neutral=0,
                                                  upper_neutral=0.05,
                                                  negative_color="red",
                                                  neutral_color="gray",
                                                  positive_color="blue"):
        """
        Creates a histogram plot of the sentiment polarity scores with color differentiation for
        negative, neutral, and positive scores.

        Args:
            lower_neutral (float, optional): The lower bound for the neutral category. Defaults to 0.
            upper_neutral (float, optional): The upper bound for the neutral category. Defaults to 0.05.
            negative_color (str, optional): The color for negative scores. Defaults to "red".
            neutral_color (str, optional): The color for neutral scores. Defaults to "gray".
            positive_color (str, optional): The color for positive scores. Defaults to "blue".
        """
        # plt parameters
        plt.rcParams['figure.figsize'] = (10.0, 10.0)
        plt.style.use('seaborn-dark-palette')
        plt.rcParams['axes.grid'] = True
        plt.rcParams["patch.force_edgecolor"] = True

        # Set bins based on the analyzer used
        if self.last_sentiment_analyzer_used == "TextBlob":
            bins = 10
        elif self.last_sentiment_analyzer_used == "Flair":
            bins = 4

        # This creates the plot
        p = sns.histplot(self.polarities, bins=bins, stat='count')

        # Set negative polarity scores to the negative color
        # neutral scores to the neutral, and so on.
        for rectangle in p.patches:
            if rectangle.get_x() < lower_neutral:
                rectangle.set_facecolor(negative_color)
            if lower_neutral <= rectangle.get_x() < upper_neutral:
                rectangle.set_facecolor(neutral_color)
            if rectangle.get_x() >= upper_neutral:
                rectangle.set_facecolor(positive_color)

        # add cosmetics
        plt.ylabel('Number of reviews', fontsize=16)
        plt.xlabel(
            'Polarity scores ranging from -1 to 1, negative to positive respectively.', fontsize=16)
        plt.show()

class WebscrapeAnalysisManager:
    """A class for managing web-scraped data analysis, such as generating plots and analyzing dataframes.

    This class is designed to handle the code written in a Jupyter notebook during exploratory data analysis,
    providing methods for sentiment analysis, keyword extraction, and data manipulation.

    Some of this code was written in a Jupyter notebook, and is included here for convenience. Although
    the Jupyter notebook is not included in this repository because it contains sensitive information.
    """
    def __init__(self, filename=None, input_df=None):
        """Initializes the WebscrapeAnalysisManager with a filename or a pandas DataFrame.

        Args:
            filename (str, optional): The path to the input CSV file.
            input_df (pd.DataFrame, optional): The input pandas DataFrame.
        """
        if (not filename or not isinstance(filename, str)) and not isinstance(input_df, pd.DataFrame):
            raise Exception(
                f"Please provide a valid filename! {type(filename)} not supported. Or, pass a pandas DataFrame.")

        if isinstance(input_df, pd.DataFrame):
            self.main_df = input_df
        else:
            self.main_df = pd.read_csv(filename)
        # For Yelp reviews
        if "customer_review_body" in self.main_df.columns:
            self.main_df["review_body"] = self.main_df["customer_review_body"]
            self.main_df = self.main_df.drop(columns="customer_review_body")

        if "pros" in self.main_df.columns and "cons" in self.main_df.columns and "review_body" not in self.main_df.columns:
            self.main_df["review_body"] = self.main_df["pros"]
            self.pros_cons_or_title = "pros"

        # For Glassdoor reviews
        self.main_df = self.main_df.fillna("")
        self.main_df = self.main_df.query(
            "review_body != ''").reset_index(drop=True)
        try:
            self.main_df = self.main_df.drop(columns=["Unnamed: 0", "index"])
        except:
            pass
        self.output_df = None
        self.input_df = copy.copy(self.main_df) # May lead to memory issues down the line.
        self.SentimentAnalyzer = SentimentAnalyzer(self.main_df["review_body"])

        if "review_body" not in self.main_df.columns:
            raise Exception(
                "Please pass in a dataframe with a column titled 'review_body' with text.")

    def change_column_to_analyze(self, column="pros"):
        """Changes the column to analyze in the DataFrame.
        
        If you're analyzing Glassdoor or Facebook Pro or Cons lists, use this to choose "pros", "cons",
        "advice_to_management", "review_title", or "review_body".

        Args:
            column (str, optional): The name of the column to analyze.
        """
        # if column not in ["pros", "cons", "advice_to_management", "review_body", "review_title"]:
        #     raise Exception(
        #         "The argument column must be one of 'pros', 'cons', 'advice_to_management', or 'review_title'")

        self.main_df["review_body"] = self.main_df[column]
        self.pros_cons_or_title = column

    def get_polarity_scores(self, sentiment_classifiers=["Flair", "TextBlob"]):
        """Calculates polarity scores for the text in the DataFrame using specified sentiment classifiers.

        Optional values for sentiment_classifiers are "Flair" and "TextBlob".

        Args:
            sentiment_classifiers (list, optional): A list of sentiment classifiers to use.
        """
        sentiment_analyzer = SentimentAnalyzer(self.main_df["review_body"])
        for sentiment_classifier in sentiment_classifiers:
            if sentiment_classifier == "TextBlob":
                sentiment_analyzer.get_polarity_scores(
                    sentiment_classifier=sentiment_classifier)
                self.main_df["textblob_polarities"] = sentiment_analyzer.polarities
            if sentiment_classifier == "Flair":
                sentiment_analyzer.get_polarity_scores(
                    sentiment_classifier=sentiment_classifier)
                self.main_df["flair_polarities"] = sentiment_analyzer.polarities

    def generate_polarity_score_plots(self, sentiment_classifiers=["Flair", "TextBlob"]):
        """Generates polarity score plots for the text in the DataFrame using specified sentiment classifiers.

        Args:
            sentiment_classifiers (list, optional): A list of sentiment classifiers to use.
        """
        sentiment_analyzer = SentimentAnalyzer(self.main_df["review_body"])
        sentiment_analyzer.get_polarity_scores(sentiment_classifier="Flair")
        sentiment_analyzer.create_color_differentiated_polarity_plot()
        sentiment_analyzer.get_polarity_scores(sentiment_classifier="TextBlob")
        sentiment_analyzer.create_color_differentiated_polarity_plot()

    def save_csv_of_df_with_polarity_scores(self, sentiment_classifiers=["Flair", "TextBlob"]):
        """Saves the DataFrame with polarity scores to a CSV file in the output folder.

        Args:
            sentiment_classifiers (list, optional): A list of sentiment classifiers to use.
        """
        # Create and add the pandas series of TextBlob polarities to the original Dataframe
        for sentiment_classifier in sentiment_classifier:
            self.get_polarity_scores(sentiment_classifier=sentiment_classifier)
            textblob_polarities = self.polarities
            self.main_df["textblob_polarities"] = textblob_polarities

        self.main_df.to_csv("./output/polarity_df.csv")

    def extract_suggestion_candidates(self, suggest_keywords=None):
        """Extracts sentences containing suggestions from the DataFrame using specified keywords.

        Look at the reviews and look for sentences that have 'should', 'recommend', or 'suggest'.
        
        Args:
            suggest_keywords (list, optional): A list of keywords to use for identifying suggestions.
        """
        suggestion_candidates_list = []

        if not suggest_keywords:
            suggest_keywords = ["should", "could", "recommend", "suggest"]

        for document in self.main_df["review_body"]:
            suggestion_candidates = ""
            for suggest_keyword in suggest_keywords:
                if suggest_keyword in document:
                    document = document.replace("?", ".").replace("!", ".")
                    for sentence in document.split("."):
                        if suggest_keyword in sentence:
                            suggestion_candidates += sentence
            suggestion_candidates_list.append(suggestion_candidates)

        suggestion_candidates = []
        for candidate in suggestion_candidates_list:
            if candidate != "":
                suggestion_candidates.append(candidate)
        return sorted(set(suggestion_candidates))

    def extract_keyphrases(self, sort_values=False):
        """Extracts keyphrases from the DataFrame and sorts them by polarity, if specified.

        Args:
            sort_values (bool, optional): Whether to sort the DataFrame by polarity scores.
        """
        # Allow all of the text to be displayed
        pd.set_option('display.max_colwidth', 0)

        facebook_keyword_extraction = KeywordExtractor(
            list(self.main_df["review_body"]), create_matrix=False, ngram_range=(1, 3))
        facebook_keyword_extraction.extract_keywords(
            extractors=["keybert", "rake", "yake", "slow_manual_bert"])

        self.main_df["yake_topics"] = facebook_keyword_extraction.output_df['yake_topics']
        self.main_df["rake_topics"] = facebook_keyword_extraction.output_df['rake_topics']
        self.main_df["slow_manual_bert_topics"] = facebook_keyword_extraction.output_df['slow_manual_bert_topics']

        if sort_values:
            self.main_df = self.main_df.sort_values(by="textblob_polarities")

    def bertopic_topic_label_predictions(self, 
                                         ngram_range=(1, 6), 
                                         min_topic_size=10, 
                                         num_topics=None,
                                         embedding_model=None):
        """Uses BERTopic to generate topic labels for the text in the DataFrame.

        Args:
            ngram_range (tuple, optional): The range of n-grams to consider for topic modeling.
            min_topic_size (int, optional): The minimum topic size for BERTopic.
            num_topics (int, optional): The number of topics for BERTopic.
            embedding_model (str, optional): The embedding model to use for BERTopic.
        """
        if ngram_range is not None and num_topics is not None:
            vectorizer_model = CountVectorizer(
                ngram_range=ngram_range, stop_words="english")
            topic_model = BERTopic(vectorizer_model=vectorizer_model,
                                   min_topic_size=min_topic_size, 
                                   nr_topics=num_topics,
                                   embedding_model=embedding_model)

        elif ngram_range is None and num_topics is not None:
            topic_model = BERTopic(nr_topics=num_topics,
                                   min_topic_size=min_topic_size,
                                   embedding_model=embedding_model)

        elif ngram_range is not None and num_topics is None:
            vectorizer_model = CountVectorizer(
                ngram_range=ngram_range, stop_words="english")
            topic_model = BERTopic(vectorizer_model=vectorizer_model, 
                                   min_topic_size=min_topic_size,
                                   embedding_model=embedding_model)

        elif ngram_range is None and num_topics is None:
            topic_model = BERTopic(min_topic_size=min_topic_size,
                                   embedding_model=embedding_model)

        self.bertopic_assignments, self.bertopic_probs = topic_model.fit_transform(
            list(self.main_df["review_body"]))
        self.bertopic_model = topic_model

    def common_keyphrases(self, filter=None, ascending=False):
        """Finds common keyphrases and their frequency in the DataFrame, optionally filtering by polarity.

        Args:
            filter (str, optional): The polarity filter, either "negative", "positive", or None.
            ascending (bool, optional): Whether to sort the keyphrases in ascending order of frequency.
        """
        manipulation_df = copy.copy(self.main_df)

        if filter == "negative":
            manipulation_df = manipulation_df.query("flair_polarities <= 0")
        if filter == "positive":
            manipulation_df = manipulation_df.query("flair_polarities >= 0")

        # Create a new pandas series with a set of the keywords and keyphrases extracted
        manipulation_df["keyphrase_list"] = manipulation_df["yake_topics"] + ", " + \
            manipulation_df["rake_topics"] + ", " + \
            manipulation_df["slow_manual_bert_topics"]
        manipulation_df["set_of_keywords"] = [set(keyphrase_list.split(", ")).union(set(
            keyphrase_list.replace(",", "").split(" "))) for keyphrase_list in manipulation_df["keyphrase_list"]]

        # Create a master set of all keywords and keyphrases
        master_set = set()
        for set_of_keywords in manipulation_df["set_of_keywords"]:
            master_set = master_set.union(set_of_keywords)

        # Create a dictionary of occurance counts by iterating through each document's set.
        master_dict = {key: 0 for key in master_set}

        for set_of_keywords in manipulation_df["set_of_keywords"]:
            for keyword in set_of_keywords:
                master_dict[keyword] += 1

        resulting_series = pd.Series(
            master_dict).sort_values(ascending=ascending)
        if not filter:
            self.common_words = resulting_series
        if filter == "negative":
            self.negative_words = resulting_series
        if filter == "positive":
            self.positive_words = resulting_series
        return resulting_series
    
    def reset_main_df(self, target_column=None):
        """Resets the main DataFrame to the original input DataFrame.

        Args:
            target_column (str, optional): The name of the target column to reset in the main DataFrame.
        """
        if target_column is None:
            raise ValueError("target_column must be specified.")

        self.main_df = self.input_df[target_column]

    def assign_topic_label_candidates_to_df(self):
        """Assigns topic label candidates to the main DataFrame."""
        self.bertopic_assignments = ["" if assignment == -1 else assignment for assignment in self.bertopic_assignments]
        formatted_topics = {}
        formatted_topics[""] = "No Topic"

        for topic_idx in self.bertopic_model.topics:
            formatted_topics[topic_idx] = str(topic_idx) + ": " + ", ".join([label_candidate[0] for label_candidate in self.bertopic_model.topics[topic_idx]][0:5])

        # Now we have the list of topics, we can 
        topic_list = [formatted_topics[topic_idx] for topic_idx in self.bertopic_assignments]
        self.main_df["topic_label_candidates"] = topic_list

class AnalysisManager(WebscrapeAnalysisManager):
    """This class is used to do more general analysis on the data.
    
    NOTE: Consider deprecating this class in favor of WebscrapeAnalysisManager.
    """
    def __init__(self, input_df=None, target_column=None, filename=None, webscrape=False):
        if webscrape:
            super().__init__(filename=filename, input_df=input_df)
            return None

        # Check that the input is valid
        if (not filename or not isinstance(filename, str)) and not isinstance(input_df, pd.DataFrame):
            raise Exception(
                f"Please provide a valid filename! {type(filename)} not supported. Or, pass a pandas DataFrame.")

        if isinstance(input_df, pd.DataFrame):
            self.main_df = input_df
        else:
            self.main_df = pd.read_csv(filename)

        # Set review_body to the target column
        if target_column is None:
            raise ValueError("target_column must be specified.")
        else:
            self.main_df["review_body"] = self.main_df[target_column]   
        
        self.input_df = copy.copy(self.main_df)
        self.output_df = None
        self.SentimentAnalyzer = SentimentAnalyzer(self.main_df["review_body"])


if __name__ == "__main__":
    analysis_manager = WebscrapeAnalysisManager(
        filename="./datasets/glassdoor_aba_scrape-2022-3-1.csv")
    analysis_manager.main_df = analysis_manager.main_df.query("pros != ''")
    analysis_manager.extract_keyphrases()
