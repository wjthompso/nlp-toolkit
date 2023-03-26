from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from similarity_matrix import SimilarityMatrix
except:
    from nlp_tools.similarity_matrix import SimilarityMatrix
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import itertools
import time
import copy
from termcolor import cprint
from keybert import KeyBERT
from rake_nltk import Rake
import nltk
import ssl
import yake
from bertopic import BERTopic


"""
Most of the code comes from here: https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea

https://towardsdatascience.com/keyword-extraction-a-benchmark-of-7-algorithms-in-python-8a905326d93f
"""

class KeywordExtractor(SimilarityMatrix):
    """
    A class which extracts keywords and phrases and candidate topic labels.

    Args:
        df_or_list (pandas dataframe or list of documents): Input data to extract keywords and phrases from.
        autocorrect (bool): Whether to autocorrect spelling mistakes.
        document_column_name (str): Name of the column to use as the source of documents, if `df_or_list` is a dataframe.
        create_matrix (bool): Whether to create a similarity matrix for the input documents.
        top_n (int): Number of top keywords or phrases to extract from each document.
        ngram_range (tuple): Range of n-grams to use for extracting phrases, default is (2, 2).
        stop_words (str or list): List of stop words to use in keyword extraction, default is English stop words.
        extractors (list): List of keyword extraction methods to use. Available options: ['keybert', 'rake', 'yake', 'slow_manual_bert'].
    """
    def __init__(self, 
                df_or_list, 
                autocorrect=False, 
                document_column_name=None, 
                create_matrix=False, 
                top_n=5, 
                ngram_range=(2,2),
                stop_words="english",
                extractors=["keybert"]):
        super().__init__(df_or_list, autocorrect, document_column_name, create_matrix, top_n)

        self.top_n = top_n
        self.extractors = extractors
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.keywords_candidates = None
        self.topics = None
        self.last_algorithm_used = None
        self.output_df = None

        # This is all here to make sure the stopwords and punkt gets downloaded so nltk will work
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download("stopwords")
        nltk.download("punkt")

    def extract_keywords(self, extractors=None) -> None:
        """
        Extract the top keywords from each document using a given method.

        Args:
        extractors: list of keyword extraction methods to use, default is None.

        Returns:
        None
        """

        if not extractors:
            extractors = self.extractors

        for extractor in extractors:
            if extractor not in ["keybert", "rake", "yake", "slow_manual_bert"]:
                raise Exception("""Only 'keybert', 'rake', 'yake', and 'slow_manual_bert" are available options for extractors!""")

        df_dict = {"document": self.document_series}

        for extractor in extractors:
            if extractor == "keybert":
                self._extract_keywords_with_keybert()
                df_dict[extractor + "_topics"] = self.topics
                df_dict[extractor + "_topic_candidates"] = self.keywords_candidates
            if extractor == "rake":
                self._extract_keywords_with_rake()
                df_dict[extractor + "_topics"] = self.topics
                df_dict[extractor + "_topic_candidates"] = self.keywords_candidates
            if extractor == "yake":
                self._extract_keywords_with_yake()
                df_dict[extractor + "_topics"] = self.topics
                df_dict[extractor + "_topic_candidates"] = self.keywords_candidates
            if extractor == "slow_manual_bert":
                self._extract_keywords_directly()
                df_dict[extractor + "_topics"] = self.topics
                df_dict[extractor + "_topic_candidates"] = self.keyword_candidates

        self.output_df = pd.DataFrame(df_dict)

    def extract_topics(self, method="bertopic") -> None:
        """
        Use BERTopic to extract topics from the input documents.

        Args:
        method: string specifying the method to use for topic extraction, default is 'bertopic'.

        Returns:
        None
        """
        model = BERTopic(language="English")
        topic_assignments, probabilities = model.fit_transform(self.document_series)
        self.topic_identifiers = topic_assignments
        topic_identifiers = list(model.get_topic_freq()['Topic'])
        # Remove stop words for understandability
        topic_descriptions = {topic_idx: self._rounddown(model.get_topic(topic_idx)) if topic_idx != -1 else None for topic_idx in topic_identifiers}
        self.topic_descriptions = [topic_descriptions[topic_idx] for topic_idx in topic_assignments]

        if isinstance(self.output_df, pd.DataFrame):
            self.output_df["bertopic_topics"] = self.topic_descriptions

        else:
            df_dict = {"document": self.document_series,
                       "bertopic_topics": self.topic_descriptions}

            self.output_df = pd.DataFrame(df_dict)

    def to_csv(self, filename="") -> None:
        """
        Save the extracted topics and keywords to a CSV file.

        Args:
        filename: name of the CSV file to create, default is 'extracted_topics.csv'.

        Returns:
        None
        """
        if filename == "":
            self.output_df.to_csv("extracted_topics.csv")
        
        elif filename.endswith(".csv"):
            self.output_df.to_csv(filename)

        else:
            self.output_df.to_csv(filename)

    def _extract_keywords_with_keybert(self) -> None:
        """Extracts the top_n keywords from each document using KeyBERT and then also selects the highest ranked
        keyword.

        KeyBERT is a state-of-the-art method for extractive keyword generation that uses the BERT model to extract keywords
        from documents. It first extracts a set of candidate keywords for each document, then scores them using a fine-tuned
        BERT model to obtain a ranking of the most relevant keywords. This method also selects the highest ranked keyword
        for each document as a potential topic label.

        Args:
            None

        Returns:
            None
        """
        kw_model = KeyBERT()
        keyword_candidates = kw_model.extract_keywords(self.document_series, top_n=self.top_n, keyphrase_ngram_range=self.ngram_range, stop_words=None)
        self.keywords_candidates = [[keyword[0] for keyword in keywords] for keywords in keyword_candidates]
        self.topics = [keybert_keywords[-1][0] for keybert_keywords in keyword_candidates]
        self.last_algorithm_used = "keybert"

    def _extract_keywords_with_rake(self):
        """Extracts the top_n keywords from each document using Rake and then also selects the highest ranked keyword.

        RAKE (Rapid Automatic Keyword Extraction) is a well-known keyword extraction algorithm that extracts key phrases
        by splitting the text into words, then scoring phrases based on their frequency, co-occurrence and other factors. 
        Rake extracts the most significant phrases from the document based on the ranking of its phrases.

        Args:
            None

        Returns:
            None
        """
        r = Rake()

        rake_keywords = []
        rake_topics = []

        for document in self.document_series:
            r.extract_keywords_from_text(document)
            rake_keywords.append(r.ranked_phrases[0:self.top_n])
            try:
                rake_topics.append(r.ranked_phrases[0])
            except Exception as e:
                rake_topics.append(None)

        self.keywords_candidates = rake_keywords
        self.topics = rake_topics
        self.last_algorithm_used = "rake"

    def _extract_keywords_with_yake(self):
        """Extracts the top_n keywords from each document using YAKE and then also selects the highest ranked keyword.

        YAKE (Yet Another Keyword Extractor) is a simple and effective keyword extraction algorithm that extracts keywords
        from text by applying a combination of statistical and linguistic criteria. YAKE first tokenizes the text into 
        words, then filters out stop words and low-frequency terms. It then ranks the remaining words based on their
        importance and relevance to the document using statistical and linguistic measures. This method also selects the
        highest ranked keyword for each document as a potential topic label.

        Args:
            None

        Returns:
            None
        """
        yake_keywords = []
        yake_topics = []

        for document in self.document_series:
            keywords = self._extract_single_document_using_yake(document, top_n=self.top_n)
            yake_keywords.append(keywords)
            try:
                yake_topics.append(keywords[0])
            except:
                yake_topics.append("")

        self.keywords_candidates = yake_keywords
        self.topics = yake_topics
        self.last_algorithm_used = "yake"

    def _extract_keywords_directly(self):
        """
        Extracts the top_n keywords from each document by directly calculating BERT embeddings.

        This algorithm is significantly slower than other keyword extraction algorithms.

        Args:
        algo: string, the algorithm to use for calculating embeddings. Default is "bert".

        Returns:
        None

        WARNING: Slow performance!
        """
        ngram_range = self.ngram_range
        stop_words = self.stop_words

        # Extract *all* candidate n grams (words or phrases) from the corpus.
        count = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words).fit(self.document_series)
        candidates = count.get_feature_names_out()

        # Extract candidate words/phrases, aka, ngrams
        extracted_ngrams = [] # list of lists of extracted ngrams for each document
        for document in self.document_series:
            try:
                count = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words).fit([document])
                candidates = count.get_feature_names_out()
                extracted_ngrams.append(candidates)
            except:
                extracted_ngrams.append(["none"])

        # Save the lengths of the list of extracted ngrams for each document
        candidate_lengths = [len(list_i) for list_i in extracted_ngrams]

        # Create the embeddings for each of the documents and their respective lists of ngrams
        bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        doc_embeddings = bert_model.encode(self.document_series)
        candidates = list(itertools.chain.from_iterable(extracted_ngrams)) #Flattens the list of ngrams
        candidate_embeddings = bert_model.encode(candidates)

        # Partition the flattened list of embeddings for both the documents and their respective
        candidate_embeddings = self._repartition_list(candidate_embeddings, candidate_lengths)
        candidates = self._repartition_list(candidates, candidate_lengths)

        self.keyword_candidates = []
        self.topics = []
        self.last_algorithm_used = "slow_manual_bert"
        for idx in range(doc_embeddings.shape[0]):
            distances = cosine_similarity([doc_embeddings[idx]], candidate_embeddings[idx])
            keyword_candidates = [[candidates[idx][index] for index in distances.argsort()[i][-self.top_n:]] for i in range(distances.shape[0])][0]
            self.keyword_candidates.append(keyword_candidates)
            self.topics.append(keyword_candidates[-1])

    def _max_sum_sim(self, doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates):
        """
        Returns the top_n most representative keywords from a list of candidate keywords, given a document's embedding,
        by maximizing the sum of cosine similarities between pairs of keywords, while ensuring that the chosen keywords
        are the least similar to each other.

        Note: Consider deprecation.

        Args:
        doc_embedding: numpy array, the embedding vector for a single document.
        candidate_embeddings: numpy array, the embedding vectors for all candidate keywords.
        candidates: list of strings, the candidate keywords to choose from.
        top_n: integer, the number of keywords to extract.
        nr_candidates: integer, the number of candidates to consider before extracting keywords.

        Returns:
        A list of the top_n most representative keywords for the given document, chosen from the given candidates list.
        """
        # Calculate distances and extract keyword_candidates
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        distances_candidates = cosine_similarity(candidate_embeddings, 
                                                candidate_embeddings)

        # Get top_n words as candidates based on cosine similarity
        words_idx = list(distances.argsort()[0][-nr_candidates:])
        words_vals = [candidates[index] for index in words_idx]
        distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

        # Calculate the combination of words that are the least similar to each other
        min_sim = np.inf
        candidate = None
        for combination in itertools.combinations(range(len(words_idx)), top_n):
            sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
            if sim < min_sim:
                candidate = combination
                min_sim = sim

        return [words_vals[idx] for idx in candidate]

    def _mmr(self, doc_embedding, word_embeddings, words, top_n, diversity):
        """
        Calculates the top_n keyword phrases from a list of words based on a maximum marginal relevance (MMR) algorithm,
        which optimizes for both the relevance and diversity of the extracted phrases.

        Args:
        doc_embedding: numpy array of BERT embeddings for the document to extract keywords from.
        word_embeddings: numpy array of BERT embeddings for each word in the list of words.
        words: list of words to extract phrases from.
        top_n: integer, the number of phrases to extract.
        diversity: float, parameter to control the balance between relevance and diversity in the extraction.

        Returns:
        A list of the top_n keyword phrases extracted from the list of words, ordered by relevance and diversity.

        Note:
        This method implements a maximum marginal relevance (MMR) algorithm, which is used to optimize for both the 
        relevance and diversity of the extracted phrases. It works by selecting the keyword phrase with the highest 
        relevance to the document, and then iteratively selecting the keyword phrase with the highest marginal 
        relevance (which balances relevance and diversity) until the top_n keyword phrases have been selected.
        """
        # Extract similarity within words, and between words and the document
        word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
        word_similarity = cosine_similarity(word_embeddings)

        # Initialize candidates and already choose best keyword/keyphras
        keywords_idx = [np.argmax(word_doc_similarity)]
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

        for _ in range(top_n - 1):
            # Extract similarities within candidates and
            # between candidates and selected keyword_candidates/phrases
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

            # Calculate MMR
            mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
            mmr_idx = candidates_idx[np.argmax(mmr)]

            # Update keyword_candidates & candidates
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        return [words[idx] for idx in keywords_idx]

    def _extract_single_document_using_yake(self, text, top_n=5):
            """
            Extracts the top n keywords from a text using YAKE.

            Args:
                text (str): The text to extract keywords from.
                top_n (int): The number of keywords to extract. Default is 5.

            Returns:
                A list of the top n keywords extracted from the text.
            """
            keywords = yake.KeywordExtractor(lan="en", n=3, windowsSize=3, top=5).extract_keywords(text)
            results = []
            for scored_keywords in keywords:
                for keyword in scored_keywords:
                    if isinstance(keyword, str):
                        results.append(keyword) 
            return results 

    def _repartition_list(self, list_or_array, partition_lengths):
        """
        Takes a flattened list of lists and, using the lengths of the original partitions,
        re-partitions the flattened list back into a list of lists.

        Args:
            list_or_array (list or array): A flattened list of elements.
            partition_lengths (list): A list of integers, indicating the number of elements
                                      in each partition of the original list.

        Returns:
            list: A list of lists, where each sub-list contains the elements of one of the
                  original partitions.
        
        Raises:
            Exception: If the sum of `partition_lengths` is not equal to the length of
                       `list_or_array`.
        """
        total_length = 0
        for num in partition_lengths:
            total_length += num
        
        if len(list_or_array) != total_length:
            raise Exception("Your partition_lengths must add up to the length of the input list")

        all_partitions = []
        partition = []
        partition_i = 0
        partition_end = partition_lengths[partition_i]
        for idx in range(len(list_or_array)):
            partition.append(list_or_array[idx])
            if idx == (partition_end - 1):
                all_partitions.append(partition)
                partition = []
                partition_i += 1
                if partition_i != len(partition_lengths):
                    partition_end += partition_lengths[partition_i]

        return all_partitions   

    def _rounddown(self, topic_obj, n=3):
        """Rounds down the match score to three decimal values."""
        newList = [(topicword[0], round(topicword[1], 3)) for topicword in topic_obj]

if __name__ == "__main__":
    start = time.time()

    df = pd.read_csv("datasets/graham_and_allison_btv.csv")
    df = df.query("response_comment_author == 'Graham'")
    df = copy.copy(df[["response_comment_id", "response_comment_body"]])
    df['index'] = df['response_comment_id']
    df = df.set_index("index")

    thought_grouper = KeywordExtractor(df.iloc[0:1000,:], create_matrix=False, ngram_range=(1,3))
    thought_grouper.extract_keywords(extractors=["slow_manual_bert", "keybert", "rake"])
    thought_grouper.to_csv()

    end = time.time()
    total_time = end - start
    print(f"Took us {total_time / 60} minutes")


    print("hello")