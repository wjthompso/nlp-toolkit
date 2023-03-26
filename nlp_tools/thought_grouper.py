from sentence_transformers import SentenceTransformer
from similarity_matrix import SimilarityMatrix
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time
import copy
from termcolor import cprint
import re

class ThoughtGrouper(SimilarityMatrix):
    """A class which groups similar sentences within each document in a corpus into thoughts and then flattens the results into a pandas DataFrame."""

    def __init__(self, 
                 df_or_list, 
                 autocorrect=False, 
                 document_column_name=None, 
                 create_matrix=True, 
                 top_n=5):
        """
        Initialize the ThoughtGrouper class by calling the constructor of the parent class (SimilarityMatrix) and initializing instance variables.
        
        Parameters:
            df_or_list:                 A pandas DataFrame or a list of documents.
            autocorrect (bool):         Whether to apply autocorrect on input text. Defaults to False.
            document_column_name (str): The column name for documents in the input DataFrame. Required if df_or_list is a DataFrame.
            create_matrix (bool):       Whether to create the similarity matrix immediately upon initialization. Defaults to True.
            top_n (int):                The number of top similar sentences to consider when calculating similarity. Defaults to 5.
        """
        super().__init__(df_or_list, autocorrect, document_column_name, create_matrix, top_n)

        self.documents_split_by_sentence = None
        self.all_thought_groupings = None
        self.corpus_of_sentence_embeddings = None
        self.all_thoughts_df = None
        self.thought_similarity_dataframe = None
        self.filtered_thought_similarity_dataframe = None
        self._split_documents_into_sentences()

    def _split_documents_into_sentences(self):
        """Converts the list of documents into a list of lists of sentences.
        """

        documents_split_by_sentence = []

        for document in self.document_series:
            sentence_split = [sentence.strip() for sentence in re.split(r'[.?!]+', document)]
            for idx, sentence in enumerate(sentence_split):
                if sentence == "":
                    del sentence_split[idx]
            documents_split_by_sentence.append([sentence.strip() for sentence in sentence_split])
        
        self.documents_split_by_sentence = documents_split_by_sentence

    def _encode_bert_sentence_embeddings(self):
        """
        Encodes sentences within a single document into BERT embeddings using SentenceTransformer.
        Note: Consider using parallel processing for large-scale processing.
        """

        model = SentenceTransformer('bert-base-nli-mean-tokens')

        self.corpus_of_sentence_embeddings = [[model.encode(sentence) for sentence in document] for document in self.documents_split_by_sentence] #TODO: Consider refactoring

    def _create_within_document_similarity_matrix(self, list_of_sentence_embeddings):
        """
        Creates a square similarity matrix using the cosine similarities between each of the sentences within a single document.
        
        Parameters:
            list_of_sentence_embeddings (list of vectors): Sentence embeddings for a document.
        
        Returns:
            cosine_similarity_matrix (np.ndarray): A square similarity matrix containing cosine similarities between sentences.
        """

        if len(list_of_sentence_embeddings) > 1:
            for i in range(len(list_of_sentence_embeddings)):
                if i == 0:
                    all_other_sentence_embeddings = list_of_sentence_embeddings[1:]
                elif i == len(list_of_sentence_embeddings) - 1:
                    all_other_sentence_embeddings = list_of_sentence_embeddings[0:i]
                else:
                    all_other_sentence_embeddings = np.append(list_of_sentence_embeddings[0:i], list_of_sentence_embeddings[i + 1:], axis = 0)
                embeddings_for_single_sentence = list_of_sentence_embeddings[i]

                cosine_similarity_result = cosine_similarity([embeddings_for_single_sentence], all_other_sentence_embeddings)
                cosine_similarity_row = np.insert(cosine_similarity_result, i, 0, axis = 1)
                if i == 0:
                    cosine_similarity_matrix = cosine_similarity_row
                else:
                    cosine_similarity_matrix = np.append(cosine_similarity_matrix, cosine_similarity_row, axis = 0)
                    
            return cosine_similarity_matrix

        # In this case, there was only one sentence embedding, nothing to compare it to.
        else:
            cosine_similarity_matrix = None
            return 

    def _create_thought_groupings_for_one_document(self, document, document_similarity_matrix, split_groups_on=0.4, algo="partial-axial"):
        """Groups sentences together into "thoughts", given that there are more than 2 sentences, if their similarity is greater than split_groups_on.

        The "sequential" algorithm compares the similarity of the first and second sentence of the document. If their similarity score is above
        the split_groups_on value, the two sentences get grouped into the same "thought" under the key integer 1. It then compares the similarity 
        between the 2nd and 3rd sentence. If their similarity is high enough, the third sentence gets added to the same thought grouping key 1.
        If not, the 3rd sentence gets added to a new thought under the group key integer 2. The process gets repeated.

        The "center" algorithm finds the most similar two sentences, creates a thought grouping, and then works outwards from there.
        
        The "axial" algorithm (named so because it looks forward and backward) works sequentially but if the next sentence isn't similar, it compares 
        the next sentence to all previous sentences in the group.

        The "partial-axial" algorithm (named so because it looks forward and backward) works sequentially but if the next sentence isn't similar, it 
        compares the next sentence to the 2 previous sentences in the group.

        Parameters:
            document (list of strings):                     The document with multiple sentences separated by periods.
            document_similarity_matrix (list of vectors):   Matrix of inter-document sentence similarities.
            split_groups_on (float):                        The similarity score calculated from cosine similarity to split the groups on. A sort of boundary.
            algo (str):                                     Algorithm used for grouping sentences. One of ['sequential', 'center', 'axial', 'partial-axial'].

        Returns:
            thought_groupings (list): List of grouped sentences representing thoughts.
        """

        if len(document) < 2:
            return document

        if algo not in ["sequential", "center", "axial", "partial-axial"]:
            raise Exception("You need to set algo to one of either 'sequential', 'center', 'axial', or 'partial-axial'")

        # If the comment only has one sentence, there will be no similarity_matrix
        if not isinstance(document_similarity_matrix, np.ndarray):
            return document

        if algo == "sequential":
            group_i = 1
            thought_dict = {group_i: []}
            matrix_length = len(document_similarity_matrix)
            for i in range(0, matrix_length - 1):
                if i == 0:
                     thought_dict[group_i].append(document[i])
                if document_similarity_matrix[i, i + 1] > split_groups_on:
                    thought_dict[group_i].append(document[i + 1])
                else:
                    group_i += 1
                    thought_dict[group_i] = [document[i + 1]]
                
            #This is a list of sentence groupings. The sentence groupings will be sentences joined by a period.
            thought_groupings = [". ".join(sentences) for sentences in thought_dict.values()]
            return thought_groupings

        if algo == "center":
            indices_of_max = list(np.where(document_similarity_matrix == np.max(document_similarity_matrix))[0])

            group_i = 1
            thought_dict = {group_i: []}
            matrix_length = len(document_similarity_matrix)
            for i in range(indices_of_max[0], matrix_length - 1):
                # Add the starting sentence to group 1
                if i == indices_of_max[0]:
                    thought_dict[group_i].append(document[i])
                    thought_dict[group_i].append(document[i + 1])
                # Look at the next sentence, add it to the existing group if it's similar enough. Make new group if it's not similar
                else:
                    if document_similarity_matrix[i, i + 1] > split_groups_on:
                        thought_dict[group_i].append(document[i + 1])
                    else:
                        group_i += 1
                        thought_dict[group_i] = [document[i + 1]]

            group_i = 1
            for i in range(indices_of_max[0], 0, -1):
                if document_similarity_matrix[i, i - 1] > split_groups_on:
                    thought_dict[group_i].append(document[i - 1])
                else:
                    group_i -= 1
                    thought_dict[group_i].append(document[i - 1])

            thought_groupings = [". ".join(sentences) for sentences in thought_dict.values()]
            return thought_groupings 

        if algo == "axial":
            # I think we should boost the score a bit more dramatically for the axial algo.
            split_groups_on += 1

            group_i = 1
            thought_dict = {group_i: []}
            matrix_length = len(document_similarity_matrix)
            for i in range(0, matrix_length - 1):
                if i == 0:
                    thought_dict[group_i].append(document[i])
                if document_similarity_matrix[i, i + 1] > split_groups_on:
                    thought_dict[group_i].append(document[i + 1])
                else:
                    for j in range(i - len(thought_dict[group_i]), i):
                        if document_similarity_matrix[j, i + 1] > split_groups_on:
                            thought_dict[group_i].append(document[i + 1])
                            break
                        if j == i - 1:
                            group_i += 1
                            thought_dict[group_i] = [document[i + 1]]
                
            #This is a list of sentence groupings. The sentence groupings will be sentences joined by a period.
            thought_groupings = [". ".join(sentences) for sentences in thought_dict.values()]
            return thought_groupings

        if algo == "partial-axial":
            # I think we should boost the score for partial-axial.

            group_i = 1
            thought_dict = {group_i: []}
            matrix_length = len(document_similarity_matrix)
            for i in range(0, matrix_length - 1):
                if i == 0:
                    thought_dict[group_i].append(document[i])
                if document_similarity_matrix[i, i + 1] > split_groups_on:
                    thought_dict[group_i].append(document[i + 1])
                else:
                    for j in range(i - 2, i):
                        if document_similarity_matrix[j, i + 1] > split_groups_on:
                            thought_dict[group_i].append(document[i + 1])
                            break
                        if j == i - 1:
                            group_i += 1
                            thought_dict[group_i] = [document[i + 1]]
                
            #This is a list of sentence groupings. The sentence groupings will be sentences joined by a period.
            thought_groupings = [". ".join(sentences) for sentences in thought_dict.values()]
            return thought_groupings

    def _create_all_thought_groupings(self, asynchronous=False, algo="sequential"):
        """
        Creates flattened series of thought groupings for all documents returned as a a dataframe.
        The resulting dataframe contains document id, thought id, thought text, and optionally the document the thought came from.
        
        Parameters:
            asynchronous (bool): Whether to use asynchronous processing for creating thought groupings. Defaults to False.
            algo (str):          Algorithm used for grouping sentences. One of ['sequential', 'center', 'axial', 'partial-axial'].
        """

        print("Encoding the BERT embeddings for all sentences in all documents.")

        self._encode_bert_sentence_embeddings()

        cprint("Done!", color="green", end=" ")
        print("We created the sentence embeddings for all documents.")

        all_thought_groupings = []

        print(f"Creating thought groupings for {len(self.document_series)} documents")

        for i in range(len(self.documents_split_by_sentence)):
            cosine_similarity_matrix = self._create_within_document_similarity_matrix(self.corpus_of_sentence_embeddings[i])
            thought_grouping = self._create_thought_groupings_for_one_document(self.documents_split_by_sentence[i], cosine_similarity_matrix, algo=algo)
            
            # Note: Each thought_grouping, which is a list of thought groupings
            all_thought_groupings.append(thought_grouping)

        self.all_thought_groupings = all_thought_groupings

        cprint("Done!", color="green", end=" ")
        print(f"We created the thought groupings for {len(self.documents_split_by_sentence)} documents!")

    def _create_thought_grouping_df(self, to_csv=False):
        """
        Creates a pandas DataFrame containing thought groupings and the original comments they were derived from.
        
        Parameters:
            to_csv (bool): Whether to save the resulting DataFrame as a CSV file. Defaults to False.
        """

        if not self.all_thought_groupings:
            self._create_all_thought_groupings(algo="partial-axial")

        all_thoughts_dict = {"thought": [],
                            "original_comment": [],
                            "original_comment_id_index": [],
                            }

        for doc_i in range(len(self.all_thought_groupings)):
            for thought_j in range(len(self.all_thought_groupings[doc_i])):
                all_thoughts_dict['thought'].append(self.all_thought_groupings[doc_i][thought_j])
                all_thoughts_dict['original_comment'].append(self.document_series[doc_i])
                all_thoughts_dict['original_comment_id_index'].append(self.document_indeces[doc_i])

        self.all_thoughts_df = pd.DataFrame(all_thoughts_dict)

    def create_thought_similarity_dataframe(self, keep_body=False, to_csv=False, fix_spelling=False, top_n=5):
        """
        Creates a thought recommendation DataFrame with similarity scores between thoughts.

        Parameters:
            keep_body (bool): Whether to include the original comment body in the resulting DataFrame. Defaults to False.
            to_csv (bool): Whether to save the resulting DataFrame as a CSV file. Defaults to False.
            fix_spelling (bool): Whether to autocorrect spelling errors in the text. Defaults to False.
            top_n (int): Number of most similar thoughts to return. Defaults to 5.
        """

        if not isinstance(self.all_thoughts_df, pd.DataFrame) and not self.all_thoughts_df:
            self._create_thought_grouping_df()

        self.thought_similarity_dataframe = SimilarityMatrix(
            self.all_thoughts_df[["thought", "original_comment_id_index"]],
            autocorrect=fix_spelling,
            document_column_name="thought",
            create_index=True,
            top_n=top_n,
            ).return_similarity_dataframe()

        if keep_body:
            self.thought_similarity_dataframe["original_comment"] = self.all_thoughts_df["original_comment"]

        if to_csv:
            self.thought_similarity_dataframe.to_csv("./output/thought_similarity_dataframe.csv")

    def filter_for_long_thoughts(self, min_sentences=2, to_csv=False):
        """Filters the thought_similarity_dataframe to only include thoughts with a minimum number of sentences.
        
        Loops through the indeces of the thought_similarity_dataframe and only saves indeces which correspond to thoughts with the specified
        minimum number of sentences. The filtered results are saved to a new dataframe titled filtered_thought_similarity_dataframe.
        
        Parameters:
            min_sentences (int): The minimum number of sentences a thought should have to be included in the filtered DataFrame. Defaults to 2.
            to_csv (bool): Whether to save the resulting DataFrame as a CSV file. Defaults to False.
        """
        
        if not isinstance(self.thought_similarity_dataframe, pd.DataFrame):
            raise Exception("Please create the thought_similarity_dataframe using create_thought_similarity_dataframe first.")
        
        long_thoughts = []
        for index in self.thought_similarity_dataframe.index:
            if len(re.split(r'[.?!]+', self.thought_similarity_dataframe["thought"][index])) >= min_sentences:
                long_thoughts.append(index)
        
        self.filtered_thought_similarity_dataframe = self.thought_similarity_dataframe.loc[long_thoughts, :]

        if to_csv:
            self.filtered_thought_similarity_dataframe.to_csv("./output/filtered_thought_similarity_dataframe.csv")

    def save_to_csv(self, filtered=False, min_sentences=2):
        """Saves either the thought_similarity_dataframe or the filtered_thought_similarity_dataframe to a csv.
        
        Parameters:
            filtered (bool): Whether to save the filtered DataFrame or the unfiltered one. Defaults to False (saves unfiltered).
            min_sentences (int): The minimum number of sentences a thought should have to be included in the filtered DataFrame. Only used if 'filtered' is set to True. Defaults to 2.
        """
        if filtered and not isinstance(self.filtered_thought_similarity_dataframe, pd.DataFrame):
            self.filter_for_long_thoughts(min_sentences=min_sentences)
            self.filtered_thought_similarity_dataframe.to_csv("./output/filtered_thought_similarity_dataframe.csv")

        elif filtered and isinstance(self.filtered_thought_similarity_dataframe, pd.DataFrame):
            self.filtered_thought_similarity_dataframe.to_csv("./output/filtered_thought_similarity_dataframe.csv")
        
        else:
            self.thought_similarity_dataframe.to_csv("./output/thought_similarity_dataframe.csv")

if __name__ == "__main__":

    start = time.time()
    # myList = ["Hello there", "why hello to you", "and let me give you some information"]
    # similarity_matrix = SimilarityMatrix(myList)
    # print(similarity_matrix.document_indeces)
    # end = time.time()
    # total_time = end - start
    # print(f"Took us {total_time}")

    df = pd.read_csv("datasets/graham_and_allison_btv.csv")
    df = df.query("response_comment_author == 'Graham'")
    df = copy.copy(df[["response_comment_id", "response_comment_body"]])
    df['index'] = df['response_comment_id']
    df = df.set_index("index")
    # similarity_matrix = SimilarityMatrix(df.iloc[0:5,:])
    # # similarity_matrix = SimilarityMatrix(df)
    # top_5_recommends = similarity_matrix.return_similarity_dataframe()

    thought_grouper = ThoughtGrouper(df.iloc[0:100,:], create_matrix=False)
    thought_grouper.create_thought_similarity_dataframe()
    thought_grouper.filter_for_long_thoughts(min_sentences=4, to_csv=True)
    # thought_grouper._create_all_thought_groupings(algo="partial-axial")
    # for row in thought_grouper.all_thought_groupings[0]:
    #     print(row)
    #     print("\n")
    # print("Hello")

    end = time.time()
    total_time = end - start
    print(f"Took us {total_time / 60} minutes")


    print("hello")