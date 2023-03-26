from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import autocorrect
from autocorrect import Speller
import shortuuid
import time
import copy
import os
from termcolor import cprint
import re
import datetime

"""

We want a SimilarityMatrix that takes the document dataset and finds the similarities between the different documents. 
We want to show the similarity scores for a given document. We also want to find the thought groupings within a document 
and tweak to find the ideal similarity score to properly distinguish similar thoughts, we then want to create a data frame 
with the top five most similar thoughts to a given thought and a dataframe of the top five most similar documents for 
a given document. We then want to extract the keywords and keyphrases from each document and return those keywords and 
keyphrases and ultimately build a dataframe with it.

Note: A document could be a comment on social media, a video transcript, a book, anything.

SimilarityMatrix
    - Inter-document similarity scores as a square pandas dataframe with document id's as both row and document labels.
"""

class SimilarityMatrix:
    """A class which creates the similarity matrix and a dataframe with the top n most similar documents per document from a corpus.

    Attribute:
        input_corpus (list, series, dataframe, etc):        The series of documents initially inputted
        document_series (list):                             The series of documents (e.g. comments, transcripts, articles, books, etc)
        document_indeces (list):                            The series of unique identifiers for each document
        similarity_matrix (pandas DataFrame):               The matrix of inter-document similarity scores calculated using BERT.
        document_embeddings (numpy array):                  BERT embeddings of sentence similarities using BERT
        most_similar_documents_column (list of strings):    The list of most similar documents with respect to each input document separated by two newline characters.
        top_n (integer):                                    The number of similar document to be returned per document in the return_similarity_dataframe method.

    Methods:
        create_similarity_matrix: 
            Creates a square similarity matrix using the cosine similarities between each of the documents.
        return_similarity_dataframe:
            Return the initial input with an additional column added which has the top n most similar/different documents for per document.

    alternate class name ideas:
    SimilarityGenerator
    
    """
    def __init__(self, df_or_list, autocorrect=False, document_column_name=None, create_matrix=True, top_n=5, create_index=False, split_document_on="."):
        self.input_corpus = df_or_list

        if self._is_input_a_dataframe(df_or_list):
            self._create_document_series_from_dataframe(df_or_list, document_column_name, create_index=create_index)
        else:
            self.document_series = list(df_or_list)
            self._create_unique_identifiers_for_documents()

        if type(self.document_series) != list and type(self.document_indeces) != list:
            raise Exception("Either the document series or the document indeces is not a list")

        if autocorrect: self._autocorrect_document_series()
        
        self.top_n = top_n
        self.similarity_matrix = None
        self.similarity_df = None
        self.most_similar_documents_column = None

        if create_matrix: self.create_similarity_matrix()

    def create_similarity_matrix(self, return_matrix=False):
        """Creates a square similarity matrix using the cosine similarities between each of the documents."""
        self._encode_bert_document_embeddings()

        print("\nCreating the similarity matrix from the sentence embeddings...", end=" ")
        document_embeddings = self.document_embeddings #TODO: Consider removing
        
        for i in range(len(self.document_series)):
            if i == 0:
                all_other_sentence_embeddings = document_embeddings[1:]
            elif i == len(document_embeddings) - 1:
                all_other_sentence_embeddings = document_embeddings[0:i]
            else:
                all_other_sentence_embeddings = np.append(document_embeddings[0:i], document_embeddings[i + 1:], axis = 0)
            embeddings_for_single_sentence = document_embeddings[i]

            cosine_similarity_result = cosine_similarity([embeddings_for_single_sentence], all_other_sentence_embeddings)
            cosine_similarity_row = np.insert(cosine_similarity_result, i, 0, axis = 1)
            if i == 0:
                cosine_similarity_matrix = cosine_similarity_row
            else:
                cosine_similarity_matrix = np.append(cosine_similarity_matrix, cosine_similarity_row, axis = 0)
                
        self.similarity_matrix = cosine_similarity_matrix

        cprint("Done!", color="green")
        if return_matrix:
            return self.similarity_matrix

    def return_similarity_dataframe(self, top_n=5, different_instead=False):
        """Return the initial input with an additional column added which has the top n most similar/different documents for per document.
        
        If the input corpus is an instance of the pandas DataFrame class
        """
        if not self.most_similar_documents_column:
            self._similarity_matrix_to_dataframe()
            self._top_n_similar_document_ids(top_n=top_n, different_instead=different_instead)

        if isinstance(self.input_corpus, pd.DataFrame):
            similarity_dataframe = copy.copy(self.input_corpus)
            similarity_dataframe[f"top_{self.top_n}_most_similar_documents"] = self.most_similar_documents_column
            return similarity_dataframe
        else:
            top_n = len(self.most_similar_documents_column[0])

            df_dict = {
                "document_id": self.document_indeces,
                "document_body": self.document_indeces,
                f"top_{top_n}_most_similar_documents": self.most_similar_documents_column
            }

            return pd.DataFrame(df_dict)

    def _is_input_a_dataframe(self, input):
        return isinstance(input, pd.DataFrame)

    def _create_document_series_from_dataframe(self, df_or_list, document_column_name, create_index):
        """Extracts the column of the input dataframe with the documents as a list and assigns it to document_series."""
        if document_column_name:
            self.document_series = list(df_or_list[document_column_name])
        else:
            self.document_series = list(df_or_list.iloc[:,1])
            if self.document_series[0] == self.document_series[1] == self.document_series[2]:
                raise Exception("You either did not provide unique documents or you provided the wrong column label")


        self.document_indeces = list(df_or_list.index) # Use unique comment identifiers from incoming DataFrame

        if not create_index:
            if len(str(self.document_indeces[0])) < 3:
                raise Exception("Please use document identifiers with more than 2 characters to ensure identifier uniqueness")

    def _create_unique_identifiers_for_documents(self):
        """Creates UUID identifiers for each document."""
        self.document_indeces = [str(shortuuid.uuid()) for i in range(len(self.document_series))] # Create UUID unique comment_id_indexentifiers from incoming DataFrame

    def _encode_bert_document_embeddings(self):
        """Converts each document to a list of embeddings using BERT."""
        
        num_documents = len(self.document_series)
        print("\nCreating the BERT embeddings for each document...")
        print(f"  You have {num_documents} documents to embed.")
        if num_documents > 10:
            eta = round((num_documents/100)*0.79, 2)
            print(f"  Estimated time: {eta} minutes")
        
        start = time.time()
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.document_embeddings = model.encode(self.document_series)
        total_minutes = round((time.time() - start) / 60, 2)
        cprint("Done!", color="green", end = " ")
        print(f"It took us a total of {total_minutes} minutes to parse {len(self.document_series)} documents")

    def _autocorrect_document_series(self):
        """Applies autocorrect to each document. 
        
        Results may vary."""
        autocorrect = Speller(fast = True)
        self.document_series = [autocorrect(document) for document in self.document_series]

    def _similarity_matrix_to_dataframe(self):
        """Converts the similarity matrix into a pandas dataframe."""
        if self.similarity_matrix is not None:
            similarity_df = pd.DataFrame(self.similarity_matrix, columns = self.document_indeces)
            similarity_df["index"] = self.document_indeces
            similarity_df = similarity_df.set_index('index')
            self.similarity_df = similarity_df
        else:
            raise Exception("You must first create the similarity_matrix using create_similarity_matrix")

    def _top_n_similar_document_ids(self, top_n=5, different_instead=False, show_similarity_score=True):
        """Returns the list of lists of the top n most similar documents for each document."""
        self.top_n = top_n
        
        if self.similarity_df is not None:

            if show_similarity_score:
                top_most_similar_documents_dict = {}
                for comment_id in self.similarity_df.index:
                    try:
                        top_recommends = list(self.similarity_df.loc[comment_id, :].sort_values(ascending = different_instead)[0:top_n].index)
                        top_similarity_scores = list(self.similarity_df.loc[comment_id, :].sort_values(ascending = different_instead)[0:top_n])
                        top_most_similar_documents_dict[comment_id] = list(zip(top_recommends, top_similarity_scores))
                    except:
                        top_recommends = list(self.similarity_df.loc[comment_id, :].iloc[0,:].sort_values(ascending = different_instead)[0:top_n].index)
                        top_similarity_scores = list(self.similarity_df.loc[comment_id, :].sort_values(ascending = different_instead)[0:top_n])
                        top_most_similar_documents_dict[comment_id] = list(zip(top_recommends, top_similarity_scores))

                top_most_similar_ids_column = [top_most_similar_documents_dict[document_id] for document_id in self.document_indeces]
                self.most_similar_documents_column = ["\n\n".join(["(" + str(round(similar_id[1]*100, 2)) + "%" + " similarity score)\n" + self.document_series[self.document_indeces.index(similar_id[0])] for similar_id in top_recommends]) for top_recommends in top_most_similar_ids_column]

            else:
                top_most_similar_documents_dict = {}
                for comment_id in self.similarity_df.index:
                    try:
                        top_recommends = list(self.similarity_df.loc[comment_id, :].sort_values(ascending = different_instead)[0:top_n].index)
                        top_most_similar_documents_dict[comment_id] = top_recommends
                    except:
                        top_recommends = list(self.similarity_df.loc[comment_id, :].iloc[0,:].sort_values(ascending = different_instead)[0:top_n].index)
                        top_most_similar_documents_dict[comment_id] = top_recommends
    
                top_most_similar_ids_column = [top_most_similar_documents_dict[document_id] for document_id in self.document_indeces]
                self.most_similar_documents_column = ["\n\n".join([self.document_series[self.document_indeces.index(similar_id)] for similar_id in top_recommends]) for top_recommends in top_most_similar_ids_column]
                
                # Try something like this to show the similarity score
                #self.most_similar_documents_column = ["\n\n".join([self.document_series[self.document_indeces.index(similar_id)] + str(round(100 * self.similarity_df[similar_id, self.document_indeces[i]], 2)) + "%" for similar_id in top_most_similar_ids_column[i]]) for i in range(len(top_most_similar_ids_column))]


        else:
            raise Exception("You must first create the similarity_df using the similarity_matrix_to_dataframe method.")

if __name__ == "__main__":
    pass
