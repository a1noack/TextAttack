import numpy as np
import os

from textattack import utils as utils
from .word_swap import WordSwap

class WordSwapCounterfit(WordSwap):
    """
    Transforms an input by replacing its words with synonyms in the counter-fitted
    embedding space. 

    Args:
        replace_stopwords (:obj:`bool`, optional): Whether to replace the stopwords in the text. Defaults to False. 
        max_candidates (:obj:`int`, optional): The default number of words to replace. Defaults to None. 
        word_embedding_folder (:obj:`str`, optional): The path to the word_embedding folder. Defaults to paragram_300_sl999

    Raises:
        ValueError: If the word_embedding_folder is not found. 
   
    """
    
    PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/AttackGeneration/word_embeddings/'
    
    def __init__(self, replace_stopwords=False, max_candidates=None, word_embedding_folder='paragram_300_sl999'):
        super().__init__(replace_stopwords)
        self.max_candidates = max_candidates
        if word_embedding_folder == 'paragram_300_sl999':
            word_embeddings_file = 'paragram_300_sl999.npy'
            word_list_file = 'wordlist.pickle'
            word_embedding_matrix_file = 'paragramnn.py'
        else:
            raise ValueError(f'Could not find word embedding {word_embedding}')# Concatenate folder names to create full path.
    
        utils.download_if_needed(WordSwapCounterfit.PATH)
        word_embeddings_file = os.path.join(WordSwapCounterfit.PATH, word_embedding_folder, word_embeddings_file)
        word_list_file = os.path.join(WordSwapCounterfit.PATH, word_embedding_folder, word_list_file)
        word_embedding_matrix_file = os.path.join(WordSwapCounterfit.PATH, word_embedding_folder, word_embedding_matrix_file)
        
        # Actually load the files from disk.
        word_embeddings = np.load(word_embeddings_file)
        word_embedding_word2index = np.load(word_list_file, allow_pickle=True)
        word_embedding_matrix = np.load(word_embedding_matrix_file)
        # Build glove dict and index.
        word_embedding_index2word = {}
        for word, index in word_embedding_word2index.items():
            word_embedding_index2word[index] = word
        
        self.word_embeddings = word_embeddings
        self.nn = word_embedding_matrix
        self.word_embedding_index2word = word_embedding_index2word
        self.word_embedding_word2index = word_embedding_word2index
    
    def _get_replacement_words(self, word, max_candidates=10):
        """ 
        Returns a list of possible 'candidate words' to replace a word in a sentence 
        or phrase. 
        
        Based on nearest neighbors selected word embeddings.
        """
        if self.max_candidates:
            max_candidates = min(max_candidates, self.max_candidates)
        try:
            word_id = self.word_embedding_word2index[word]
            nnids = self.nn[word_id][1:max_candidates+1]
            candidate_words = []
            for i, wi in enumerate(nnids):
                candidate_words.append(self.word_embedding_index2word[wi])
            return candidate_words
        except KeyError:
            # This word is not in our word embedding database, so return an empty list.
            return []