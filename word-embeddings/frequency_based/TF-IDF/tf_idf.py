import math
import numpy as np
from collections import defaultdict



def tf(d: str, t: str)-> float:
    """ Computes the frequency of a term 't' given
     a document 'd'.
    TF = term count / number of terms in document.
    """
    n_terms: int = len(d.split()) # Amount of terms in 'd'.
    freq: int = d.count(t) # Frequency of term 't' in 'd'.
    return freq / n_terms if freq else 0.0



def idf(D: list[str], t: str)-> float:
    """ Computes the proportion of documents 'd' found in
    a corpus 'D' containing the term 't'.
    IDF = log ( num of documents in corpus / num of documents having the term).
    Uses smoothing by adding 1 to numerator, denominator, and log.
    """
    n_docs: int = len(D) # Number of documents in corpus
    n_docs_with_term: int = sum(1 for d in D if t in d)
    if n_docs_with_term == 0:
        return 0
    return math.log(n_docs + 1 / n_docs_with_term + 1) + 1 


def tf_idf(D: list[str], V: set[str])-> np.array:
    """ Calculates the tf-idf of every
     term in a corpus D.
    """
    idfs: dict = {}
    tfs = defaultdict(dict)

    for term in V:
        idfs[term] = idf(D, term)
        
    for idx, d in enumerate(D):
        for term in V:
            tfs[idx][term] = tf(d, term) # Store tf of term in document
            
    array = np.zeros((len(D), len(V))) # Initialize array with zeros: shape (|D|, |V|)
    for i, term in enumerate(V):
        for j, _ in enumerate(array):
            array[j][i] = idfs[term] * tfs[j][term] # Fill in array with tf * idf scores
    return array


class Tfidf:
    """ TF-IDF model
    """


    def __init__(self):
        self.V = None
        self.D = None
        self.idfs = {}


    def fit_transform(self, D: list[str]):
        """ Fits and transforms a corpus of texts into
        a vectorized array.
        """
        self.D = D
        tfs = defaultdict(dict)
        self.V = {word.lower() for d in self.D for word in d.split()}
        for idx, d in enumerate(self.D):
            for term in self.V:
                self.idfs[term] = self._idf(term)
                tfs[idx][term] = self._tf(d, term)
        array = np.zeros((len(self.D), len(self.V)))
        for i, term in enumerate(self.V):
            for j, _ in enumerate(array):
                array[j][i] = self.idfs[term] * tfs[j][term]
        return array
    

    def _tf(self, d: str, t: str)-> float:
        """ Computes the frequency of a term 't' given
        a document 'd'.
        TF = term count / number of terms in document.
        """
        n_terms: int = len(d.split()) # Amount of terms in 'd'.
        freq: int = d.count(t) # Frequency of term 't' in 'd'.
        return freq / n_terms if n_terms else 0.0

    
    def _idf(self, t: str)-> float:
        """ Computes the proportion of documents 'd' found in
        a corpus 'D' containing the term 't'.
        IDF = log ( num of documents in corpus / num of documents having the term).
        Uses smoothing by adding 1 to numerator, denominator, and log.
        """
        n_docs: int = len(self.D) # Number of documents in corpus
        n_docs_with_term: int = sum(1 for d in self.D if t in d)
        if n_docs_with_term == 0:
            return 0
        return math.log(n_docs + 1 / n_docs_with_term + 1) + 1 