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
    return freq / n_terms if n_terms else 0.0



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

    for idx, d in enumerate(D):
        for term in V:
            idfs[term] = idf(D, term) # Store idf of term in corpus
            tfs[idx][term] = tf(d, term) # Store tf of term in document
            
    array = np.zeros((len(D), len(V))) # Initialize array with zeros: shape (|D|, |V|)
    for i, term in enumerate(V):
        for j, _ in enumerate(array):
            array[j][i] = idfs[term] * tfs[j][term] # Fill in array with tf * idf scores
    return array