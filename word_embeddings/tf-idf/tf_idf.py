import math


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
    IDF = log ( num of documents in corpus / num of documents having the term)
    """
    n_docs: int = len(D)
    n_docs_with_term: int = sum(1 for d in D if t in d)
    if n_docs_with_term == 0:
        return 0
    return math.log10(n_docs / n_docs_with_term)