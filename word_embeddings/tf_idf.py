import math


def tf(d: str, t: str)-> float:
    n = len(d.split()) # Amount of terms in 'd'.
    freq: int = d.count(t) # Frequency of term 't' in 'd'.
    return freq / n if n else 0.0



# def idf(D: list[str], t: str)-> float:
#     n = len(D)
#     n_docs_with_term: int = sum(1 for d in D if any(t in d))
#     if n_docs_with_term == 0:
#         return 0
#     return math.log((n + 1) / (n_docs_with_term + 1)) + 1



# s1 = "the cat chased the mouse"
# s2 = "the dog played with the ball"

# my_corpus = [s1, s2]
# t = 'played'

# print(tf(s1, t))
# print(idf(my_corpus, t))