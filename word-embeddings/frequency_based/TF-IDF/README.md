# TF-IDF
Text needs to be represented as numbers in order to use them in ML models.
How can we represent a set of emails?
tf-idf is a powerful yet simple way to give more weight to words that matter more in order to grasp the context of the text.

## Terminology
**Linguistic terms** (Python terms)
#### Corpus
A **corpus** is a collection (list/dict/tuple/set) of **documents** (string). If we had a list of 2 sentences: `mylist = ["The cat chased the mouse", "The dog played with the ball"]`,
then `mylist` is a corpus.
#### Document
A **document** is a single text (string), see the example sentence above. A **document** is a member (item) of a **corpus** (list/dict/tuple/set).
`"The cat chased the mouse"` is one of two documents found within the corpus `mylist`.
#### Term
A **term** is a word (sub string) in a document (string). Ex: `"The cat chased the mouse"`, "*the*", "*cat*", etc are all **terms** found within the document (string).

## TF (Term Frequency)
The **term frequency** measures how often a term appears in a document. This operation runs on a document level, that is to say, it takes in a document *d* and counts the occurrences of term *t* within that single document.
## IDF (Inverse Document Frequency)
The **idf** works on a corpus level, that is to say, a corpus *D* is passed in, along with a term *t*. The **term** is search for in every document *d* within the corpus *D*, counting how many documents contain the term. For example, the term *cat* appears in only one document (the first sentence) in `mylist`, but *the* appears in all documents (both sentences) in `mylist` (corpus). A term that is found in nearly all documents within a corpus is probably a very common word like: the, a, an, etc. While a word hardly found in a handful of documents like "stethoscope" has more importance, therefore, IDF favors, gives more weight to infrequent terms. These infrequent terms play a larger role in the meaning of a sentence (document). 