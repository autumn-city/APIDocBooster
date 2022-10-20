from re import L
from fuzzysearch import find_near_matches
from rank_bm25 import BM25Okapi,BM25L,BM25Plus

def test_fuzzy_search():
    term_1 = 'hidden_size – The number of features in the hidden state h'
    term_2 = 'We initialize the hidden state with None but as you will see in the forward pass this will basically mean that we will populate it with zeros'
    print(find_near_matches(term_1, term_2, max_l_dist=3,max_insertions=3, max_substitutions=3))

def test_bm25():
    corpus = ['We instantiate two linear modules  and we set the bias equal to False.',
    'Internally, the projections will be done using the torch.Linear module and we\'ll have a separate  one for both the key and the query tensor.',
    'And we also prepared the scalar.',
    'Here we instantiate the LSTMcell.',
    'We initialize the hidden state with None  but as you will see in the forward pass this will basically mean that we will populate it with zeros.']
    tokenized_corpus = [doc.lower().replace('.','').replace(',','').replace('!','').replace('?','').split(" ") for doc in corpus]
    print(tokenized_corpus)
    bm25 = BM25Okapi(tokenized_corpus)
    query = 'lstmcell input_size hidden_size bias'
    tokenized_query = query.lower().replace('_',' ').split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    print(doc_scores)
    print(bm25.get_top_n(tokenized_query, corpus, n=3))

def bm25_search(corpus,query):

    # corpus = list of sentences

    # query = string containing the API name and its parameters

    tokenized_corpus = [doc.lower().replace('.','').replace(',','').replace('!','').replace('?','').split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().replace('_',' ').split(" ")

    doc_scores = bm25.get_scores(tokenized_query)

    return bm25.get_top_n(tokenized_query, corpus, n=3)


if __name__ == '__main__':
    test_bm25()
