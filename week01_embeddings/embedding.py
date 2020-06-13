# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: embedding.py
@Time: 2020/6/9 下午10:55
@Desc: embedding.py
"""
try:
    import numpy as np
    from utils.config import DATASETS_DIR

    import gensim.downloader as api
    import matplotlib.pyplot as plt

    from nltk.tokenize import WordPunctTokenizer
    from gensim.models import Word2Vec
    from sklearn.manifold import TSNE

except ImportError as e:
    print(e)
    raise ImportError

'''
get data
'''
data = list(open('{}/quora.txt'.format(DATASETS_DIR), encoding='utf-8'))

'''
tokenization for word
'''
tokenizer = WordPunctTokenizer()
data_tok = [tokenizer.tokenize(s.lower()) for s in data]

'''
word embeddings
'''
model = Word2Vec(data_tok, size=30, min_count=5, window=5).wv
a = model.get_vector('is')
s = model.most_similar('is')

# using pre-trained model
# model = api.load('glove-twitter-100')
# # s = model.most_similar(positive=['coder', 'money'], negative=['brain'])

'''
visualizing word vectors
'''
# words = sorted(model.vocab.keys(),
#                key=lambda word: model.vocab[word].count,
#                reverse=True)[:1000]
# f_embeddings = np.array([model.get_vector(w) for w in words])
# t_model = TSNE(n_components=2).fit_transform(f_embeddings)
#
# text_embed_data = t_model[:200]
# plt.scatter(text_embed_data[:, 0], text_embed_data[:, 1])
# for i in range(len(text_embed_data)):
#     plt.annotate(words[i], xy=(text_embed_data[i, 0], text_embed_data[i, 1]),
#                  xytext=(text_embed_data[i, 0]+0.1, text_embed_data[i, 1]+0.1))
# plt.show()
#
# plt.scatter(t_model[:, 0], t_model[:, 1])
# plt.show()

'''
get phrase embeddings
'''
def get_phrase_embeddings(phrase, model):

    # tokenization
    datas = WordPunctTokenizer().tokenize(phrase.lower())
    # delete word if vocab is not exit.
    vector = np.array([model.get_vector(s) for s in datas if s in model.vocab.keys()])

    if len(vector) > 0:
        vector = np.mean(vector, axis=0)
        return vector
    else:
        return np.ones(model.vector_size) * 1e-10
# vector = get_phrase_embeddings("I'm very sure. This never happened to me before...", model)

# chosen_phrases = data[::len(data) // 1000]
# vectors = np.array([get_phrase_embeddings(phrase, model) for phrase in chosen_phrases])
# t_model = TSNE(n_components=2).fit_transform(vectors)
# plt.scatter(t_model[:, 0], t_model[:, 1])
# plt.show()

'''
build a simple system for similar question
'''
vectors = np.array([get_phrase_embeddings(s, model) for s in data]) # for global var
def get_similar_question(phrase, k=10):
    """
    this system is using the most similar top-k vector to answer.
    fist_step: get all line vector using get_phrase_embeddings.
    second_step: using model to get top-k vector.
    :return: k vector
    """
    vector = get_phrase_embeddings(phrase, model)
    # calculate cosine similarity
    similar_matrix = vectors.dot(vector[:, np.newaxis]) / (np.linalg.norm(vectors, axis=1, keepdims=True) *
                                                           np.linalg.norm(vector))
    top = (np.argsort(similar_matrix.reshape(-1))[-k:])[::-1]

    return np.array(data)[top]

result = get_similar_question('How do i enter the matrix?', k=10)
print(result)
