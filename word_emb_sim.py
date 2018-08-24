#  -*- coding: UTF-8 -*-
from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import range
import numpy as np
from gensim.models import Word2Vec


def extrema(A):
    if not A.shape[0]:
        return 0.  # empty array has no max
    amax, amin = A.max(0), A.min(0)
    amax[amax < np.abs(amin)] = amin[amax < np.abs(amin)]
    return amax


def _wmdist(x, y, word_embeddings):
    """
    word movers distance between x and y (both dicts with {word: count})
    """
    d = word_embeddings.wmdistance(x.keys(), y.keys())
    if d < np.inf:
        return d
    else:
        return -1


def _avg_cos(x, y, word_embeddings):
    """
    cosine similarity between the average word embeddings of x and y (both dicts with {word: count})
    the average is taken based on the counts in the dict, i.e., can be tfidf weighting or straight average
    """
    vec_x = np.sum([x[word] * word_embeddings[word] for word in x if word in word_embeddings], axis=0)
    vec_y = np.sum([y[word] * word_embeddings[word] for word in y if word in word_embeddings], axis=0)
    if isinstance(vec_x, float) or isinstance(vec_y, float):
        return 0.  # no dot product with no vectors
    return np.dot(vec_x, vec_y) / (np.linalg.norm(vec_x) * np.linalg.norm(vec_y))


def _simple_avg_cos(x, y, word_embeddings):
    """
    cosine similarity between the average word embeddings of x and y (both dicts with {word: count} or just lists of words)
    the average is only taken based on the occurrence of the words
    """
    vec_x = np.sum([word_embeddings[word] for word in x if word in word_embeddings], axis=0)
    vec_y = np.sum([word_embeddings[word] for word in y if word in word_embeddings], axis=0)
    if isinstance(vec_x, float) or isinstance(vec_y, float):
        return 0.  # no dot product with no vectors
    return np.dot(vec_x, vec_y) / (np.linalg.norm(vec_x) * np.linalg.norm(vec_y))


def _max_cos(x, y, word_embeddings):
    """
    cosine similarity between the dimension-wise max word embeddings of x and y (both dicts with {word: count})
    the max is taken based on the counts in the dict, i.e., can be tfidf weighting or straight average
    """
    vec_x = extrema(np.array([x[word] * word_embeddings[word] for word in x if word in word_embeddings]))
    vec_y = extrema(np.array([y[word] * word_embeddings[word] for word in y if word in word_embeddings]))
    if isinstance(vec_x, float) or isinstance(vec_y, float):
        return 0.  # no dot product with no vectors
    return np.dot(vec_x, vec_y) / (np.linalg.norm(vec_x) * np.linalg.norm(vec_y))


def _simple_max_cos(x, y, word_embeddings):
    """
    cosine similarity between the dimension-wise max word embeddings of x and y (both dicts with {word: count} or just lists of words)
    the max is only taken based on the occurrence of the words
    """
    vec_x = extrema(np.array([word_embeddings[word] for word in x if word in word_embeddings]))
    vec_y = extrema(np.array([word_embeddings[word] for word in y if word in word_embeddings]))
    if isinstance(vec_x, float) or isinstance(vec_y, float):
        return 0.  # no dot product with no vectors
    return np.dot(vec_x, vec_y) / (np.linalg.norm(vec_x) * np.linalg.norm(vec_y))


def _greedy_match(x, y, word_embeddings):
    """
    take the average of the word-wise maximum cosine similarities (x and y can be dicts with {word: count} or just lists of words)
    """
    # construct 2 feature matrices - word_embeddings[word] is already length normalized
    mat_x = np.array([word_embeddings[word] for word in x if word in word_embeddings])
    mat_y = np.array([word_embeddings[word] for word in y if word in word_embeddings])
    if not mat_x.shape[0] or not mat_y.shape[0]:
        return 0.  # no point in multiplying anything if one of the sentences has no (valid) words
    word_sims = mat_x.dot(mat_y.T)
    return 0.5 * (np.mean(np.max(word_sims, axis=0)) + np.mean(np.max(word_sims, axis=1)))


def _weightedmat_greedy_match(x, y, word_embeddings):
    """
    take the average of the word-wise maximum cosine similarities, weight by max idf (x and y both dicts with {word: count})
    """
    # construct 2 feature matrices - word_embeddings[word] is already length normalized
    mat_x = np.array([word_embeddings[word] for word in x if word in word_embeddings])
    mat_y = np.array([word_embeddings[word] for word in y if word in word_embeddings])
    if not mat_x.shape[0] or not mat_y.shape[0]:
        return 0.  # no point in multiplying anything if one of the sentences has no (valid) words
    word_sims = mat_x.dot(mat_y.T)
    # create the weight matrix as the max (mean/min?) idf scores of the words
    vec_x = np.array([[x[word] for word in x if word in word_embeddings]])
    vec_y = np.array([[y[word] for word in y if word in word_embeddings]])
    mat_x = np.tile(vec_x.T, (1, vec_y.shape[1]))
    mat_y = np.tile(vec_y, (vec_x.shape[1], 1))
    weights = np.maximum(mat_x, mat_y)
    weighted_word_sims = weights*word_sims
    return 0.5 * (np.mean(np.max(weighted_word_sims, axis=0)) + np.mean(np.max(weighted_word_sims, axis=1)))


def _weighted_greedy_match(x, y, word_embeddings):
    """
    take the average of the word-wise maximum cosine similarities, weight by max idf (x and y both dicts with {word: count})
    """
    # construct 2 feature matrices - word_embeddings[word] is already length normalized
    mat_x = np.array([word_embeddings[word] for word in x if word in word_embeddings])
    mat_y = np.array([word_embeddings[word] for word in y if word in word_embeddings])
    if not mat_x.shape[0] or not mat_y.shape[0]:
        return 0.  # no point in multiplying anything if one of the sentences has no (valid) words
    word_sims = mat_x.dot(mat_y.T)
    # weight the maximum of the words' similarities with their idf value
    vec_x = np.array([x[word] for word in x if word in word_embeddings])
    vec_y = np.array([y[word] for word in y if word in word_embeddings])
    return 0.5 * (np.inner(vec_x, np.max(word_sims, axis=1)) + np.inner(vec_y, np.max(word_sims, axis=0)))


def compute_sim(x, y, word_embeddings, sim):
    if sim == 'avg':
        return _avg_cos(x, y, word_embeddings)
    elif sim == 'max':
        return _max_cos(x, y, word_embeddings)
    elif sim == 'simple avg':
        return _simple_avg_cos(x, y, word_embeddings)
    elif sim == 'simple max':
        return _simple_max_cos(x, y, word_embeddings)
    elif sim == 'greedy':
        return _greedy_match(x, y, word_embeddings)
    elif sim == 'weighted greedy':
        return _weighted_greedy_match(x, y, word_embeddings)
    elif sim == 'wmdist':
        return _wmdist(x, y, word_embeddings)
    else:
        raise NotImplementedError('Unknown sim %s' % sim)


def transform_word2vec_featmat(docfeats, doc_ids, word_embeddings, avg='avg'):
    # represent each document as an averaged (based on the docfeats, i.e. possibly tfidf weighted) word2vec vector
    # the featmat will have the shape len(doc_ids) x word2vec_dim
    featmat = np.zeros((len(doc_ids), word_embeddings.vector_size))
    for i, did in enumerate(doc_ids):
        if avg == 'avg':
            featmat[i, :] = np.sum([docfeats[did][word] * word_embeddings[word]
                                    for word in docfeats[did] if word in word_embeddings], axis=0)
        elif avg == 'simple avg':
            featmat[i, :] = np.sum([word_embeddings[word] for word in docfeats[did] if word in word_embeddings], axis=0)
        elif avg == 'max':
            featmat[i, :] = extrema(np.array([docfeats[did][word] * word_embeddings[word]
                                              for word in docfeats[did] if word in word_embeddings]))
        elif avg == 'simple max':
            featmat[i, :] = extrema(np.array([word_embeddings[word] for word in docfeats[did] if word in word_embeddings]))
        else:
            raise NotImplementedError('Unknown average %s' % avg)
    fnorm = np.linalg.norm(featmat, axis=1)
    # avoid division by 0
    fnorm[fnorm == 0.] = 1.
    featmat /= fnorm.reshape(featmat.shape[0], 1)
    # catch division by 0
    #featmat[~np.isfinite(featmat)] = 0.
    return featmat


def compute_w2v_K(docids, docfeats, word_embeddings, sim='avg'):
    """
    Input:
        docids: list of document ids (keys of docfeats)
        docfeats: dict with doc_id:{feat:count} (or for simple avg/max or greedy lists of words suffice)
        word_embeddings: (gensim) word embeddings where word_embeddings[word] gives the embedding vector of 'word'
        sim: type of similarity
    Returns:
        symmetric similarity matrix of size len(docids)xlen(docids)
    """
    if sim in ('avg', 'max', 'simple avg', 'simple max'):
        featmat = transform_word2vec_featmat(docfeats, docids, word_embeddings, sim)
        return featmat.dot(featmat.T)
    # compute general similarity matrix
    N = len(docids)
    K = np.zeros((N, N))
    for i, did in enumerate(docids):
        for j in range(i + 1):
            similarity = compute_sim(docfeats[did], docfeats[docids[j]], word_embeddings, sim)
            K[i, j], K[j, i] = similarity, similarity
    if sim == 'wmdist':
        # by default this is a distance not a similarity. the points furthest apart should have a similarity of 0.
        inf_mask = K == -1
        K = K.max() - K
        K[inf_mask] = 0.
    return K


def compute_w2v_K_map(train_ids, test_ids, docfeats, word_embeddings, sim='avg'):
    """
    Input:
        train_ids, test_ids: list of document ids (keys of docfeats)
        docfeats: dict with doc_id:{feat:count} (or for simple avg/max or greedy lists of words suffice)
        word_embeddings: (gensim) word embeddings where word_embeddings[word] gives the embedding vector of 'word'
        sim: type of similarity
    Returns:
        kernel map of size len(test_ids)xlen(train_ids) with similarities of the test to the training docs
    """
    # if linear similarity or variant thereof, we're quicker with a dot product
    if sim in ('avg', 'max', 'simple avg', 'simple max'):
        # transform features into matrix
        featmat_train = transform_word2vec_featmat(docfeats, train_ids, word_embeddings, sim)
        featmat_test = transform_word2vec_featmat(docfeats, test_ids, word_embeddings, sim)
        # compute kernel map
        return featmat_test.dot(featmat_train.T)
    # compute similarity of all test examples to all training examples (normal way)
    K_map = np.array([[compute_sim(docfeats[did_ts], docfeats[did_tr], word_embeddings, sim)
                       for did_tr in train_ids] for did_ts in test_ids])
    if sim == 'wmdist':
        # by default this is a distance not a similarity. the points furthest apart should have a similarity of 0.
        inf_mask = K_map == -1
        K_map = K_map.max() - K_map
        K_map[inf_mask] = 0.
    return K_map
