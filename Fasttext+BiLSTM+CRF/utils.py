import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pickle

from gensim.models import KeyedVectors

import numpy as np
import torch
from torch import Tensor
import chainer
import scipy.sparse as sp
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

PAD_IDX = 0
UNK_IDX = 1
unique_tags = ["B", "I", "O", "PAD"]
'''
unique_tags = [
    "I-Immaterial_anatomical_entity", "I-Organ",
    "B-Organism_substance", "I-Cell", "B-Organism",
    "I-Amino_acid", "B-Tissue", "B-Anatomical_system",
    "B-Cellular_component", "I-Developing_anatomical_structure",
    "B-Pathological_formation", "B-Organism_subdivision", "B-Simple_chemical",
    "B-Immaterial_anatomical_entity", "I-Multi-tissue_structure", "I-Cancer", "I-Pathological_formation",
    "I-Gene_or_gene_product", "B-Multi-tissue_structure", "B-Developing_anatomical_structure", "O",
    "B-Organ", "I-Tissue", "I-Anatomical_system", "I-Cellular_component", "I-Organism", "B-Cell", "B-Cancer",
    "B-Amino_acid",
    "I-Organism_substance", "I-Organism_subdivision", "I-Simple_chemical", "B-Gene_or_gene_product", "PAD"
]
'''
tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}


def predict_sentence(model, sentence, w2i, i2t, device="cuda"):
    encoded_sent = encode_sent(sentence, w2i)
    score, tags = model([encoded_sent])
    tags = decode_tags(tags[0], i2t)
    return score.item(), tags


def load_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    return data


def load_data(filename: str):
    """
    Load the data from the given filename.
    """
    with Path(filename).open('r', encoding="utf-8") as f:
        data = f.read().split('\n')

    return data


def strip_sents_and_tags(sents: List, tags: List):
    tmp_train_sents, tmp_train_tags = [], []
    for sent, tag in zip(sents, tags):
        if sent.strip():
            tmp_train_sents.append(sent.strip())
            tmp_train_tags.append(tag.strip())

    return tmp_train_sents, tmp_train_tags


def encode_tags(sent_tags: List, tag2idx: Dict):
    """
    Replace the tags (O, B-LOC etc.) with the corresponding idx from the
    tag2idx dictionary
    """
    encoded_tags = [tag2idx[token_tag] for token_tag in sent_tags]
    return encoded_tags


def decode_tags(tags: List, idx2tag: Dict):
    """
    Decode the tags by replacing the tag indices with the original tags.
    """
    decoded_tags = [idx2tag[tag_idx] for tag_idx in tags]  # if tag_idx != PAD_IDX
    return decoded_tags


def load_wv(filename: str, limit: Optional[int] = None) -> KeyedVectors:
    """
    Load the fastText pretrained word embeddings from given filename.
    """
    embeddings = KeyedVectors.load_word2vec_format(filename,
                                                   binary=False,
                                                   limit=limit,
                                                   unicode_errors='ignore')
    return embeddings


def encode_sent(sent_tokens: List, word2idx: Dict) -> List:
    """
    Replace the tokens with the corresponding index from `word2idx`
    dictionary.
    """
    encoded_sent = [word2idx.get(token, UNK_IDX) for token in sent_tokens]
    return encoded_sent


def decode_sent(sent: List, idx2word: Dict) -> List:
    """
    Decode the sentence to the original form by replacing token indices
    with the words.
    """
    decoded_sent = [idx2word[token_idx] for token_idx in sent if token_idx != PAD_IDX]
    return decoded_sent


def pad_sequences(sequences: List[List], pad_idx: Optional[int] = 0) -> List[List]:
    """
    Pad the sequences to the maximum length sequence.
    """
    max_len = max([len(seq) for seq in sequences])

    padded_sequence = []
    for seq in sequences:
        seq_len = len(seq)
        pad_len = max_len - seq_len
        padded_seq = seq + [pad_idx] * pad_len
        padded_sequence.append(padded_seq)

    return padded_sequence


def to_tensor(sents: List[List], device: str = "cuda") -> Tensor:
    """
    Pad the sentences and convert them to the torch tensor.
    """
    padded_sents = pad_sequences(sents)
    sent_tensor = torch.tensor(padded_sents, dtype=torch.long, device=device)
    return sent_tensor  # (batch_size, max_seq_len)


def generate_sent_masks(sents: Tensor, lengths: Tensor) -> Tensor:
    """
    Generate the padding masking for given sents from lenghts.
    Assumes lengths are sorted by descending order (batch_iter provides this).
    """
    max_len = lengths[0]
    bs = sents.shape[0]
    mask = torch.arange(max_len).expand(bs, max_len) < lengths.unsqueeze(1)
    return mask.byte()


def batch_iter(data: List[List], batch_size: int, shuffle: bool = False) -> Tuple[List, List]:
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        sents = [e[0] for e in examples]
        tags = [e[1] for e in examples]
        yield sents, tags


def create_text_adjacency_matrix(x):
    """Create adjacency matrix from texts
        Arguments:
            texts (list of list of str): List of documents, each consisting
                of tokenized list of text
        Returns:
            adj (scipy.sparse.coo_matrix): (Node, Node) shape
                normalized adjency matrix.
        """
    # The authors removed words occuring less than 5 times. It is not directory
    # applicable to min_df, so I set bit smaller value
    transformer = TfidfVectorizer(
        max_df=1.0, ngram_range=(1, 1), min_df=3, analyzer='word',
        preprocessor=lambda x: x, tokenizer=lambda x: x,
        norm=None, smooth_idf=False
    )
    freq_doc = transformer.fit_transform(x)
    freq_window = transformer.transform(
        moving_window_window_iterator(x, 20))
    freq_window.data.fill(1)
    mat_pmi = calc_pmi(freq_window)
    adj = sp.bmat([[None, freq_doc], [freq_doc.T, mat_pmi]])
    adj.setdiag(np.ones([adj.shape[0]], dtype=adj.dtype))
    adj.eliminate_zeros()
    # it should already be COO, but behavior of bmat is not well documented
    # so apply it
    adj = adj.tocoo()
    adj = normalize_pygcn(adj)
    # adj = normalize(adj)
    adj = to_chainer_sparse_variable(adj)
    print("Done")
    return adj


def normalize_pygcn(a):
    """ normalize adjacency matrix with normalization-trick. This variant
    is proposed in https://github.com/tkipf/pygcn .
    Refer https://github.com/tkipf/pygcn/issues/11 for the author's comment.
    Arguments:
        a (scipy.sparse.coo_matrix): Unnormalied adjacency matrix
    Returns:
        scipy.sparse.coo_matrix: Normalized adjacency matrix
    """
    # no need to add identity matrix because self connection has already been added
    # a += sp.eye(a.shape[0])
    rowsum = np.array(a.sum(1))
    rowsum_inv = np.power(rowsum, -1).flatten()
    rowsum_inv[np.isinf(rowsum_inv)] = 0.
    # ~D in the GCN paper
    d_tilde = sp.diags(rowsum_inv)
    return d_tilde.dot(a)


def build_word_doc_edges(doc_list):
    # build all docs that a word is contained in
    words_in_docs = defaultdict(set)
    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        for word in words:
            words_in_docs[word].add(i)
    word_doc_freq = {}
    for word, doc_list in words_in_docs.items():
        word_doc_freq[word] = len(doc_list)
    return words_in_docs, word_doc_freq


def init_node_feats(graph, device):
    num_nodes = graph.shape[0]
    identity = sp.identity(num_nodes)
    ind0, ind1, values = sp.find(identity)
    inds = np.stack((ind0, ind1), axis=0)
    node_feats = torch.sparse_coo_tensor(inds, values, device=device, dtype=torch.float)
    return node_feats


def moving_window_window_iterator(sentences, window):
    for sentence in sentences:
        for i in range(0, len(sentence) - window + 1):
            yield sentence[i:i + window]


def calc_pmi(X):
    Y = X.T.dot(X).astype(np.float32)
    Y_diag = Y.diagonal()
    Y.data /= Y_diag[Y.indices]
    Y.data *= X.shape[0]
    for col in range(Y.shape[1]):
        Y.data[Y.indptr[col]:Y.indptr[col + 1]] /= Y_diag[col]
    Y.data = np.maximum(0., np.log(Y.data))
    return Y


def to_chainer_sparse_variable(mat):
    mat = mat.tocoo().astype(np.float32)
    ind = np.argsort(mat.row)
    data = mat.data[ind]
    row = mat.row[ind]
    col = mat.col[ind]
    shape = mat.shape
    # check that adj's row indices are sorted
    assert np.all(np.diff(row) >= 0)
    return chainer.utils.CooMatrix(data, row, col, shape, order='C')
