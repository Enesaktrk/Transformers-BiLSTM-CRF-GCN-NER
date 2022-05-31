import nltk
import numpy
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from collections import defaultdict, OrderedDict
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

import csv
import itertools

import chainer
from chainer import training
from chainer.training import extensions

from nets import TextGCN
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report

import sklearn
import sys

maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

MAX_LEN = 196
BATCH_SIZE = 32


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


def create_text_adjacency_matrix(texts):
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
    freq_doc = transformer.fit_transform(texts)

    freq_window = transformer.transform(
        moving_window_window_iterator(texts, 20))
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


def normalize(adj):
    """ normalize adjacency matrix with normalization-trick that is faithful to
    the original paper.

    Arguments:
        a (scipy.sparse.coo_matrix): Unnormalied adjacency matrix

    Returns:
        scipy.sparse.coo_matrix: Normalized adjacency matrix
    """
    # no need to add identity matrix because self connection has already been added
    # a += sp.eye(a.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # ~D in the GCN paper
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)


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


def readtsv(path):
    sentences = []
    tags = []
    sent = SentenceFetch(path).getSentences()
    tag = SentenceFetch(path).getTags()
    sentences.extend(sent)
    tags.extend(tag)

    return sentences, tags


'''
def tok_with_labels(sent, text_labels):
    tokenize and keep labels intact
    tok_sent = []
    labels = []
    for word, label in zip(sent, text_labels):
        tok_word = tokenizer.tokenize(word)
        n_subwords = len(tok_word)

        tok_sent.extend(tok_word)
        labels.extend([label] * n_subwords)
    return tok_sent, labels
'''


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, max_grad_norm):
    model = model.train()
    losses = []
    correct_predictions = 0
    for step, batch in enumerate(data_loader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        outputs, y_hat = model(b_input_ids, b_input_mask)

        _, preds = torch.max(outputs, dim=2)
        outputs = outputs.view(-1, outputs.shape[-1])
        b_labels_shaped = b_labels.view(-1).long()
        loss = loss_fn(outputs, b_labels_shaped)
        correct_predictions += torch.sum(preds == b_labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader), np.mean(losses)


def model_eval(model, data_loader, loss_fn, device):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            outputs, y_hat = model(b_input_ids, b_input_mask)

            _, preds = torch.max(outputs, dim=2)
            outputs = outputs.view(-1, outputs.shape[-1])
            b_labels_shaped = b_labels.view(-1).long()
            loss = loss_fn(outputs, b_labels_shaped)
            correct_predictions += torch.sum(preds == b_labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader), np.mean(losses)


def main():
    # get train and test data
    train_paths = ["../datasets/Mimic Radiology Reports/train.tsv"]
    test_paths = ["../datasets/Mimic Radiology Reports/test.tsv"]
    for trainpath, testpath in zip(train_paths, test_paths):
        dataset = trainpath.split("/")
        print("Running GCN for", dataset[2])
        # trainpath = "train.tsv"
        # testpath = "test.tsv"

        #trainpath = "../datasets/NER/JNLPBA/train.tsv"
        #testpath = "../datasets/NER/JNLPBA/test.tsv"

        #trainpath = "../datasets/NER/NCBI-disease/train.tsv"
        #testpath = "../datasets/NER/NCBI-disease/test.tsv"

        # trainpath = "../datasets/NER/s800/train.tsv"
        # testpath = "../datasets/NER/s800/test.tsv"

        # trainpath = "../datasets/NER/BC5CDR-disease/train.tsv"
        # testpath = "../datasets/NER/BC5CDR-disease/test.tsv"

        # trainpath = "../datasets/NER/BC2GM/train.tsv"
        # testpath = "../datasets/NER/BC2GM/test.tsv"

        # trainpath = "../datasets/Mimic Radiology Reports/train.tsv"
        # testpath = "../datasets/Mimic Radiology Reports/test.tsv"

        train_sentences, train_tags = readtsv(trainpath)
        test_sentences, test_tags = readtsv(testpath)
        tags = train_tags + test_tags
        sentences = train_sentences + test_sentences

        # get stopwords
        clinical_stopwords = pd.read_csv("clinical-stopwords.txt", sep='\t', header=None)
        clinical_stopwords = clinical_stopwords[0].values.tolist()

        words = [item for sublist in sentences for item in sublist]

        # back up of instances and labels
        sentences2 = sentences
        tags2 = tags

        unique_words = []
        for word in words:
            if word not in unique_words and word not in clinical_stopwords:
                unique_words.append(word)

        # vocab = unique words in instances
        vocab_len = len(unique_words)
        print(vocab_len)

        '''tok_texts_and_labels = [tok_with_labels(sent, labs) for sent, labs in zip(sentences, tags)]
    
        tok_texts = [tok_label_pair[0] for tok_label_pair in tok_texts_and_labels]
    
        for char in tok_texts:
            print('WordPiece Tokenizer Preview:\n', char)
            break'''

        # kinds of labels
        tag_values = list(set(itertools.chain.from_iterable(tags)))
        tag_values.append("PAD")

        # indexes of kinds of labels
        tag2idx = {t: i for i, t in enumerate(tag_values)}

        # Test train test split with masking
        test1 = list(itertools.chain.from_iterable(tags2))
        labelsInt = []
        for i in test1:
            labelsInt.append(tag2idx.get(i))

        # indexes of labels as integer
        labelsInt = np.array(labelsInt)
        labelsInt = labelsInt.astype(np.int32)

        # convert backup of all sentences to 1d list
        X = list(itertools.chain.from_iterable(sentences2))
        # ??????? how to store a lot of string items as graph -> too much memory usage

        # all the y values -> all the labels as integer -> for train test split
        y_all = numpy.array(labelsInt)
        testRatio = 0.1

        print("Train-test split data")
        # sentences labels
        n_nodes_y = len(y_all)  # total y amount
        n_train_y = int(n_nodes_y * (1.0 - 2 * testRatio))  # amount to be trained
        n_val_y = int(n_nodes_y * testRatio)  # amount for validation

        train_mask = torch.zeros(n_nodes_y, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes_y, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes_y, dtype=torch.bool)

        train_mask[:n_train_y] = True
        val_mask[n_train_y:n_train_y + n_val_y] = True
        test_mask[n_train_y + n_val_y:] = True

        test_data_tags = y_all[test_mask]
        train_data_tags = y_all[train_mask]
        val_data_tags = y_all[val_mask]

        # Create document x term list
        X = numpy.array(X)

        train_data_instances = X[:n_train_y]
        val_data_instances = X[n_train_y:n_train_y + n_val_y]
        test_data_instances = X[n_train_y + n_val_y:]

        # n_Train is amount to be trained
        n_train = int(len(train_data_tags) * (1.0 - (2 * testRatio)))
        n_all = len(train_data_instances) + len(test_data_instances)

        # indexes of train test val
        idx_train = np.array(list(range(n_train)), np.int32)
        idx_val = np.array(list(range(n_train, len(train_data_instances))), np.int32)
        idx_test = np.array(list(range(len(train_data_instances), n_all)), np.int32)

        train_iter = chainer.iterators.SerialIterator(
            idx_train, batch_size=len(train_data_tags), shuffle=False)
        dev_iter = chainer.iterators.SerialIterator(
            idx_val, batch_size=len(val_data_tags), repeat=False, shuffle=False)

        # send train and test data instances to adj
        print("Creating adjacency matrix")
        adj = create_text_adjacency_matrix(train_data_instances.tolist() + test_data_instances.tolist())

        # 3300 x 3300

        # 3400 x 3400
        # Append -1 to  missing parts of labels i.e. Adjacency matrix - labels
        # ( [3400 x 3400] - [3300 x 3300] => append [100 x 100] -1 values so both matrices are same size

        labels = np.concatenate((train_data_tags, test_data_tags, np.full([adj.shape[0] - n_all], -1)))
        labels = labels.astype(np.int32)

        model = TextGCN(adj, labels, 1000, len(tag_values), dropout=0.3)
        '''
        # Create gcn model
        if os.path.exists('../result/best_model.npz'):
            print("Loading model from best_model.npz")
            chainer.serializers.load_npz(os.path.join('result', 'best_model.npz'), model)
        '''
        # create adam optimizer could be AdamW too
        optimizer = chainer.optimizers.Adam(alpha=0.01)
        optimizer.setup(model)

        # initialize the necessary trainer variables
        updater = training.StandardUpdater(train_iter, optimizer, device=-1)
        trigger = training.triggers.EarlyStoppingTrigger(
            monitor='validation/main/loss', patients=12,
            check_trigger=(16, 'epoch'),
            max_trigger=(500, 'epoch'))
        trainer = training.Trainer(updater, trigger, out='result')

        trainer.extend(extensions.Evaluator(dev_iter, model, device=-1),
                       trigger=(3, 'epoch'))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

        # run the model
        trainer.run()

        # predict the labels
        y_pred = model.predict_proba(idx_test)
        y_pred = np.argmax(y_pred, axis=1)

        #print("test data tags ", test_data_tags)
        #print("predicted test data tags", y_pred)

        tag_values.remove("PAD")
        print(classification_report(test_data_tags, y_pred, target_names=tag_values, labels=range(len(tag_values))))
        print("F1 score for GCN", dataset[2]," = ",f1_score(test_data_tags,y_pred,average="macro",labels=range(len(tag_values))))
        print("Precision score for GCN", dataset[2]," = ",precision_score(test_data_tags,y_pred, average="macro", zero_division=0))
        print("Recall score for GCN", dataset[2], " = ", recall_score(test_data_tags,y_pred, average="macro", zero_division=0))

        chainer.serializers.save_npz(
            os.path.join('result', 'best_model_mimic.npz'), model)

        print('Running test...')
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            _, accuracy = model.evaluate(idx_test)
            print("Model accuracy = ", accuracy)

    return 0


class SentenceFetch(object):

    def __init__(self, data):
        self.data = data
        self.sentences = []
        self.tags = []
        self.sent = []
        self.tag = []

        # make tsv file readable
        with open(self.data, "rt", encoding="latin1") as tsv_f:
            reader = csv.reader(tsv_f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) == 0:
                    if len(self.sent) != len(self.tag):
                        break
                    self.sentences.append(self.sent)
                    self.tags.append(self.tag)
                    self.sent = []
                    self.tag = []
                else:
                    self.sent.append(row[0])
                    self.tag.append(row[1])


    def getSentences(self):
        return self.sentences

    def getTags(self):
        return self.tags


if __name__ == '__main__':
    main()
