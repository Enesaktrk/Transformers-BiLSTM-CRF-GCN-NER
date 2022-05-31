import argparse
import gc
import logging
import time
import datetime
import math
import sys
import csv

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_data, strip_sents_and_tags
from utils import encode_sent, encode_tags, load_wv, batch_iter, decode_tags
from config import UNIQUE_TAGS, PAD_IDX, idx2tag, tag2idx
from model import NERTagger
from metrics import flat_f1_score, flat_classification_report
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, accuracy_score

import pickle

from utils import predict_sentence, load_pickle
from crf import CRF

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
dropout_p = 0.3
n_epochs = 20
hidden_dim = 64
num_layers = 2

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def generate_tags(model, data, batch_size=32, device="cuda"):
    """
    Generate the tags (predictions) for the samples in the data.
    """
    all_decoded_targets = []
    all_decoded_preds = []

    for batch in batch_iter(data, batch_size=batch_size, shuffle=False):
        # batch = (b.to(device) for b in batch)
        sents, tags = batch
        scores, pred_tags = model(sents)
        len_test_tags = [len(test_tag) for test_tag in tags]
        cleaned_test_preds = [pred[:l] for l, pred in zip(len_test_tags, pred_tags)]

        gt_tags = [decode_tags(tag, idx2tag) for tag in tags]
        pred_tags = [decode_tags(tag, idx2tag) for tag in cleaned_test_preds]

        all_decoded_targets.extend(gt_tags)
        all_decoded_preds.extend(pred_tags)

    return all_decoded_targets, all_decoded_preds


def train_step(model, loss_fn, optimizer, train_data, batch_size=32, device="cuda"):
    """
    Train the model for 1 epoch.
    """
    total_loss = 0.0
    model.train()
    start_time = time.time()
    total_step = math.ceil(len(train_data) / batch_size)

    for step, batch in enumerate(batch_iter(train_data, batch_size=batch_size, shuffle=True)):
        if step % 250 == 0 and not step == 0:
            elapsed_since = time.time() - start_time
            logger.info("Batch {}/{}\tElapsed since: {}".format(step, total_step,
                                                                str(datetime.timedelta(seconds=round(elapsed_since)))))
        # batch = (b.to(device) for b in batch)
        sents, tags = batch
        optimizer.zero_grad()
        train_loss = model.loss(sents, tags)
        total_loss += train_loss.item()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    avg_train_loss = total_loss / total_step
    return avg_train_loss


def eval_step(model, loss_fn, data, batch_size=32, device="cuda"):
    """
    Evaluate the model for the given data_loader.
    """
    total_loss = 0.0
    model.eval()
    total_step = math.ceil(len(data) / batch_size)

    for batch in batch_iter(data, batch_size=batch_size, shuffle=False):
        # batch = (b.to(device) for b in batch)
        sents, tags = batch
        eval_loss = model.loss(sents, tags)
        total_loss += eval_loss.item()

    average_eval_loss = total_loss / total_step
    return average_eval_loss


def train(model, loss_fn, optimizer, train_dl, valid_dl, n_epochs=1, device="cuda"):
    """
    Training loop.
    """
    print("...Training for {} epochs...".format(n_epochs))
    print("Number of training samples: ", len(train_dl))
    train_losses = []
    if valid_dl is not None:
        valid_losses = []

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = train_step(model, loss_fn, optimizer, train_dl, device=device)
        train_losses.append(train_loss)

        elapsed_time = time.time() - start_time
        logger.info("Epoch {}/{} is done. Took: {} Loss: {:.5f}".format(epoch + 1,
                                                                        n_epochs,
                                                                        str(datetime.timedelta(
                                                                            seconds=round(elapsed_time))),
                                                                        train_loss))

        valid_loss = eval_step(model, loss_fn, valid_dl, device=device)
        valid_losses.append(valid_loss)
        print("Validation Loss: {:.5f}".format(valid_loss))
        val_targets, val_preds = generate_tags(model, valid_dl, device=device)
        print("Validation f1-score: ", flat_f1_score(val_targets, val_preds, average="macro"))

        print("=" * 50)

    return train_losses, valid_losses


def readtsv(path):
    sentences = []
    tags = []
    sent = SentenceFetch(path).getSentences()
    tag = SentenceFetch(path).getTags()
    sentences.extend(sent)
    tags.extend(tag)

    return sentences, tags


def main():
    train_paths = ["../../datasets/Mimic Radiology Reports/mimic labeled/train.tsv"]
    test_paths = ["../../datasets/Mimic Radiology Reports/mimic labeled/test.tsv"]
    for trainpath, testpath in zip(train_paths, test_paths):
        gc.collect()
        torch.cuda.empty_cache()
        dataset = trainpath.split("/")
        print("Running bilstm crf fasttext for", dataset[3])

        # trainpath = "../fastText_bilstm_train.tsv"
        # testpath = "../test.tsv"
        # trainpath = "../datasets/NER/JNLPBA/train.tsv"
        # testpath = "../datasets/NER/JNLPBA/test.tsv"

        # trainpath = "../datasets/Mimic Radiology Reports/train.tsv"
        # testpath = "../datasets/Mimic Radiology Reports/test.tsv"

        train_sents, train_tags = readtsv(trainpath)
        test_sentences, test_tags = readtsv(testpath)

        test_sents = [item for sublist in test_sentences for item in sublist]
        test_data_tags = [item for sublist in test_tags for item in sublist]

        train_sents = [x for x in train_sents if x != []]
        train_tags = [x for x in train_tags if x != []]

        n_train = len(train_sents)
        val_ratio = 0.1
        n_val = int(n_train * val_ratio)
        n_test = len(test_sentences)

        valid_sents = train_sents[n_train - n_val:n_train]
        valid_tags = train_tags[n_train - n_val:n_train]

        train_sents = train_sents[:n_train - n_val]
        train_tags = train_tags[:n_train - n_val]

        logger.info(f"Total train sents/tags: {n_train}/{len(train_tags)}")
        logger.info(f"Total valid sents/tags: {n_val}/{len(valid_tags)}")

        # Replace the tags with the indices
        tag2idx = {tag: idx for idx, tag in enumerate(UNIQUE_TAGS)}
        idx2tag = {idx: tag for tag, idx in tag2idx.items()}

        train_tags_idx = [encode_tags(tags, tag2idx) for tags in train_tags]
        valid_tags_idx = [encode_tags(tags, tag2idx) for tags in valid_tags]

        ############### = > ok

        # Load the pretrained word embeddings.

        # args["w2v_file"] = "cc.tr.300.vec"

        # w2v_fn = "muse.wiki.en.vec"
        w2v_fn = "cc.en.300.vec"
        logger.info(f"Loading the pretrained word embeddings from {w2v_fn}")

        word_vectors = load_wv(w2v_fn)
        # We will add 2 additional vectors for the padding & unknown tokens.
        # padding_idx will be the first index of the word vector matrix.
        # unknown_idx will be the second index of the word vector matrix.
        additional_vectors = np.zeros(shape=(2, 300))
        # remove unk
        index2word = ["<pad>", "<unk>"] + word_vectors.index_to_key
        word2index = {word: index for index, word in enumerate(index2word)}
        weights = np.concatenate((additional_vectors, word_vectors.vectors))

        weights = torch.from_numpy(weights).float()

        # embedding = nn.Embedding.from_pretrained(weights, padding_idx=0)
        # embedding(torch.LongTensor([2])) # embeddings for token '.'

        # Replace the sent_tokens with the indices from word2index.
        train_sents_idx = [encode_sent(sent, word2index) for sent in train_sents]
        valid_sents_idx = [encode_sent(sent, word2index) for sent in valid_sents]

        # Final form of the data
        """
        [
            [
                [sent1_token1_idx, sent1_token2_idx, sent1_token3_idx, ...], 
                [sent_1_tag1_idx, sent1_tag2_idx, sent1_tag3_idx, ...]
            ],
            [
                [sent2_token1_idx, sent2_token2_idx, sent2_token3_idx, ...], 
                [sent2_tag1_idx, sent2_tag2_idx, sent2_tag3_idx, ...]
            ],
            ...
        ]
        """
        train_data = list(zip(*[train_sents_idx, train_tags_idx]))
        valid_data = list(zip(*[valid_sents_idx, valid_tags_idx]))

        model = NERTagger(
            hidden_size=hidden_dim,
            output_size=len(UNIQUE_TAGS),
            num_layers=num_layers,
            bidirectional=True,
            dropout_p=dropout_p,
            weights=weights,
            device="cuda"
        )

        model.to(device)
        print(model)

        # Defining the optimizer
        optimizer = optim.Adam(model.parameters())
        # Defining the loss function
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        train_losses, valid_losses = train(model, criterion, optimizer, train_data, valid_dl=valid_data,
                                           n_epochs=n_epochs, device="cuda")

        logger.info("Saving the trained model to the {}".format("NER with fasttext embedding"))
        params = {
            "model": model
        }
        torch.save(params, "NER with fasttext embedding")

        targets, preds = generate_tags(model, valid_data, device="cuda")
        # print(flat_classification_report(targets, preds))

        # testing

        model.eval()
        # model.device = "cpu"
        print(model)
        model.crf.device = "cuda"
        model.device = "cuda"

        # model.to(device)
        w2i_file = "word2index.pkl"

        w2i = load_pickle(w2i_file)
        sentence = (
            "beyi̇n bt i̇nceleme tekni̇k i̇nceleme 5 mm kesi̇t kalınlığında yapılmıştır. "
            "pons ve mezensefalon normal i̇zlenmektedi̇r ıv. ventri̇kül orta hatta ve normal boyutlarındadır."
            "i̇i̇i̇ ventri̇kül normal boyutlardadır. lateral ventri̇küller normal boyutlardadır."
            " bazal gangli̇onlar normal i̇zlenmektedi̇r serebellar hemısferler, serebral korti̇kal yapılar normal görünümdedi̇r si̇sterna magna ön arka çapı 15 mm ölçülmüş olup artmıştır mega si̇sterna magna . "
            "krani̇um kemi̇kleri̇ normal i̇zlenmektedi̇r"
        )

        sentence_tokens = sentence.split()
        score, tags = predict_sentence(model, test_sents, w2i, idx2tag)
        # score, tags = predict_sentence(model, sentence_tokens, w2i, idx2tag)

        print(classification_report(test_data_tags, tags, target_names=["B", "I", "O"]))
        print("Macro F1 score for bilstm crf fasttext ", dataset[3], " = ",
              f1_score(test_data_tags, tags, average="macro"))
        print("Recall score for bilstm crf fasttext ", dataset[3], " = ",
              recall_score(test_data_tags, tags, average="macro"))
        print("Precision score  for bilstm crf fasttext ", dataset[3], " = ",
              precision_score(test_data_tags, tags, average="macro"))
        print("Accuracy score for bilstm crf fasttext ", dataset[3], " = ",
              accuracy_score(test_data_tags, tags))
        '''
        for token, tag in zip(test_sents, tags):
            print("{:<15}{:<5}".format(token, tag))
        '''


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


if __name__ == "__main__":
    main()
