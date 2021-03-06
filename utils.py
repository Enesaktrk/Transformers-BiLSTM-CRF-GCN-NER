# coding=utf-8
import torch
import os
import datetime
import unicodedata


class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def read_corpus(path, max_length, label_dic, vocab):
    """
    :param path
    :param max_length
    :param label_dic
    :return:
    """
    file = open(path, encoding='utf-8')
    content = file.readlines()
    file.close()
    result = []

    tokens, labels = [], []
    for line in content:
        if line == "\n":
            # print(tokens)
            # print(labels)
            if len(tokens) > max_length - 2:
                tokens = tokens[0:(max_length - 2)]
                labels = labels[0:(max_length - 2)]
            tokens_f = ['[CLS]'] + tokens + ['[SEP]']
            label_f = ["<start>"] + labels + ['<eos>']
            input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
            label_ids = [label_dic[i] for i in label_f]
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_length:
                input_ids.append(0)
                input_mask.append(0)
                label_ids.append(label_dic['<pad>'])
            # print(f"max_length : {max_length}")
            # print(input_ids)
            # print(f"input_ids : {len(input_ids)}")
            # print(f"input_mask : {len(input_mask)}")
            # print(label_ids)
            assert len(input_ids) == max_length
            assert len(input_mask) == max_length
            assert len(label_ids) == max_length
            feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids)
            result.append(feature)
            tokens, labels = [], []
            continue
        token, label = line.split("\t")
        tokens.append(token)
        if label[:1] == "\n":
            labels.append("O")
        else:
            labels.append(label[:1])
    #
    # for line in content:
    #     text, label = line.strip().split('|||')
    #     tokens = text.split()
    #     label = label.split()
    #     if len(tokens) > max_length-2:
    #         tokens = tokens[0:(max_length-2)]
    #         label = label[0:(max_length-2)]
    #     tokens_f =['[CLS]'] + tokens + ['[SEP]']
    #     label_f = ["<start>"] + label + ['<eos>']
    #     input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
    #     label_ids = [label_dic[i] for i in label_f]
    #     input_mask = [1] * len(input_ids)
    #     while len(input_ids) < max_length:
    #         input_ids.append(0)
    #         input_mask.append(0)
    #         label_ids.append(label_dic['<pad>'])
    #     assert len(input_ids) == max_length
    #     assert len(input_mask) == max_length
    #     assert len(label_ids) == max_length
    #     feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids)
    #     result.append(feature)
    return result


def save_model(model, epoch, path='result', **kwargs):
    """
    ????????????????????????
    :param model: ??????
    :param path: ????????????
    :param loss: ????????????
    :param last_loss: ??????epoch??????
    :param kwargs: every_epoch or best_epoch
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    if kwargs.get('name', None) is None:
        name = 'epoch_{}'.format(epoch)
        full_name = os.path.join(path, name)
        torch.save(model.state_dict(), full_name)
        print('Saved model at epoch {} successfully'.format(epoch))
        with open('{}/checkpoint'.format(path), 'w') as file:
            file.write(name)
            print('Write to checkpoint')


def load_model(model, path='result', **kwargs):
    if kwargs.get('name', None) is None:
        with open('{}/checkpoint'.format(path)) as file:
            content = file.read().strip()
            name = os.path.join(path, content)
    else:
        name = kwargs['name']
        name = os.path.join(path, name)
    model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(name))
    return model
