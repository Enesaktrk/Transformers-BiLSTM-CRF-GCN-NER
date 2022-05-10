# coding=utf-8


class Config(object):
    def __init__(self):
        self.label_file = './data/tag.txt'
        self.train_file = './NER/Mimic/train.tsv'
        self.dev_file = './NER/Mimic/devel.tsv'
        self.test_file = './NER/Mimic/test.tsv'
        self.vocab = './bert-base-uncased/vocab.txt'
        self.max_length = 192
        self.use_cuda = False
        self.gpu = 0
        self.batch_size = 50
        self.bert_path = './bert-base-uncased'
        self.rnn_hidden = 500
        self.bert_embedding = 768
        self.dropout1 = 0.5
        self.dropout_ratio = 0.5
        self.rnn_layer = 1
        self.lr = 0.0001
        self.lr_decay = 0.00001
        self.weight_decay = 0.00005
        self.checkpoint = 'NER/CombinedResult/'
        self.optim = 'Adam'
        self.load_model = False
        self.load_path = None
        self.base_epoch = 10

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':
    con = Config()
    con.update(gpu=8)
    print(con.gpu)
    print(con)
