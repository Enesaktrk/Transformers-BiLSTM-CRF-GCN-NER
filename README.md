# Bert-BiLstm-Crf-Pytorch

use it with python 3.6

Usage 
Change train/dev/test files in config.py
Change bert vocab and bert path in config.py
Change self.checkpoint for save result path in config.py
You can change max-length property in config.py

Run

train : python main.py train --use_cuda=False --batch_size=32
test :  python main.py test --use_cuse=False --batch_size=32
