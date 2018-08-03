# QANet
A Pytorch implementation of [QANet](https://arxiv.org/pdf/1804.09541.pdf)  
The code is mostly based on the two repositories:
[hengruo/QANet-pytorch](https://github.com/hengruo/QANet-pytorch)
[NLPLearn/QANet](https://github.com/NLPLearn/QANet)

## Performance
| Training epochs / Steps | BatchSize | HiddenSize | Attention Heads |  EM  |  F1  |
|:-----------------------:|:---------:|:----------:|:---------------:|:----:|:----:|
|      12.8 / 35,000     |     32    |     96     |        1        | 69.0 | 78.6 |
|      22 / 60,000       |     32    |     96     |        1        | 69.7 | 79.2 |
|      12.8 / 93,200     |     12    |     128    |        8        | 70.3 | 79.7 |
|      22 / 160,160      |     12    |     128    |        8        | 70.7 | 80.0 |

*The results of hidden size 128 with 8 heads were run with 12 batches.

## Requirements
  * python 3.6
  * pytorch 0.4.0
  * tqdm
  * spacy 2.0.11
  * tensorboardX
  * absl-py

## Usage
Download and preprocess the data
```bash
# download SQuAD and Glove
$ sh download.sh
# preprocess
$ python3.6 main.py --mode data
```

Train the model
```bash
# model/model.pt will be generated every epoch
$ python3.6 main.py --mode train
```
## Tensorboard
```bash
# Run tensorboard for visualisation
$ tensorboard --logdir ./log/
```
## TODO
- [X] Add Exponential Moving Average
- [X] Reach the performance of the paper with hidden size 96, 1 head.
- [X] Test on hidden size 128, 8 head.
