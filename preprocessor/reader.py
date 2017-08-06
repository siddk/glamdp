"""
reader.py
"""
import numpy as np

DATA_PATH = 'data/'
PAD = "<<PAD>>"


def parse():
    train_x, train_y, idx = [], [], 0
    with open(DATA_PATH + 'train.txt', 'r') as f:
        for line in f:
            if idx == 0:
                train_x.append(line.lower().strip().split())
            elif idx == 1:
                train_y.append(line.lower().strip())
            idx = (idx + 1) % 3

    test_x, test_y, idx = [], [], 0
    with open(DATA_PATH + 'test.txt', 'r') as f:
        for line in f:
            if idx == 0:
                test_x.append(line.lower().strip().split())
            elif idx == 1:
                test_y.append(line.lower().strip())
            idx = (idx + 1) % 3

    # Assert Output Set is Same
    assert(set(train_y) == set(test_y))

    # Get Lengths
    train_x_len, test_x_len = map(lambda x: len(x), train_x), map(lambda x: len(x), test_x)

    # Get Max Len
    max_len = max(max(train_x_len), max(test_x_len))

    # Build Vocabulary
    word2id = {w: i for i, w in enumerate([PAD] + list(set(reduce(lambda x, y: x + y, train_x + test_x))))}

    # Build Output Vocabulary
    labels = {l: i for i, l in enumerate(list(set(test_y)))}

    # Build Vectors
    trX, trX_len = np.zeros([len(train_x), max_len], dtype=int), np.zeros([len(train_x)], dtype=int)
    tsX, tsX_len = np.zeros([len(test_x), max_len], dtype=int), np.zeros([len(test_x)], dtype=int)
    trY, tsY = np.zeros([len(train_y)], dtype=int), np.zeros([len(test_y)], dtype=int)

    for i, line in enumerate(train_x):
        for j, word in enumerate(line):
            trX[i][j] = word2id[word]
        trX_len[i] = train_x_len[i]
        trY[i] = labels[train_y[i]]

    for i, line in enumerate(test_x):
        for j, word in enumerate(line):
            tsX[i][j] = word2id[word]
        tsX_len[i] = test_x_len[i]
        tsY[i] = labels[test_y[i]]

    return trX, trX_len, trY, tsX, tsX_len, tsY, word2id, labels