"""
reader.py
"""
import numpy as np
import pickle

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


def parse_goals():
    with open('data/goals.en', 'r') as f:
        nl_commands = map(lambda x: x.strip().split(), f.readlines())

    with open('data/goals.ml', 'r') as f:
        ml_commands = map(lambda x: x.strip(), f.readlines())

    tr_x, tr_y, ts_x, ts_y = nl_commands[:700], ml_commands[:700], nl_commands[700:], ml_commands[700:]

    # Get Lengths
    train_x_len, test_x_len = map(lambda x: len(x), tr_x), map(lambda x: len(x), ts_x)

    # Get Max Len
    max_len = max(max(train_x_len), max(test_x_len))

    # Build Vocabulary
    word2id = {w: i for i, w in enumerate([PAD] + list(set(reduce(lambda x, y: x + y, tr_x + ts_x))))}

    # Build Output Vocabulary
    labels = {l: i for i, l in enumerate(list(set(tr_y)))}

    # Build Vectors
    trX, trX_len = np.zeros([len(tr_x), max_len], dtype=int), np.zeros([len(tr_x)], dtype=int)
    tsX, tsX_len = np.zeros([len(ts_x), max_len], dtype=int), np.zeros([len(ts_x)], dtype=int)
    trY, tsY = np.zeros([len(tr_y)], dtype=int), np.zeros([len(ts_y)], dtype=int)

    for i, line in enumerate(tr_x):
        for j, word in enumerate(line):
            trX[i][j] = word2id[word]
        trX_len[i] = train_x_len[i]
        trY[i] = labels[tr_y[i]]

    for i, line in enumerate(ts_x):
        for j, word in enumerate(line):
            tsX[i][j] = word2id[word]
        tsX_len[i] = test_x_len[i]
        tsY[i] = labels[ts_y[i]]

    return trX, trX_len, trY, tsX, tsX_len, tsY, word2id, labels


def parse_multi_goals():
    with open('data/goals.en', 'r') as f:
        nl_commands = map(lambda x: x.strip().split(), f.readlines())

    with open('data/goals.ml', 'r') as f:
        ml_commands = map(lambda x: x.strip(), f.readlines())

    tr_x, tr_y, ts_x, ts_y = nl_commands[:700], ml_commands[:700], nl_commands[700:], ml_commands[700:]

    # Get Lengths
    train_x_len, test_x_len = map(lambda x: len(x), tr_x), map(lambda x: len(x), ts_x)

    # Get Max Len
    max_len = max(max(train_x_len), max(test_x_len))

    # Build Vocabulary
    word2id = {w: i for i, w in enumerate([PAD] + list(set(reduce(lambda x, y: x + y, tr_x + ts_x))))}

    # Build Output Programs/Args
    programs, arguments = {}, {}
    for example in tr_y + ts_y:
        pairs = example.split()
        if len(pairs) == 2:
            if pairs[0] not in programs:
                programs[pairs[0]] = len(programs)
            elif pairs[1] not in arguments:
                arguments[pairs[1]] = len(arguments)
        elif len(pairs) == 4:
            if pairs[0] + "_" + pairs[2] not in programs:
                programs[pairs[0] + "_" + pairs[2]] = len(programs)
            elif pairs[1] + "_" + pairs[3] not in arguments:
                arguments[pairs[1] + "_" + pairs[3]] = len(arguments)

    # Build Vectors
    trX, trX_len = np.zeros([len(tr_x), max_len], dtype=int), np.zeros([len(tr_x)], dtype=int)
    tsX, tsX_len = np.zeros([len(ts_x), max_len], dtype=int), np.zeros([len(ts_x)], dtype=int)
    trY, tsY = np.zeros([len(tr_y), 2], dtype=int), np.zeros([len(ts_y), 2], dtype=int)

    for i, line in enumerate(tr_x):
        for j, word in enumerate(line):
            trX[i][j] = word2id[word]
        trX_len[i] = train_x_len[i]
        output = tr_y[i].split()
        if len(output) == 2:
            trY[i][0] = programs[output[0]]
            trY[i][1] = arguments[output[1]]
        elif len(output) == 4:
            trY[i][0] = programs[output[0] + "_" + output[2]]
            trY[i][1] = arguments[output[1] + "_" + output[3]]

    for i, line in enumerate(ts_x):
        for j, word in enumerate(line):
            tsX[i][j] = word2id[word]
        tsX_len[i] = test_x_len[i]
        output = ts_y[i].split()
        if len(output) == 2:
            tsY[i][0] = programs[output[0]]
            tsY[i][1] = arguments[output[1]]
        elif len(output) == 4:
            tsY[i][0] = programs[output[0] + "_" + output[2]]
            tsY[i][1] = arguments[output[1] + "_" + output[3]]

    return trX, trX_len, trY, tsX, tsX_len, tsY, word2id, programs, arguments


def parse_valid():
    with open('data/validation_function.pik', 'r') as f:
        train, test = pickle.load(f)

    # Build X, Y
    train_x, train_x_state, train_y = map(lambda x: list(x), zip(*train))
    test_x, test_x_state, test_y = map(lambda x: list(x), zip(*test))

    # Labels
    labels = {w: i for i, w in enumerate(reduce(lambda x, y: x | y, train_y + test_y))}

    # Split train_x, test_x
    train_x, test_x = map(lambda x: x.lower().split(), train_x), map(lambda x: x.lower().split(), test_x)

    # Get Lengths
    train_x_len, test_x_len = map(lambda x: len(x), train_x), map(lambda x: len(x), test_x)

    # Get Max Len
    max_len = max(max(train_x_len), max(test_x_len))

    # Build Vocabulary
    word2id = {w: i for i, w in enumerate([PAD] + list(set(reduce(lambda x, y: x + y, train_x + test_x))))}

    # Build Vectors
    trX, trX_len = np.zeros([len(train_x), max_len], dtype=int), np.zeros([len(train_x)], dtype=int)
    tsX, tsX_len = np.zeros([len(test_x), max_len], dtype=int), np.zeros([len(test_x)], dtype=int)
    trY, tsY = [[] for _ in range(len(train_x))], [[] for _ in range(len(test_y))]
    trXS, tsXS = np.array(train_x_state, dtype=int), np.array(test_x_state, dtype=int)

    for i, line in enumerate(train_x):
        for j, word in enumerate(line):
            trX[i][j] = word2id[word]
        trX_len[i] = train_x_len[i]
        trY[i] += map(lambda x: labels[x], list(train_y[i]))

    for i, line in enumerate(test_x):
        for j, word in enumerate(line):
            tsX[i][j] = word2id[word]
        tsX_len[i] = test_x_len[i]
        tsY[i] += map(lambda x: labels[x], list(test_y[i]))

    return trX, trX_len, trXS, trY, tsX, tsX_len, tsXS, tsY, word2id, labels