"""
run_supervised.py
"""
from models.super_draggn import SuperDRAGGN
from preprocessor.reader import parse
import tensorflow as tf


def main(_):
    # Parse Train, Test
    trainX, trainX_len, trainY, testX, testX_len, testY, word2id, labels = parse()

    # Build Supervised DRAGGN
    super_draggn = SuperDRAGGN(trainX, trainX_len, trainY, word2id, labels)

    # Fit
    super_draggn.fit(35)

    # Evaluate
    super_draggn.eval(testX, testX_len, testY)

if __name__ == "__main__":
    tf.app.run()