"""
run_validation.py
"""
from models.reinforced_draggn import ReinforcedDRAGGN
from preprocessor.reader import parse_valid
import tensorflow as tf


def main(_):
    # Parse Train, Test
    trainX, trainX_len, trainX_State, trainY, testX, testX_len, testX_State, testY, word2id, labels = parse_valid()

    # Instantiate Reinforced DRAGGN
    reinforced_draggn = ReinforcedDRAGGN(trainX, trainX_len, trainX_State, trainY, word2id, labels)

    # Fit
    reinforced_draggn.fit(600, validate=True)

    # Eval
    reinforced_draggn.eval(testX, testX_len, testX_State, testY, validate=True)

if __name__ == "__main__":
    tf.app.run()