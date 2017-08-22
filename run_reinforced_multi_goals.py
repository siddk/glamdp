"""
run_reinforced_multi_goals.py
"""
from models.reinforced_multi_draggn import ReinforcedMultiDRAGGN
from preprocessor.reader import parse_multi_goals
import tensorflow as tf


def main(_):
    # Parse Train, Test
    trainX, trainX_len, trainY, testX, testX_len, testY, word2id, programs, arguments = parse_multi_goals()


    # Instantiate Reinforced DRAGGN
    reinforced_draggn = ReinforcedMultiDRAGGN(trainX, trainX_len, trainY, word2id, programs, arguments)

    # Fit
    reinforced_draggn.fit(1600)

    reinforced_draggn.eval(testX, testX_len, testY, mode='test')

if __name__ == "__main__":
    tf.app.run()
