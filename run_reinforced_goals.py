"""
run_reinforced_goals
"""
from models.reinforced_draggn import ReinforcedDRAGGN
from preprocessor.reader import parse_goals
import tensorflow as tf


def main(_):
    # Parse Train, Test
    trainX, trainX_len, trainY, testX, testX_len, testY, word2id, labels = parse_goals()

    # Instantiate Reinforced DRAGGN
    reinforced_draggn = ReinforcedDRAGGN(trainX, trainX_len, trainY, word2id, labels)

    # Fit
    reinforced_draggn.fit(600)

    # Eval
    reinforced_draggn.eval(testX, testX_len, testY)

if __name__ == "__main__":
    tf.app.run()