'''
Script for training the classifier RNN, the NPI, and the CCG models on their corresponding data sets, testing the individual models,
and testing the full pipeline.

'''
from models.classifier_rnn import ClassifierRNN
from models.lifted_npi import NPI
import tensorflow as tf

#loading data
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_npi", "data/combined_rss_data/ends/no_l0_train", "Path to training data for NPI.")
tf.app.flags.DEFINE_string("test_npi", "data/combined_rss_data/ends/no_l0_test", "Path to test data for NPI.")
tf.app.flags.DEFINE_string("train_classifier", "data/combined_rss_data/combined/train", "Path to training data for classifier.")
tf.app.flags.DEFINE_string("test_classifier", "data/combined_rss_data/combined/test", "path to test data for classifier")
tf.app.flags.DEFINE_string("train_ccg", "data/combined_rss_data/means/means_train", "path to training data for CCG.")
tf.app.flags.DEFINE_string("test_ccg", "data/combined_rss_data/means/means_train", "path to training data for CCG.")
tf.app.flags.DEFINE_string("test_all", "data/combined_rss_data/combined_test", "path to testing data for full pipeline.")

def main(_):

    classifier_rnn = ClassifierRNN(FLAGS.train_classifier, FLAGS.test_classifier)

    classifier_rnn.fit()


if __name__ == "__main__":
    tf.app.run()
