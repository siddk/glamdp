"""
run_classifier.py

Core script for loading, training, and evaluating the Single RNN model for classifing language as goal- or action- oriented
"""
from models.classifier_rnn import ClassifierRNN
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_path", "data/language_classifier/classifier_train.txt", "Path to training data.")
tf.app.flags.DEFINE_string("test_path", "data/language_classifier/classifier_test.txt", "Path to testing data.")

def main(_):
    # Create Model
    classifier = ClassifierRNN(FLAGS.train_path, FLAGS.test_path)
    
    #running one epoch for now
    classifier.fit()
    classifier.eval()




if __name__ == "__main__":
    tf.app.run()
