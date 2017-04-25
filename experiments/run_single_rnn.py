"""
run_single_rnn.py

Core script for loading, training, and evaluating the Single RNN model for grounding language 
to grounded reward functions.
"""
from models.single_rnn import SingleRNN
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("means_train_path", "npi_train_test/L0_rnn_train", "Path to means training data.")
tf.app.flags.DEFINE_string("ends_train_path", "npi_train_test/L2_train", "Path to ends training data.")
tf.app.flags.DEFINE_string("means_test_path", "npi_train_test/L0_test", "Path to means test data.")
tf.app.flags.DEFINE_string("ends_test_path", "npi_train_test/L2_train", "Path to ends test data.")

def main(_):
    # Create Model
    single_rnn = SingleRNN(FLAGS.means_train_path, FLAGS.ends_train_path, FLAGS.means_test_path, FLAGS.ends_test_path)

    # Fit Model 5 Times, Running Evaluation Epochs 
    for _ in range(5):
        single_rnn.fit()
        single_rnn.eval_means()
        single_rnn.eval_ends()
    
if __name__ == "__main__":
    tf.app.run()