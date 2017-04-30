"""
run_lifted_rnn.py 
"""
from models.lifted_rnn import LiftedRNN
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("means_train_path", "npi_train_test/L0_npi_train", "Path to means training data.")
tf.app.flags.DEFINE_string("ends_train_path", "npi_train_test/L2_train", "Path to ends training data.")
tf.app.flags.DEFINE_string("means_test_path", "npi_train_test/L0_test", "Path to means test data.")
tf.app.flags.DEFINE_string("ends_test_path", "npi_train_test/L2_test", "Path to ends test data.")
tf.app.flags.DEFINE_string("permuted_ends_test_path", "permuted_ends_test/L2_test", "Path to permuted ends test data.")

def main(_):
    # Create Model
    lifted_rnn = LiftedRNN(FLAGS.means_train_path, FLAGS.ends_train_path, FLAGS.means_test_path, FLAGS.ends_test_path)

    # Fit Model 5 Times, Running Evaluation Epochs 
    for _ in range(5):
        lifted_rnn.fit()
        lifted_rnn.eval_means()
        lifted_rnn.eval_ends()
    
if __name__ == "__main__":
    tf.app.run()