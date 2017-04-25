"""
run_npi.py

Core script for loading, training, and evaluating the NPI model for grounding language 
to lifted reward functions.
"""
from models.lg_npi import NPI
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("means_train_path", "npi_train_test/L0_npi_train", "Path to means training data.")
tf.app.flags.DEFINE_string("ends_train_path", "npi_train_test/L2_train", "Path to ends training data.")
tf.app.flags.DEFINE_string("means_test_path", "npi_train_test/L0_test", "Path to means test data.")
tf.app.flags.DEFINE_string("ends_test_path", "npi_train_test/L2_train", "Path to ends test data.")

def main(_):
    # Create Model
    npi = NPI(FLAGS.means_train_path, FLAGS.ends_train_path, FLAGS.means_test_path, FLAGS.ends_test_path)

    # Train Model + Evaluate
    for _ in range(25):
        npi.fit()
        npi.eval()


if __name__ == "__main__":
    tf.app.run()