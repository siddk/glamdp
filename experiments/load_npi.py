"""
load_npi.py

Core script for loading and evaluating the NPI model for grounding language 
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
    npi = NPI(FLAGS.means_train_path, FLAGS.ends_train_path, FLAGS.means_test_path, FLAGS.ends_test_path, 
              restore='checkpoints/npi.ckpt')

    # Evaluate Model
    npi.eval_means()
    npi.eval_means_all()
    npi.eval_ends()

if __name__ == "__main__":
    tf.app.run()