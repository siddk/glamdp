"""
run_npi.py

Core script for loading, training, and evaluating the NPI model for grounding language 
to lifted reward functions.
"""
from models.lifted_npi import NPI
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_path", "data/lifted_merged/no_l0/no_l0_train", "Path to training data.")
tf.app.flags.DEFINE_string("test_path", "data/lifted_merged/no_l0/no_l0_test", "Path to testing data.")

def main(_):
    # Create Model
    npi = NPI(FLAGS.train_path, FLAGS.test_path)

    # Train Model + Evaluate
    for _ in range(25):
        npi.fit()
        npi.eval()


if __name__ == "__main__":
    tf.app.run()