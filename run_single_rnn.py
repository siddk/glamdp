"""
run_single_rnn.py

Core script for loading, training, and evaluating the Single RNN model for grounding language 
to lifted reward functions.
"""
from models.single_rnn import SingleRNN
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_path", "data/lifted_merged/no_l0/no_l0_train", "Path to training data.")
tf.app.flags.DEFINE_string("test_path", "data/lifted_merged/no_l0/no_l0_test", "Path to testing data.")

def main(_):
    # Create Model
    single_rnn = SingleRNN(FLAGS.train_path, FLAGS.test_path)

    # Fit Model 5 Times, Running Evaluation Epochs 
    for _ in range(5):
        single_rnn.fit()
        single_rnn.eval()
    
    
if __name__ == "__main__":
    tf.app.run()