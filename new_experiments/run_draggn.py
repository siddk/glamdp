"""
run_npi.py

Core script for loading, training, and evaluating the NPI model for grounding language 
to lifted reward functions.
"""
from models.lg_npi import NPI
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Vanilla Dataset
# tf.app.flags.DEFINE_string("means_train_path", "data/vanilla/train_actions.pik", "Path to means training data.")
# tf.app.flags.DEFINE_string("ends_train_path", "data/vanilla/goals", "Path to ends training data.")
# tf.app.flags.DEFINE_string("means_test_path", "data/vanilla/test_actions.pik", "Path to means test data.")
# tf.app.flags.DEFINE_string("ends_test_path", "data/vanilla/goals", "Path to ends test data.")
# tf.app.flags.DEFINE_bool("is_pik", False, "Use pickled version of goals commands.")
# tf.app.flags.DEFINE_string("pik_train", "data/unseen/goals_train.pik", "Path to train pickle file.")
# tf.app.flags.DEFINE_string("pik_test", "data/unseen/goals_test.pik", "Path to test pickle file.")

# Unseen Dataset
tf.app.flags.DEFINE_string("means_train_path", "data/unseen/unseen_train_actions.pik", "Path to means training data.")
tf.app.flags.DEFINE_string("ends_train_path", "data/unseen/goals", "Path to ends training data.")
tf.app.flags.DEFINE_string("means_test_path", "data/unseen/unseen_test_actions.pik", "Path to means test data.")
tf.app.flags.DEFINE_string("ends_test_path", "data/unseen/goals", "Path to ends test data.")
tf.app.flags.DEFINE_bool("is_pik", True, "Use pickled version of goals commands.")
tf.app.flags.DEFINE_string("pik_train", "data/unseen/goals_train.pik", "Path to train pickle file.")
tf.app.flags.DEFINE_string("pik_test", "data/unseen/goals_test.pik", "Path to test pickle file.")

def main(_):
    # Create Model
    npi = NPI(FLAGS.means_train_path, FLAGS.ends_train_path, FLAGS.means_test_path, FLAGS.ends_test_path, is_pik=FLAGS.is_pik,
              pik_train_path=FLAGS.pik_train, pik_test_path=FLAGS.pik_test)

    # Train Model + Evaluate
    for i in range(25):
        print 'ITERATION:', i + 1
        npi.fit()
        npi.eval_means()
        npi.eval_ends()
    
    # Save Model
    npi.saver.save(npi.session, "checkpoints/npi.ckpt")

if __name__ == "__main__":
    tf.app.run()