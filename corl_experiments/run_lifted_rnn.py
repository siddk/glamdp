"""
run_lifted_rnn.py 
"""
from models.lifted_rnn import LiftedRNN
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# Vanilla Dataset
tf.app.flags.DEFINE_string("means_train_path", "data/vanilla/train_actions.pik", "Path to means training data.")
tf.app.flags.DEFINE_string("ends_train_path", "data/vanilla/goals", "Path to ends training data.")
tf.app.flags.DEFINE_string("means_test_path", "data/vanilla/test_actions.pik", "Path to means test data.")
tf.app.flags.DEFINE_string("ends_test_path", "data/vanilla/goals", "Path to ends test data.")
tf.app.flags.DEFINE_bool("is_pik", True, "Use pickled version of goals commands.")
tf.app.flags.DEFINE_string("pik_train", "data/vanilla/train_goals.pik", "Path to train pickle file.")
tf.app.flags.DEFINE_string("pik_test", "data/vanilla/test_goals.pik", "Path to test pickle file.")
tf.app.flags.DEFINE_integer("until", 400, "Number of training examples (for learning curve)")

# Unseen Dataset
# tf.app.flags.DEFINE_string("means_train_path", "data/unseen/unseen_train_actions.pik", "Path to means training data.")
# tf.app.flags.DEFINE_string("ends_train_path", "data/unseen/goals", "Path to ends training data.")
# tf.app.flags.DEFINE_string("means_test_path", "data/unseen/unseen_test_actions.pik", "Path to means test data.")
# tf.app.flags.DEFINE_string("ends_test_path", "data/unseen/goals", "Path to ends test data.")
# tf.app.flags.DEFINE_bool("is_pik", True, "Use pickled version of goals commands.")
# tf.app.flags.DEFINE_string("pik_train", "data/unseen/unseen_train_goals.pik", "Path to train pickle file.")
# tf.app.flags.DEFINE_string("pik_test", "data/unseen/unseen_test_goals.pik", "Path to test pickle file.")

def main(_):
    # Create Model
    lifted_rnn = LiftedRNN(FLAGS.means_train_path, FLAGS.ends_train_path, FLAGS.means_test_path, FLAGS.ends_test_path,
                           is_pik=FLAGS.is_pik, pik_train_path=FLAGS.pik_train, pik_test_path=FLAGS.pik_test,
                           until=FLAGS.until)

    # Fit Model 5 Times, Running Evaluation Epochs 
    print '###' * 10
    print 'UNTIL:', FLAGS.until
    for _ in range(5):
        lifted_rnn.fit()
        lifted_rnn.eval_means()
        lifted_rnn.eval_ends()
    
if __name__ == "__main__":
    tf.app.run()