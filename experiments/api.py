"""
api.py 

Starts a lightweight Flask API Server, that can be queried via calls to CURL
"""
from flask import Flask, request, jsonify
from models.lg_npi import NPI
import sys
import tensorflow as tf

app = Flask(__name__)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("means_train_path", "npi_train_test/L0_npi_train", "Path to means training data.")
tf.app.flags.DEFINE_string("ends_train_path", "npi_train_test/L2_train", "Path to ends training data.")
tf.app.flags.DEFINE_string("means_test_path", "npi_train_test/L0_test", "Path to means test data.")
tf.app.flags.DEFINE_string("ends_test_path", "npi_train_test/L2_train", "Path to ends test data.")

# Create Model
npi = NPI(FLAGS.means_train_path, FLAGS.ends_train_path, FLAGS.means_test_path, FLAGS.ends_test_path, 
            restore='checkpoints/npi.ckpt')

@app.route('/model')
def model():
    nl_command = request.args.get('command')
    x = npi.score_nl(nl_command)
    return x + "\n"

if __name__ == "__main__":
    app.run(host=('0.0.0.0'))