'''
Trains the RNN on provided training data. Runs the RNN on the predicted "ends" language produced by predict_class.rf

'''

from models.single_rnn import SingleRNN
import numpy as np
from sys import argv
from argparse import ArgumentParser
from predict_class import load_data

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--train", help="training data for RNN.")
    parser.add_argument("--test", help="test data for RNN.")
    parser.add_argument("--results", help="results log file")
    return parser.parse_args(args)

def run(args):
    single_rnn = SingleRNN(args.train, args.test)
    # Fit Model 5 Times, Running Evaluation Epochs
    for _ in range(5):
        single_rnn.fit()
        rf_acc = single_rnn.eval()

    with open(args.results, 'w') as f:
        f.write("train data: {} \ntest data: {} \nreward function prediction accuracy: {}".format(args.train, args.test, rf_acc))


    #TODO: output reward functions for action sequence edit distance?

if __name__=="__main__":
    run(parse(argv[1:]))
