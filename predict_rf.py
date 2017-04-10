'''
Trains the NPI on provided training data. Runs the NPI on the predicted "ends" language produced by predict_class.rf

'''

from models.lifted_npi import NPI
import numpy as np
from sys import argv
from argparse import ArgumentParser
from predict_class import load_data

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--train", help="training data for NPI.")
    parser.add_argument("--test", help="test data for NPI.")
    parser.add_argument("--pred_ends", help="predicted ends language and machine language/action traces.")
    parser.add_argument("--results", help="results log file") #TODO: this
    return parser.parse_args(args)

def run(args):

    npi = NPI(args.train, args.test)

    for _ in range(25):
        npi.fit()
        npi.eval()

    #perform inference on predicted ends data
    pred_ends_nl = load_data(args.pred_ends + ".en")
    pred_ends_ml = load_data(args.pred_ends + ".ml")

    num_correct = 0
    for nl, ml in zip(pred_ends_nl, pred_ends_ml):
        pred_rf = npi.score_nl(nl)
        if pred_rf == ml:
            num_correct += 1

    print "Accuracy on predicted means data: {}".format(float(num_correct)/len(pred_ends_ml))


if __name__=="__main__":
    run(parse(argv[1:]))
