'''
Trains classifier RNN on combined means/ends training data, scores test data
produces output sets of predicted ends/means data to run the NPI and the CCG on
'''

from argparse import ArgumentParser
from sys import argv
from models.classifier_rnn import ClassifierRNN
import numpy as np

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--train", help="training data for classifier")
    parser.add_argument("--test", help="combined ends/means test data")
    parser.add_argument("--means", help="output location for means")
    parser.add_argument("--ends", help="output location for ends")
    return parser.parse_args(args)

#save strings to file
def save_data(outfile, data):
    with open(outfile, 'w') as textfile:
        outstr = "\n".join(data)
        textfile.write(outstr)
        print "saved to {}".format(outfile)

def load_data(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f]

def load_labels(filepath):
    with open(filepath, 'r') as f:
        return np.array([1 if line.strip() == "E" else 0 for line in f])

def run(args):
    #create classifier RNN, train model
    classifier_rnn = ClassifierRNN(args.train, args.test)
    #classifier_rnn.fit()

    #load test data and labels from full corpus
    test_data = load_data(args.test + ".en")
    test_ml = load_data(args.test + ".ml")
    test_labels = load_labels(args.test + "_labels.txt")

    #score all commands, assign them to predicted class

    predicted = []
    for nl in test_data:
        pred, score = classifier_rnn.score(nl)
        predicted.append(pred)

    #report accuracy
    predicted = np.array(predicted)

    accuracy = np.sum(np.equal(test_labels, predicted))/float(len(predicted))

    print "means/ends prediction accuracy: {}".format(accuracy)

    #split data into parts
    combined = zip(test_data, test_ml, predicted)

    predicted_ends = [dat for dat in combined if dat[2] == 1] #filtering commands
    predicted_means = [dat for dat in combined if dat[2] == 0]

    pred_ends_nl, pred_ends_ml, _ = zip(*predicted_ends)
    pred_means_nl, pred_means_ml, _ = zip(*predicted_means)

    save_data(args.ends + ".en", pred_ends_nl)
    save_data(args.ends + ".ml", pred_ends_ml)
    save_data(args.means + ".en", pred_means_nl)
    save_data(args.means + ".ml", pred_means_ml)


if __name__=="__main__":
    run(parse(argv[1:]))
