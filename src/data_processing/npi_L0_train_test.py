#splitting the segmented/annotated NPI train/test data, ensuring that it's synchronized with the RNN train/test data

from argparse import ArgumentParser

from sys import argv
import random as r


#yes my arguments are stupidly long but I just wanna be clear okay

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--seg_actions", help="segmented actions for NPI")
    parser.add_argument("--seg_rf", help="corresponding reward functions for segmented data")
    parser.add_argument("--seg_en", help="natural language for segmented data")
    parser.add_argument("--npi_test_frac", help="fraction of SEGMENTED data that is train data", type=float)
    parser.add_argument("--unseg_en", help="non-segmented RNN english data")
    parser.add_argument("--unseg_rf", help="reward functions for training single RNN")
    parser.add_argument("--out", help="directory of output split")
    return parser.parse_args(args)

#list of tuples, saves them to .en and .ml files
def save_pc(outfile, pc):
    en, ml = zip(*pc)
    with open(outfile + ".en", 'w') as f:
        f.write("\n".join(en))

    with open(outfile + ".ml", 'w') as f:
        f.write("\n".join(ml))

def save_strings(outfile, strings):
    with open(outfile, 'w') as f:
        f.write("\n".join(strings))

    print "saved to {}".format(outfile)

#file name to list of strings
def load_strings(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def run(args):

    #open segmented data

    seg_actions = load_strings(args.seg_actions)
    seg_rf = load_strings(args.seg_rf)
    seg_en = load_strings(args.seg_en)

    #load unsegmented data
    nonseg_rf = load_strings(args.unseg_rf)
    nonseg_en = load_strings(args.unseg_en)

    nonseg_pc = zip(nonseg_en, nonseg_rf)

    r.shuffle(nonseg_pc) #shuffling order of data

    nonseg_en, nonseg_rf = zip(*nonseg_pc)

    #number of segmented train data

    seg_pc = zip(seg_en, seg_rf, seg_actions)

    print "{} segmented commands".format(len(seg_pc))
    r.shuffle(seg_pc)

    num_test = int(args.npi_test_frac*len(seg_pc))

    seg_test = seg_pc[:num_test] #test data for both

    test_en, test_rf, test_actions = zip(*seg_test)

    #saving test data
    save_strings(args.out + "L0_test.en", test_en)
    save_strings(args.out + "L0_test_rf.ml", test_rf)
    save_strings(args.out + "L0_test_actions.ml", test_actions)

    seg_train = seg_pc[num_test:] #train data for NPI + part of train data for RNN

    npi_train_en, seg_train_rf, npi_train_actions = zip(*seg_train)

    #saving NPI training data
    save_strings(args.out + "L0_npi_train.en", npi_train_en)
    save_strings(args.out + "L0_npi_train_actions.ml", npi_train_actions)

    #merge the NPI training data and the train reward functions into the rest of the corpus

    rnn_train_en = list(nonseg_en) + list(npi_train_en)
    rnn_train_rf = list(nonseg_rf) + list(seg_train_rf)

    save_strings(args.out + "L0_rnn_train.en", rnn_train_en)
    save_strings(args.out + "L0_rnn_train.ml", rnn_train_rf)


if __name__=="__main__":
    run(parse(argv[1:]))
