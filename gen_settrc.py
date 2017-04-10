'''
generates a .ccgsettrc file from the provided natural language and machine language annotations
'''

from argparse import ArgumentParser
from sys import argv
from predict_class import load_data

default_start = "(6,6,0)" #for lack of parse failure
default_map = "cleanupclassic"

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--parallel_data", help="path with .en and .ml extensions, providing natural language and machine language annotations")
    parser.add_argument("--test", help="if data is test, i.e. a ccgsettrc", action="store_true")
    parser.add_argument("--out", help="file path to .output file")
    return parser.parse_args(args)

def gen_trace(nl, ml, trace_id, map_name=default_map, is_ccgsettrc=True):
    #parsing machine language
    #detecting invalid action trace (i.e. block manipulation task or incorrectly classified task)
    #TODO: assume that all action traces are present, collect files
    if ml == "NONE" or ml.startswith("L1") or ml.startswith("L2"):
        start = default_start
        end = default_start
        actions = default_start #assuming automatic failures, probably invalid
    else:
        split_ml = ml.split()
        start = split_ml[0]
        end = split_ml[1]
        actions = split_ml[2]

    #add header information
    #TODO: check this info
    header = "CleanupTrace_{}\nmap={} end={} start={} valid=True correct=True efficiency=(0.8,0.4) implicit=False numFollowers=5 confidence=(5.0,2.0) directionRating=(4.6,1.8547236990991407)	annotated=True	targetFound=(0.8,0.4)".format(trace_id, map_name, end, start)
    if is_ccgsettrc:
        trace = "\n".join([header, nl, "you:ps", actions])
    else:
        trace = "\n".join([header, nl, actions])

    return trace

def save_traces(traces, outfile):
    with open(outfile, 'w') as f:
        outstr = "\n\n".join(traces)
        f.write(outstr)
        print "traces saved to {}".format(outfile)

def run(args):
    pred_means_en = load_data(args.parallel_data + ".en")
    pred_means_ml = load_data(args.parallel_data + ".ml")

    id_header = "test" if args.test else "train"

    traces = []
    for i, pc in enumerate(zip(pred_means_en, pred_means_ml)):
        nl, ml = pc
        traces.append(gen_trace(nl, ml, "{}_{}".format(id_header, i), is_ccgsettrc=args.test))

    save_traces(traces, args.out)




if __name__=="__main__":
    run(parse(argv[1:]))
