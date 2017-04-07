#reannotates a "means" data set with action sequences rather than reward functions

from argparse import ArgumentParser
from sys import argv

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--means", help="location of means machine language data")
    parser.add_argument("--traces", help="location of action traces")
    parser.add_argument("--out", help="outfile location")
    return parser.parse_args(args)

#loads traces from the traces_amdp file, match their ML commands to their traces via a hashmap
def load_traces(filename):
    traces_dict = {}
    start_dict = {}
    end_dict = {}
    with open(filename, 'r') as tracefile:
        blocks = []
        all_text = "".join(line for line in tracefile)
        blocks = all_text.split("\n\n")

        for block in blocks:
            if not block.startswith("#"):
                block_split = block.split("\n")
                traces_dict[block_split[0]] = block_split[3] #mapping AMDP RF to trace
                start_dict[block_split[0]] = block_split[1]
                end_dict[block_split[0]] = block_split[2]

    return traces_dict, start_dict, end_dict

def run(args):
    #load data
    with open(args.means, 'r') as mlfile:
        ml = [line.strip() for line in mlfile]

    #load action traces corresponding to each RF in the RSS domain
    traces_dict, start_dict, end_dict = load_traces(args.traces)

    ml_actions = []

    for rf in ml:
        if rf in traces_dict.keys(): #can't handle manipulation tasks
            outstr = "{} {} {}".format(start_dict[rf], end_dict[rf], traces_dict[rf])
        else:
            outstr = "NONE" #in case the task is a manipulation task

        ml_actions.append(outstr)

    with open(args.out, 'w') as out:
        out.write("\n".join(ml_actions))

if __name__=="__main__":
    run(parse(argv[1:]))
