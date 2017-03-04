"""
data_utils.py

Added utilities for handling data.

"""

from argparse import ArgumentParser
from sys import argv

class Error(Exception):
    pass

class AlignmentError(Error):
    def __init__(self, message):
        self.message = message

#parses arguments for testing.
def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--natural_language", help="file path to natural language list")
    parser.add_argument("--machine_language", help="file path to machine language list")
    return parser.parse_args(args)

#loads list of list of tokens from filepath.
def load_strings(filepath):
    with open(filepath, 'r') as textfile:
        return [line.strip().split() for line in textfile]

#loads parallel corpus.
#throws exception if not aligned.
def load_pc(natural_lang, machine_lang):
    ml, nl =  load_strings(natural_lang), load_strings(machine_lang)

    if (len(ml) != len(nl)):
        raise AlignmentError("corpuses of different lengths!")
    return ml, nl

def run(args):
    print "testing parallel corpus loading"
    nl, ml = load_pc(args.natural_language, args.machine_language)

    #should be the same, but just for diagnostics
    print "loaded from {} and {}".format(args.natural_language, args.machine_language)
    print "loaded {} NL expressions and {} ML expressions".format(len(nl), len(ml))

if __name__=="__main__":
    run(parse(argv[1:]))
