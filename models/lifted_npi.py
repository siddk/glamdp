"""
lifted_npi.py

Core model definition file for the NPI for Lifted RF Grounding.
"""
import numpy as np
import tensorflow as tf

TERMINATE, CONTINUE = 1, 0

class NPI():
    def __init__(self, train_path, test_path, embedding_size=30, num_args=1, 
                 initializer=tf.random_normal_initializer(stddev=0.1)):
        """
        Instantiate an NPI for grounding language to lifted Reward Functions, with the necessary
        parameters.

        :param train_path: Path to training data
        :param test_path: Path to test data
        """
        self.train_path, self.test_path = train_path, test_path
        self.embed_sz, self.num_args, self.init = embedding_size, num_args, initializer

        # Build Vectorized Sentences
        self.word2id, self.trainX, self.testX, self.trainX_len, self.testX_len = self.parse_sentences()

        # Build up Program Set 
        self.progs, self.args, self.trainY, self.testY = self.parse_programs()

        # Add GO Program
        self.progs["<<GO>>"] = len(self.progs)
        
        # Setup Placeholders
        self.X = tf.placeholder(tf.int32, shape=[None, self.trainX.shape[1]], name='NL_Directive')
        self.P = tf.placeholder(tf.int32, shape=[None, 1], name='Program_ID')
        self.P_out = tf.placeholder(tf.int32, shape=[None], name='Program_Out')
        self.A_out = tf.placeholder(tf.int32, shape=[None], name='Argument_Out')
        self.T_out = tf.placeholder(tf.int32, shape=[None], name='Termination_Out')

        # Instantiate Network Weights
        self.instantiate_weights()
        
        # Generate Input Representation
        self.s = self.encode_input()

        # Feed through NPI Core, get Hidden State
        self.h = self.npi_core()
    
    def instantiate_weights(self):
        """
        Instantiate all network weights, including NPI Core GRU Cell.
        """
        # Create NL Embedding Matrix, with 0 Vector for PAD_ID (0)
        E = tf.get_variable("Embedding", [len(self.word2id), self.embed_sz], initializer=self.init)
        zero_mask = tf.constant([0 if i == 0 else 1 for i in range(len(self.word2id))], 
                                    dtype=tf.float32, shape=[len(self.word2id, 1])
        self.E = E * zero_mask

        # Create Learnable Mask
        self.inp_mask = tf.get_variable("Inp_Mask", [self.trainX.shape[1], 1], initializer=tf.constant_initializer(1.0))

        # Create Program Embedding Matrix
        self.PE = tf.get_variable("Program_Embedding", [len(self.progs), self.embed_sz], initializer=self.init)

    def encode_input(self):
        """
        Map Natural Language Directive to Fixed Size Vector Embedding.
        """
        directive_embedding = tf.nn.embedding_lookup(self.E, self.X)          # [None, sent_len, embed_sz]
        directive_embedding = tf.multiply(directive_embedding, self.inp_mask) # [None, sent_len, embed_sz]
        directive_embedding = tf.reduce_sum(directive_embedding, axis=[1])    # [None, embed_sz]
        return directive_embedding

    def npi_core(self):
        """
        Concatenate input encoding and program embedding, and feed through NPI Core.
        """
        pass

    def parse_sentences(self, max_sentence_len=50):
        """
        Parse the english sentences in the training and test data, generating vocabularies,
        as well as vectorized forms of each sentence.
        """
        with open(self.train_path + ".en", 'r') as f:
            train_sentences = [x.split() for x in f.readlines()]
        
        with open(self.test_path + ".en", 'r') as f:
            test_sentences = [x.split() for x in f.readlines()]
        
        # Create Vocabulary + [0 PAD]
        word2id = {w: i for i, w in enumerate(["PAD"] + list(set(reduce(lambda x, y: x + y, 
                                                                        train_sentences + test_sentences))))}

        # Get Maximum Sentence Length
        max_len = max(map(lambda x: len(x), train_sentences + test_sentences))
        if max_len > max_sentence_len:
            max_len = max_sentence_len
        
        # Vectorize Data
        trainX, testX = np.zeros((len(train_sentences), max_len)), np.zeros((len(test_sentences), max_len))
        trainX_len, testX_len = np.zeros((len(trainX)), dtype=np.int32), np.zeros((len(testX)), dtype=np.int32)
        for i in range(len(train_sentences)):
            trainX_len[i] = min(max_len, len(train_sentences[i]))
            for j in range(trainX_len[i]):
                trainX[i][j] = word2id[train_sentences[i][j]]
        for i in range(len(test_sentences)):
            testX_len[i] = min(max_len, len(test_sentences[i]))
            for j in range(testX_len[i]):
                testX[i][j] = word2id[test_sentences[i][j]]
        
        return word2id, trainX, testX, trainX_len, testX_len

    def parse_programs(self):
        """
        Parse the machine language (reward functions) in the training and test data, generating
        program set, as well as set of execution traces.
        """
        with open(self.train_path + ".ml", 'r') as f:
            train_programs = [x.split()[1:-1] for x in f.readlines()]
        
        with open(self.test_path + ".ml", 'r') as f:
            test_programs = [x.split()[1:-1] for x in f.readlines()]

        # Assemble Program Set, Argument Set, Execution Traces
        program_set, arg_set, train_traces, test_traces = {}, {}, [], []
        for data_type in range(2):
            for program in ([train_programs, test_programs][data_type]):
                lvl, trace, counter = program[0], [], 1
                while counter < len(program):
                    prog_key, arg_key = lvl + "_" + program[counter], program[counter + 1]
                    if prog_key not in program_set:
                        program_set[prog_key] = len(program_set)
                    if arg_key not in arg_set:
                        arg_set[arg_key] = len(arg_set)
                    counter += 2
                    if counter == len(program):
                        trace.append((program_set[prog_key], arg_set[arg_key], TERMINATE))
                    else:
                        trace.append((program_set[prog_key], arg_set[arg_key], CONTINUE))
                if data_type == 0:
                    train_traces.append(trace)
                else:
                    test_traces.append(trace)
        
        # Return Program Set, Argument Set, Execution Traces
        return program_set, arg_set, train_traces, test_traces

    