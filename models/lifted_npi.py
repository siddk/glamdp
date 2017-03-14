"""
lifted_npi.py

Core model definition file for the NPI for Lifted RF Grounding.
"""
import numpy as np
import tensorflow as tf
import tflearn

TERMINATE, CONTINUE = 1, 0
P_IDX, A1_IDX, T_IDX = 0, 1, 2

class NPI():
    def __init__(self, train_path, test_path, embedding_size=30, num_args=1, npi_core_dim=64,
                 key_dim=32, batch_size=16, num_epochs=5, initializer=tf.random_normal_initializer(stddev=0.1)):
        """
        Instantiate an NPI for grounding language to lifted Reward Functions, with the necessary
        parameters.

        :param train_path: Path to training data
        :param test_path: Path to test data
        """
        self.train_path, self.test_path = train_path, test_path
        self.embed_sz, self.num_args, self.init = embedding_size, num_args, initializer
        self.npi_core_dim, self.key_dim, self.bsz = npi_core_dim, key_dim, batch_size
        self.epochs = num_epochs
        self.session = tf.Session()

        # Build Vectorized Sentences
        self.word2id, self.trainX, self.testX, self.trainX_len, self.testX_len = self.parse_sentences()

        # Build up Program Set 
        self.progs, self.args, self.trainY, self.testY = self.parse_programs()

        # Add GO Program
        self.progs["<<GO>>"] = len(self.progs)
        
        # Setup Placeholders
        self.X = tf.placeholder(tf.int32, shape=[None, self.trainX.shape[1]], name='NL_Directive')
        self.X_len = tf.placeholder(tf.int32, shape=[None], name="NL_Length")
        self.P = tf.placeholder(tf.int32, shape=[None], name='Program_ID')
        self.P_out = tf.placeholder(tf.int64, shape=[None], name='Program_Out')
        self.A1_out = tf.placeholder(tf.int64, shape=[None], name='Argument1_Out')
        self.T_out = tf.placeholder(tf.int64, shape=[None], name='Termination_Out')
        self.keep_prob = tf.placeholder(tf.float32, name="Dropout_Prob")

        # Instantiate Network Weights
        self.instantiate_weights()
        
        # Generate Input Representation
        self.s = self.encode_input()

        # Feed through NPI Core, get Hidden State
        self.h = self.npi_core()

        # Build Termination Network => Returns Probability of Terminating
        self.terminate = self.terminate_net()

        # Build Program Network => Generates probability distribution over programs
        self.program_distribution = self.program_net()

        # Build Argument Networks => Generates list of argument distributions
        self.arguments = self.argument_net()

        # Build Losses
        self.t_loss, self.p_loss, self.a_losses = self.build_losses()
        self.loss = 1 * sum([self.t_loss, self.p_loss]) + sum(self.a_losses)

        # Build Training Operation
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        # Build Accuracy Operation
        correct_prog = tf.equal(tf.argmax(self.program_distribution, 1), self.P_out)
        self.p_accuracy = tf.reduce_mean(tf.cast(correct_prog, tf.float32), name="Prog_Accuracy")
        correct_a1 = tf.equal(tf.argmax(self.arguments[0], 1), self.A1_out)
        self.a1_accuracy = tf.reduce_mean(tf.cast(correct_a1, tf.float32), name="A1_Accuracy")

        # Initialize all Variables
        self.session.run(tf.global_variables_initializer())

    def instantiate_weights(self):
        """
        Instantiate all network weights, including NPI Core GRU Cell.
        """
        # Create NL Embedding Matrix, with 0 Vector for PAD_ID (0)
        E = tf.get_variable("Embedding", [len(self.word2id), self.embed_sz], initializer=self.init)
        zero_mask = tf.constant([0 if i == 0 else 1 for i in range(len(self.word2id))], 
                                    dtype=tf.float32, shape=[len(self.word2id), 1])
        self.E = E * zero_mask

        # Create Learnable Mask
        # self.inp_mask = tf.get_variable("Inp_Mask", [self.trainX.shape[1], 1], initializer=tf.constant_initializer(1.0))

        # Create Program Embedding Matrix
        self.PE = tf.get_variable("Program_Embedding", [len(self.progs), self.embed_sz], initializer=self.init)

        # Create GRU NPI Core
        self.gru = tf.contrib.rnn.GRUCell(self.npi_core_dim)

    def encode_input(self):
        """
        Map Natural Language Directive to Fixed Size Vector Embedding.
        """
        directive_embedding = tf.nn.embedding_lookup(self.E, self.X)          # [None, sent_len, embed_sz]
        directive_embedding = tf.nn.dropout(directive_embedding, self.keep_prob)

        with tf.variable_scope("Encoder"):
            self.encoder_gru = tf.contrib.rnn.GRUCell(self.embed_sz)
            _, state = tf.nn.dynamic_rnn(self.encoder_gru, directive_embedding, sequence_length=self.X_len, dtype=tf.float32)
        return state

    def npi_core(self):
        """
        Concatenate input encoding and program embedding, and feed through NPI Core.
        """
        # Embed the Program
        program_embedding = tf.nn.embedding_lookup(self.PE, self.P)           # [None, embed_sz]

        # Concatenate state and program embedding
        p_embedding = tf.expand_dims(program_embedding, axis=1)               # [None, 1, embed_sz]
        s_embedding = tf.expand_dims(self.s, axis=1)                          # [None, 1, embed_sz]
        state = tf.concat([s_embedding, p_embedding], 2)                      # [None, 1, 2 * embed_sz]
        state = tf.nn.dropout(state, self.keep_prob)

        # Feed through NPI Core
        with tf.variable_scope("Core"):
            self.gru = tf.contrib.rnn.GRUCell(self.npi_core_dim)
            _, state = tf.nn.dynamic_rnn(self.gru, state, dtype=tf.float32)
        h_state = state                                                       # Shape [None, npi_core_sz]
        return h_state
    
    def terminate_net(self):
        """
        Build the NPI Termination Network, that takes in the NPI Core Hidden State, and returns
        the probability of terminating program.
        """
        p_terminate = tflearn.fully_connected(self.h, 2, activation='linear', regularizer='L2')
        return p_terminate                                                   # Shape: [bsz, 2]

    def program_net(self):
        """
        Build the NPI Key Network, that takes in the NPI Core Hidden State, and returns a softmax
        distribution over possible next programs.
        """
        # Compute Distribution over Programs
        hidden = tflearn.fully_connected(self.h, self.key_dim, activation='elu', regularizer='L2')
        hidden = tf.nn.dropout(hidden, self.keep_prob)
        prog_dist = tflearn.fully_connected(hidden, len(self.progs))         # Shape: [bsz, num_progs]
        return prog_dist

    def argument_net(self):
        """
        Build the NPI Argument Networks (a separate net for each argument), each of which takes in
        the NPI Core Hidden State, and returns a softmax over the argument dimension.
        """
        args = []
        for i in range(self.num_args):
            arg_hidden = tflearn.fully_connected(self.h, self.key_dim, activation='elu', regularizer='L2')
            arg_hidden = tf.nn.dropout(arg_hidden, self.keep_prob)
            arg = tflearn.fully_connected(arg_hidden, len(self.args), activation='linear', 
                                          name='Argument_{}'.format(str(i)))
            args.append(arg)
        return args                                                         # Shape: [bsz, num_args]
    
    def build_losses(self):
        """
        Build separate loss computations, using the logits from each of the sub-networks.
        """
        # Termination Network Loss
        termination_loss = tf.losses.sparse_softmax_cross_entropy(self.T_out, self.terminate)
        
        # Program Network Loss 
        program_loss = tf.losses.sparse_softmax_cross_entropy(self.P_out, self.program_distribution)

        # Argument Network Losses
        arg_losses = []
        for i in range(self.num_args):
            if i == 0:
                arg_losses.append(tf.losses.sparse_softmax_cross_entropy(self.A1_out, self.arguments[i]))
        
        return termination_loss, program_loss, arg_losses

    def fit(self):
        """
        Train the model, with the specified batch size and number of epochs.
        """
        # Run through epochs
        chunk_size = (len(self.trainX) / self.bsz) * self.bsz
        for e in range(self.epochs):
            curr_loss, curr_p_acc, curr_a1_acc, curr_a2_acc, batches = 0.0, 0.0, 0.0, 0.0, 0.0
            for start, end in zip(range(0, len(self.trainX[:chunk_size]) - self.bsz, self.bsz),
                                  range(self.bsz, len(self.trainX[:chunk_size]), self.bsz)):
                loss, p_acc, a1_acc, _ = self.session.run([self.loss, self.p_accuracy, self.a1_accuracy,
                                                           self.train_op], feed_dict={
                                                                       self.X: self.trainX[start:end],
                                                                       self.X_len: self.trainX_len[start:end],
                                                                       self.P: [self.progs["<<GO>>"]] * self.bsz,
                                                                       self.P_out: self.trainY[start:end, P_IDX],
                                                                       self.A1_out: self.trainY[start:end, A1_IDX],
                                                                       self.T_out: self.trainY[start:end, T_IDX],
                                                                       self.keep_prob: 0.5})
                curr_loss, batches = curr_loss + loss, batches + 1
                curr_p_acc, curr_a1_acc = curr_p_acc + p_acc, curr_a1_acc + a1_acc
            print 'Epoch %d\tAverage Loss: %.3f\tProgram Accuracy: %.3f\tArg1 Accuracy: %.3f' % (e, curr_loss / batches, curr_p_acc / batches, curr_a1_acc / batches)
    
    def eval(self):
        """
        Evaluate the model on the test data.
        """
        num_correct, total = 0.0, 0.0
        for i in range(len(self.testX)):
            pred_prog, pred_a1 = self.score(self.testX[i], self.testX_len[i])
            true_prog, true_a1 = self.testY[i, P_IDX], self.testY[i, A1_IDX]
            if (pred_prog == int(true_prog)) and (pred_a1 == int(true_a1)):
                num_correct += 1
                
        print "Test Accuracy: %.3f" % (float(num_correct) / float(len(self.testX)))
        
    def score(self, nl_command, length):
        """
        Given a natural language command, return predicted output and score.

        :return: List of tokens representing predicted command, and score.
        """
        prog, a1 = self.session.run([self.program_distribution, self.arguments[0]], 
                                    feed_dict={self.X: [nl_command], self.X_len: [length], self.P: [self.progs["<<GO>>"]], self.keep_prob: 1.0})
        pred_prog, pred_a1 = np.argmax(prog, axis=1), np.argmax(a1, axis=1)
        return pred_prog[0], pred_a1[0]

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
        program_set, arg_set, train_traces, test_traces = {}, {"NULL": 0}, [], []
        for data_type in range(2):
            for program in ([train_programs, test_programs][data_type]):
                lvl, trace = program[0], []
                if len(program) == 3:
                    prog_key, arg_key = lvl + "_" + program[1], program[2]
                    if prog_key not in program_set:
                        program_set[prog_key] = len(program_set)
                    if arg_key not in arg_set:
                        arg_set[arg_key] = len(arg_set)
                    trace.append((program_set[prog_key], arg_set[arg_key], TERMINATE))
                else:
                    assert(len(program) == 5)
                    prog_key = lvl + "_" + program[1] + "_" + program[3]
                    arg_key = program[2] + "_" + program[4]
                    if prog_key not in program_set:
                        program_set[prog_key] = len(program_set)
                    if arg_key not in arg_set:
                        arg_set[arg_key] = len(arg_set)
                    
                    trace.append((program_set[prog_key], arg_set[arg_key], TERMINATE))
                if data_type == 0:
                    train_traces.append(trace)
                else:
                    test_traces.append(trace)
        
        # Vectorize Traces
        vtrain_traces, vtest_traces = np.zeros([len(train_traces), 3]), np.zeros([len(test_traces), 3])
        for i in range(len(train_traces)):
            trace = train_traces[i][0]
            vtrain_traces[i][P_IDX] = trace[0]
            vtrain_traces[i][A1_IDX] = trace[1]
            vtrain_traces[i][T_IDX] = trace[2]
        for i in range(len(test_traces)):
            trace = test_traces[i][0]
            vtest_traces[i][P_IDX] = trace[0]
            vtest_traces[i][A1_IDX] = trace[1]
            vtest_traces[i][T_IDX] = trace[2]

        # Return Program Set, Argument Set, Execution Traces
        return program_set, arg_set, vtrain_traces, vtest_traces

    