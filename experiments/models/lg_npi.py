"""
lg_npi.py

Core model definition file for the LG-NPI for Lifted RF Grounding.
"""
import numpy as np
import tensorflow as tf
import tflearn

TERMINATE, CONTINUE = 1, 0
P_IDX, A1_IDX, T_IDX = 0, 1, 2

class NPI():
    def __init__(self, means_train_path, ends_train_path, means_test_path, ends_test_path,
                 embedding_size=30, num_args=1, npi_core_dim=64, key_dim=32, batch_size=16, 
                 num_epochs=5, initializer=tf.random_normal_initializer(stddev=0.1), restore=False):
        """
        Instantiate an NPI for grounding language to lifted Reward Functions, with the necessary
        parameters.

        :param means_train_path: Path to means training data
        :param ends_train_path: Path to ends training data
        :param means_test_path: Path to means test data
        :param ends_test_path: Path to ends test data
        """
        self.means_train_path, self.ends_train_path = means_train_path, ends_train_path
        self.means_test_path, self.ends_test_path = means_test_path, ends_test_path
        self.embed_sz, self.num_args, self.init = embedding_size, num_args, initializer
        self.npi_core_dim, self.key_dim, self.bsz = npi_core_dim, key_dim, batch_size
        self.epochs = num_epochs
        self.session = tf.Session()

        # Parse Inputs
        self.word2id, self.progs, self.args, self.trainX, self.trainX_len, self.testMeansX, self.testMeans_len, self.testEndsX, self.testEnds_len, self.trainY, self.testMeansY, self.testEndsY = self.parse()

        # Add GO Program
        self.progs["<<GO>>"] = len(self.progs)
        
        # Create id2prog Map
        self.id2prog = {i:prog for prog, i in self.progs.iteritems()}

        # Create id2arg Map
        self.id2arg = {i:arg for arg, i in self.args.iteritems()}

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
        # self.loss = 1 * sum([self.t_loss, self.p_loss]) + sum(self.a_losses)
        self.loss = self.p_loss + sum(self.a_losses)

        # Build Training Operation
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        # Build Accuracy Operation
        correct_prog = tf.equal(tf.argmax(self.program_distribution, 1), self.P_out)
        self.p_accuracy = tf.reduce_mean(tf.cast(correct_prog, tf.float32), name="Prog_Accuracy")
        correct_a1 = tf.equal(tf.argmax(self.arguments[0], 1), self.A1_out)
        self.a1_accuracy = tf.reduce_mean(tf.cast(correct_a1, tf.float32), name="A1_Accuracy")

        # Create Saver
        self.saver = tf.train.Saver()

        if restore:
            self.saver.restore(self.session, restore)
        else:
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

        # Create Program Embedding Matrix
        # self.PE = tf.get_variable("Program_Embedding", [len(self.progs), self.embed_sz], initializer=self.init)

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
        # program_embedding = tf.nn.embedding_lookup(self.PE, self.P)           # [None, embed_sz]

        # Concatenate state and program embedding
        # p_embedding = tf.expand_dims(program_embedding, axis=1)               # [None, 1, embed_sz]
        s_embedding = tf.expand_dims(self.s, axis=1)                          # [None, 1, embed_sz]
        # state = tf.concat([s_embedding, p_embedding], 2)                      # [None, 1, 2 * embed_sz]
        state = tf.nn.dropout(s_embedding, self.keep_prob)

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
        return args                                                          # Shape: [bsz, num_args]

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

    def eval_means(self):
        """
        Evaluate the model on the test data.
        """
        num_correct, total = 0.0, 0.0
        for i in range(len(self.testMeansX)):
            pred_prog, pred_a1 = self.score(self.testMeansX[i], self.testMeans_len[i])
            true_prog, true_a1 = self.testMeansY[i, P_IDX], self.testMeansY[i, A1_IDX]
            if (pred_prog == int(true_prog)) and (pred_a1 == int(true_a1)):
                num_correct += 1

        print "Means Per-Segment Test Accuracy: %.3f" % (float(num_correct) / float(len(self.testMeansX)))
    
    def eval_means_all(self):
        """
        Evaluate the model on ALL the means data (not per-segment, but per-sentence).
        """
        num_correct, total = 0.0, 0.0
        with open(self.means_test_path + ".en", 'r') as f:
            means_sentences, lens = [x.strip().split('|') for x in f.readlines()], []
            for i in means_sentences:
                lens.append(len(i))
        
        counter = 0
        for i in range(len(means_sentences)):
            correct = 1
            for j in range(lens[i]):
                pred_prog, pred_a1 = self.score(self.testMeansX[counter], self.testMeans_len[counter])
                true_prog, true_a1 = self.testMeansY[counter, P_IDX], self.testMeansY[counter, A1_IDX]
                if (pred_prog == int(true_prog)) and (pred_a1 == int(true_a1)):
                    correct *= 1
                else:
                    correct *= 0
                counter += 1
            num_correct += correct
            total += 1
        assert(counter == len(self.testMeansX))
        print "Means Full-Sentence Test Accuracy: %.3f" % (float(num_correct) / float(total))

    def eval_ends(self):
        """
        Evaluate the model on the test data.
        """
        num_correct, total = 0.0, 0.0
        for i in range(len(self.testEndsX)):
            pred_prog, pred_a1 = self.score(self.testEndsX[i], self.testEnds_len[i])
            true_prog, true_a1 = self.testEndsY[i, P_IDX], self.testEndsY[i, A1_IDX]
            if (pred_prog == int(true_prog)) and (pred_a1 == int(true_a1)):
                num_correct += 1

        print "Ends Test Accuracy: %.3f" % (float(num_correct) / float(len(self.testEndsX)))

    def eval_permuted_ends(self, permuted_ends_path):
        """
        Evaluate the model on the permuted test data.
        """
        with open(permuted_ends_path + ".en", 'r') as f:
            ends_sentences = [x.split() for x in f.readlines()]
            
        with open(permuted_ends_path + "_lifted_gt.ml", 'r') as f:
            ends_programs = [x.split() for x in f.readlines()]
            for i in range(len(ends_programs)):
                if len(ends_programs[i]) == 4:
                    program = ends_programs[i]
                    ends_programs[i] = [program[0] + "_" + program[2], program[1] + "_" + program[3]]
        
        permuted_ends = zip(ends_sentences, ends_programs)

        # Build Language Representations
        permutedEndsX, permutedEnds_len = np.zeros((len(permuted_ends), self.trainX.shape[1])), np.zeros((len(permuted_ends)), dtype=np.int32)
        for i in range(len(permuted_ends)):
            nl_sentence = permuted_ends[i][0]
            permutedEnds_len[i] = min(self.trainX.shape[1], len(nl_sentence))
            for j in range(permutedEnds_len[i]):
                permutedEndsX[i][j] = self.word2id[nl_sentence[j]]

        # Build Program Representations
        permuted_ends_traces = []
        for _, program in permuted_ends:
            prog_key, arg = program
            permuted_ends_traces.append((self.progs[prog_key], self.args[arg], TERMINATE))
        assert(len(permuted_ends_traces) == len(permutedEndsX))

        # Vectorize Traces
        vpermuted_traces = np.zeros([len(permuted_ends_traces), 3])
        for i in range(len(permuted_ends_traces)):
            trace = permuted_ends_traces[i]
            vpermuted_traces[i][P_IDX] = trace[0]
            vpermuted_traces[i][A1_IDX] = trace[1]
            vpermuted_traces[i][T_IDX] = trace[2]

        num_correct, total = 0.0, 0.0
        for i in range(len(permutedEndsX)):
            pred_prog, pred_a1 = self.score(permutedEndsX[i], permutedEnds_len[i])
            true_prog, true_a1 = vpermuted_traces[i, P_IDX], vpermuted_traces[i, A1_IDX]
            if (pred_prog == int(true_prog)) and (pred_a1 == int(true_a1)):
                num_correct += 1

        print "Permuted Ends Test Accuracy: %.3f" % (float(num_correct) / float(len(permutedEndsX)))

    def score(self, nl_command, length):
        """
        Given a natural language command, return predicted output and score.

        :return: List of tokens representing predicted command, and score.
        """
        prog, a1 = self.session.run([self.program_distribution, self.arguments[0]],
                                    feed_dict={self.X: [nl_command], self.X_len: [length], self.P: [self.progs["<<GO>>"]], self.keep_prob: 1.0})
        pred_prog, pred_a1 = np.argmax(prog, axis=1), np.argmax(a1, axis=1)
        return pred_prog[0], pred_a1[0]

    def score_nl(self, nl_command):
        """
        Given a natural language string, return a string representing the lifted RF
        """
        vec, length = self.vectorize_sentence(nl_command)
        pred_prog, pred_a1 = self.score(vec, length)

        # Produce string representation of RF
        prog_split = self.id2prog[pred_prog].split("_")
        args = self.id2arg[pred_a1].split("_")

        # Handle AgentInRegion_BlockInRegion
        if len(prog_split) == 2:
            if len(args) != 2:
                args.append("NONE")
            return prog_split[0] + " | " + args[0] + " | " + prog_split[1] + " | " + args[1]
        elif len(prog_split) == 1:
            if prog_split[0] in ["Up", "Down", "Left", "Right"]:
                if args[0].isdigit():
                    return " | ".join([prog_split[0] for _ in range(int(args[0]))])    
            return prog_split[0] + " | " + args[0]
    
    def vectorize_sentence(self, nl_sentence):
        """
        Vectorizes a single sentence.
        """
        sent = nl_sentence.split()
        sentence_len = len(sent)

        vec = np.zeros((self.max_len,))

        # Truncate Sentences that are too long
        for i in range(min(sentence_len, self.max_len)):
            vec[i] = self.word2id.get(sent[i], self.word2id['UNK'])

        return vec, sentence_len

    def parse(self, max_sentence_len=50):
        """
        Parse the english sentences in the training and test data, generating vocabularies,
        as well as vectorized forms of each sentence.
        """
        # Parse Training Data
        counter = 0
        with open(self.means_train_path + ".en", 'r') as f:
            means_sentences, means_segments = [x.strip().split('|') for x in f.readlines()], []
            for i in means_sentences:
                counter += 1
                for j in i:
                    means_segments.append(j.split())
        
        with open(self.means_train_path + "_actions.ml", 'r') as f:
            means_sentences, means_programs = [x.strip().split('|') for x in f.readlines()], []
            for i in means_sentences:
                for j in i:
                    means_programs.append(j.split())
        
        assert(len(means_segments) == len(means_programs))
        
        with open(self.ends_train_path + ".en", 'r') as f:
            ends_sentences = [x.split() for x in f.readlines()]
            
        with open(self.ends_train_path + "_npi_lifted.ml", 'r') as f:
            ends_programs = [x.split() for x in f.readlines()]
            for i in range(len(ends_programs)):
                if len(ends_programs[i]) == 4:
                    program = ends_programs[i]
                    ends_programs[i] = [program[0] + "_" + program[2], program[1] + "_" + program[3]]
        
        assert(len(ends_sentences) == len(ends_programs))

        # Generate and Shuffle Train Set
        self.train_segments = means_segments + ends_sentences
        self.train_programs = means_programs + ends_programs
        self.train_set = zip(self.train_segments, self.train_programs)
        
        import random
        random.seed(21)
        random.shuffle(self.train_set)
        random.shuffle(self.train_set)
        random.shuffle(self.train_set)
        
        # Parse Test Data
        with open(self.means_test_path + ".en", 'r') as f:
            means_sentences, means_segments = [x.strip().split('|') for x in f.readlines()], []
            lens = []
            for i in means_sentences:
                lens.append(len(i))
                for j in i:
                    means_segments.append(j.split())
        
        with open(self.means_test_path + "_actions.ml", 'r') as f:
            means_sentences, means_programs = [x.strip().split('|') for x in f.readlines()], []
            plens = []
            for i in means_sentences:
                plens.append(len(i))
                for j in i:
                    means_programs.append(j.split())
        
        assert(len(means_segments) == len(means_programs))
        
        with open(self.ends_test_path + ".en", 'r') as f:
            ends_sentences = [x.split() for x in f.readlines()]
            
        with open(self.ends_test_path + "_npi_lifted.ml", 'r') as f:
            ends_programs = [x.split() for x in f.readlines()]
            for i in range(len(ends_programs)):
                if len(ends_programs[i]) == 4:
                    program = ends_programs[i]
                    ends_programs[i] = [program[0] + "_" + program[2], program[1] + "_" + program[3]]
        
        assert(len(ends_sentences) == len(ends_programs))

        # Generate Test Sets
        self.test_means = zip(means_segments, means_programs)
        self.test_ends = zip(ends_sentences, ends_programs)
        self.test_set = self.test_means + self.test_ends

        # Create Vocabulary, Get Maximum Sentence Length
        word2id, self.max_len = {"PAD": 0, "UNK": 1}, 0
        for sent, _ in self.train_set + self.test_set:
            if len(sent) > self.max_len:
                self.max_len = len(sent)
            for word in sent:
                if word not in word2id:
                    word2id[word] = len(word2id)
        
        if self.max_len > max_sentence_len:
            self.max_len = max_sentence_len
        
        # Vectorize English Data
        trainX, trainX_len = np.zeros((len(self.train_set), self.max_len)), np.zeros((len(self.train_set)), dtype=np.int32)
        testMeansX, testEndsX = np.zeros((len(self.test_means), self.max_len)), np.zeros((len(self.test_ends), self.max_len))
        testMeans_len, testEnds_len = np.zeros((len(self.test_means)), dtype=np.int32), np.zeros((len(self.test_ends)), dtype=np.int32)

        for i in range(len(self.train_set)):
            nl_sentence = self.train_set[i][0]
            trainX_len[i] = min(self.max_len, len(nl_sentence))
            for j in range(trainX_len[i]):
                trainX[i][j] = word2id[nl_sentence[j]]
        
        for i in range(len(self.test_means)):
            nl_sentence = self.test_means[i][0]
            testMeans_len[i] = min(self.max_len, len(nl_sentence))
            for j in range(testMeans_len[i]):
                testMeansX[i][j] = word2id[nl_sentence[j]]
        
        for i in range(len(self.test_ends)):
            nl_sentence = self.test_ends[i][0]
            testEnds_len[i] = min(self.max_len, len(nl_sentence))
            for j in range(testEnds_len[i]):
                testEndsX[i][j] = word2id[nl_sentence[j]]
        
        # Parse Program Data
        program_set, arg_set, train_traces, test_means_traces, test_ends_traces = {}, {}, [], [], []
        for i, program in self.train_set + self.test_set:
            assert(len(program) == 2)
            p, arg = program
            if p not in program_set:
                program_set[p] = len(program_set)
            if arg not in arg_set:
                arg_set[arg] = len(arg_set)
        
        # Create Traces
        for _, program in self.train_set:
            prog_key, arg = program
            train_traces.append((program_set[prog_key], arg_set[arg], TERMINATE))
        assert(len(train_traces) == len(trainX))
        
        for _, program in self.test_means:
            prog_key, arg = program
            test_means_traces.append((program_set[prog_key], arg_set[arg], TERMINATE))
        assert(len(test_means_traces) == len(testMeansX))

        for _, program in self.test_ends:
            prog_key, arg = program
            test_ends_traces.append((program_set[prog_key], arg_set[arg], TERMINATE))
        assert(len(test_ends_traces) == len(testEndsX))

        # Vectorize Traces
        vtrain_traces = np.zeros([len(train_traces), 3])
        vtest_means_traces, vtest_ends_traces = np.zeros([len(test_means_traces), 3]), np.zeros([len(test_ends_traces), 3])
        for i in range(len(train_traces)):
            trace = train_traces[i]
            vtrain_traces[i][P_IDX] = trace[0]
            vtrain_traces[i][A1_IDX] = trace[1]
            vtrain_traces[i][T_IDX] = trace[2]
        
        for i in range(len(test_means_traces)):
            trace = test_means_traces[i]
            vtest_means_traces[i][P_IDX] = trace[0]
            vtest_means_traces[i][A1_IDX] = trace[1]
            vtest_means_traces[i][T_IDX] = trace[2]

        for i in range(len(test_ends_traces)):
            trace = test_ends_traces[i]
            vtest_ends_traces[i][P_IDX] = trace[0]
            vtest_ends_traces[i][A1_IDX] = trace[1]
            vtest_ends_traces[i][T_IDX] = trace[2]

        return word2id, program_set, arg_set, trainX, trainX_len, testMeansX, testMeans_len, testEndsX, testEnds_len, vtrain_traces, vtest_means_traces, vtest_ends_traces