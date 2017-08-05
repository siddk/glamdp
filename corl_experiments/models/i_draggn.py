"""
i_draggn.py

Core model definition file for the Conditionally Independent DRAGGN (I-DRAGGN) Model.
"""
import numpy as np
import pickle
import tensorflow as tf
import tflearn

TERMINATE, CONTINUE = 1, 0
P_IDX, A1_IDX, T_IDX = 0, 1, 2

class IDRAGGN():
    def __init__(self, means_train_path, ends_train_path, means_test_path, ends_test_path,
                 is_pik=False, pik_train_path=None, pik_test_path=None, embedding_size=30, 
                 num_args=1, npi_core_dim=64, key_dim=32, batch_size=16, num_epochs=5, 
                 initializer=tf.random_normal_initializer(stddev=0.1), restore=False, until=400):
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
        self.is_pik, self.pik_train, self.pik_test = is_pik, pik_train_path, pik_test_path
        self.embed_sz, self.num_args, self.init = embedding_size, num_args, initializer
        self.npi_core_dim, self.key_dim, self.bsz = npi_core_dim, key_dim, batch_size
        self.epochs = num_epochs
        self.until = until
        self.session = tf.Session()

        # Set Random Seed (for consistency)
        # tf.set_random_seed(49)

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
        self.prog_s, self.arg_s = self.encode_input()

        # Build Termination Network => Returns Probability of Terminating
        self.terminate = self.terminate_net()

        # Build Program Network => Generates probability distribution over programs
        self.program_distribution = self.program_net()

        # Build Argument Networks => Generates list of argument distributions
        self.arguments = self.argument_net()

        # Build Losses
        self.t_loss, self.p_loss, self.a_losses = self.build_losses()
        self.loss = self.p_loss + sum(self.a_losses)

        # Build Training Operation
        self.p_train_op = tf.train.AdamOptimizer().minimize(self.p_loss)
        self.a_train_op = tf.train.AdamOptimizer().minimize(sum(self.a_losses))

        # Build Accuracy Operation
        correct_prog = tf.equal(tf.argmax(self.program_distribution, 1), self.P_out)
        self.p_accuracy = tf.reduce_mean(tf.cast(correct_prog, tf.float32), name="Prog_Accuracy")
        correct_a1 = tf.equal(tf.argmax(self.arguments[0], 1), self.A1_out)
        self.a1_accuracy = tf.reduce_mean(tf.cast(correct_a1, tf.float32), name="A1_Accuracy")

        # Create Saver
        self.saver = tf.train.Saver()

        if restore:
            self.saver.restore(self.session, restore)
            import ipdb
            ipdb.set_trace()
        else:
            # Initialize all Variables
            self.session.run(tf.global_variables_initializer())

    def instantiate_weights(self):
        """
        Instantiate all network weights, including NPI Core GRU Cell.
        """
        # Create NL Embedding Matrix, with 0 Vector for PAD_ID (0) [Program Net]
        PE = tf.get_variable("P_Embedding", [len(self.word2id), self.embed_sz], initializer=self.init)
        zero_mask = tf.constant([0 if i == 0 else 1 for i in range(len(self.word2id))],
                                    dtype=tf.float32, shape=[len(self.word2id), 1])
        self.PE = PE * zero_mask

        # Create NL Embedding Matrix, with 0 Vector for PAD_ID (0) [Arg Net]
        AE = tf.get_variable("A_Embedding", [len(self.word2id), self.embed_sz], initializer=self.init)
        self.AE = AE * zero_mask

    def encode_input(self):
        """
        Map Natural Language Directive to Fixed Size Vector Embedding.
        """
        p_directive_embedding = tf.nn.embedding_lookup(self.PE, self.X)          # [None, sent_len, embed_sz]
        p_directive_embedding = tf.nn.dropout(p_directive_embedding, self.keep_prob)

        a_directive_embedding = tf.nn.embedding_lookup(self.AE, self.X)          # [None, sent_len, embed_sz]
        a_directive_embedding = tf.nn.dropout(a_directive_embedding, self.keep_prob)

        with tf.variable_scope("P_Encoder"):
            self.p_encoder_gru = tf.contrib.rnn.GRUCell(self.embed_sz)
            _, p_state = tf.nn.dynamic_rnn(self.p_encoder_gru, p_directive_embedding, sequence_length=self.X_len, dtype=tf.float32)
        
        with tf.variable_scope("A_Encoder"):
            self.a_encoder_gru = tf.contrib.rnn.GRUCell(self.embed_sz)
            _, a_state = tf.nn.dynamic_rnn(self.a_encoder_gru, a_directive_embedding, sequence_length=self.X_len, dtype=tf.float32)
        
        return p_state, a_state

    def terminate_net(self):
        """
        Build the NPI Termination Network, that takes in the NPI Core Hidden State, and returns
        the probability of terminating program.
        """
        p_terminate = tflearn.fully_connected(self.prog_s, 2, activation='linear', regularizer='L2')
        return p_terminate                                                   # Shape: [bsz, 2]

    def program_net(self):
        """
        Build the NPI Key Network, that takes in the NPI Core Hidden State, and returns a softmax
        distribution over possible next programs.
        """
        # Compute Distribution over Programs
        hidden = tflearn.fully_connected(self.prog_s, self.key_dim, activation='relu', regularizer='L2')
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
            arg_hidden = tflearn.fully_connected(self.arg_s, self.key_dim, activation='relu', regularizer='L2')
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
                loss, p_acc, a1_acc, _, _ = self.session.run([self.loss, self.p_accuracy, self.a1_accuracy,
                                                              self.p_train_op, self.a_train_op], feed_dict={
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

    def score(self, nl_command, length):
        """
        Given a natural language command, return predicted output and score.

        :return: List of tokens representing predicted command, and score.
        """
        prog, a1 = self.session.run([self.program_distribution, self.arguments[0]],
                                    feed_dict={self.X: [nl_command], self.X_len: [length], self.P: [self.progs["<<GO>>"]], self.keep_prob: 1.0})
        pred_prog = np.argmax(prog, axis=1)[0]
        
        best_a, best_val = -1, -1000000000
        for a in range(len(a1[0])):
            if self.id2prog[pred_prog] in ['Down', 'Left', 'Up', 'Right', 'down', 'West', 'North', 'South']:
                if not self.id2arg[a].isdigit():
                    continue 
                else:
                   if a1[0][a] > best_val:
                       best_a, best_val = a, a1[0][a]
            else:
                if self.id2arg[a].isdigit():
                    continue
                else:
                    if a1[0][a] > best_val:
                       best_a, best_val = a, a1[0][a]

        return pred_prog, best_a

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
        with open(self.means_train_path, 'r') as f:
            means_segments, means_programs = map(list, zip(*pickle.load(f)))
            means_programs = map(lambda x: x.split(), means_programs)
    
        assert(len(means_segments) == len(means_programs))
        
        with open(self.ends_train_path + ".en", 'r') as f:
            ends_sentences = [x.split() for x in f.readlines()]
            ends_sentences = ends_sentences[:(9 * (len(ends_sentences) / 10))]
            
        with open(self.ends_train_path + ".ml", 'r') as f:
            ends_programs = [x.split() for x in f.readlines()]
            ends_programs = ends_programs[:(9 * (len(ends_programs) / 10))]
            for i in range(len(ends_programs)):
                if len(ends_programs[i]) == 4:
                    program = ends_programs[i]
                    ends_programs[i] = [program[0] + "_" + program[2], program[1] + "_" + program[3]]
        
        if self.is_pik:
            with open(self.pik_train, 'r') as f:
                ends_sentences, ends_labels = map(list, zip(*pickle.load(f)))
                ends_sentences = map(lambda x: [j.lower() for j in x], ends_sentences)

            ends_programs = []
            for i in range(len(ends_labels)):
                label = ends_labels[i].split()
                if len(label) == 4:
                    label = [label[0] + "_" + label[2], label[1] + "_" + label[3]]
                ends_programs.append(label)
        
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

        ### CHANGE THIS FOR LEARNING CURVE ###
        self.train_set = self.train_set[:self.until]
        ######################################
        
        # Parse Test Data
        with open(self.means_test_path, 'r') as f:
            means_segments, means_programs = map(list, zip(*pickle.load(f)))
            means_programs = map(lambda x: x.split(), means_programs)
        
        assert(len(means_segments) == len(means_programs))
        
        with open(self.ends_test_path + ".en", 'r') as f:
            ends_sentences = [x.split() for x in f.readlines()]
            ends_sentences = ends_sentences[(9 * (len(ends_sentences) / 10)):]
            
        with open(self.ends_test_path + ".ml", 'r') as f:
            ends_programs = [x.split() for x in f.readlines()]
            ends_programs = ends_programs[(9 * (len(ends_programs) / 10)):]

            for i in range(len(ends_programs)):
                if len(ends_programs[i]) == 4:
                    program = ends_programs[i]
                    ends_programs[i] = [program[0] + "_" + program[2], program[1] + "_" + program[3]]
        
        if self.is_pik:
            with open(self.pik_test, 'r') as f:
                ends_sentences, ends_labels = map(list, zip(*pickle.load(f)))
                ends_sentences = map(lambda x: [j.lower() for j in x], ends_sentences)

            ends_programs = []
            for i in range(len(ends_labels)):
                label = ends_labels[i].split()
                if len(label) == 4:
                    label = [label[0] + "_" + label[2], label[1] + "_" + label[3]]
                ends_programs.append(label)
        
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
            if len(program) != 2:
                print i, program
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