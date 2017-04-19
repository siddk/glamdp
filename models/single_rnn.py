"""
single_rnn.py

Core model definition file for the Single-RNN Language Grounding Model. Enumerates
the total set of possible reward functions as possible outputs, and predicts
singular labels.
"""
import numpy as np
import tensorflow as tf

PAD, PAD_ID = "<<PAD>>", 0
UNK, UNK_ID = "<<UNK>>", 1

class SingleRNN():
    def __init__(self, train_path, test_path, embedding_size=30, rnn_size=50, h1_size=60,
                 h2_size=50, epochs=10, batch_size=16):
        """
        Instantiate a SingleRNN Model, with the necessary parameters.

        :param train_en: Path to training natural language directives.
        :param train_rf: Path to training reward function strings.
        """
        self.embedding_sz, self.rnn_sz, self.h1_sz, self.h2_sz = embedding_size, rnn_size, h1_size, h2_size
        self.init, self.bsz, self.epochs = tf.truncated_normal_initializer(stddev=0.5), batch_size, epochs
        self.session = tf.Session()

        # Read Data + Assemble Commands, Parallel Corpus
        with open(train_path + ".en", 'r') as f:
            self.train_en = [x.split() for x in f.readlines()]
        with open(train_path + ".ml", 'r') as f:
            self.train_rf = [x.strip() for x in f.readlines()]
        with open(test_path + ".en", 'r') as f:
            self.test_en = [x.split() for x in f.readlines()]
        with open(test_path + ".ml", 'r') as f:
            self.test_rf = [x.strip() for x in f.readlines()]
        self.commands = {rf: i for i, rf in enumerate(list(set(self.train_rf)))}
        self.pc = zip(self.train_en, map(lambda x: self.commands[x], self.train_rf))
        self.test_pc = zip(self.test_en, map(lambda x: self.commands[x], self.test_rf))

        # Build vocabulary
        self.word2id, self.id2word = self.build_vocabulary()

        # Vectorize Parallel Corpus
        self.lengths = [len(n) for n, _ in self.pc]
        self.train_x, self.train_y = self.vectorize()

         # Setup Placeholders
        self.X = tf.placeholder(tf.int64, shape=[None, self.train_x.shape[-1]], name='NL_Directive')
        self.Y = tf.placeholder(tf.int64, shape=[None], name='Lifted_RF')
        self.X_len = tf.placeholder(tf.int64, shape=[None], name='NL_Length')
        self.keep_prob = tf.placeholder(tf.float32, name='Dropout_Prob')

        # Build Inference Graph
        self.logits = self.inference()

        # Build Loss Computation
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.Y, self.logits)
        self.probs = tf.nn.softmax(self.logits)

        # Create Accuracy Operation
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

        # Build Training Operation
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        # Build Saver
        self.saver = tf.train.Saver()

        # Initialize all variables
        self.session.run(tf.global_variables_initializer())

    def build_vocabulary(self):
        """
        Builds the vocabulary from the parallel corpus, adding the UNK ID.

        :return: Tuple of Word2Id, Id2Word Dictionaries.
        """
        vocab = set()
        for n, _ in self.pc:
            for word in n:
                vocab.add(word)

        id2word = [PAD, UNK] + list(vocab)
        word2id = {id2word[i]: i for i in range(len(id2word))}
        return word2id, id2word

    def vectorize(self):
        """
        Step through the Parallel Corpus, and convert each sequence to vectors.
        """
        x, y = [], []
        for nl, ml in self.pc:
            nvec, mlab = np.zeros((max(self.lengths)), dtype=np.int32), ml
            for i in range(len(nl)):
                nvec[i] = self.word2id.get(nl[i], UNK_ID)
            x.append(nvec)
            y.append(mlab)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    def inference(self):
        """
        Compile the LSTM Classifier, taking the input placeholder, generating the softmax
        distribution over all possible reward functions.
        """
        # Embedding
        E = tf.get_variable("Embedding", shape=[len(self.word2id), self.embedding_sz],
                            dtype=tf.float32, initializer=self.init)
        embedding = tf.nn.embedding_lookup(E, self.X)               # Shape [None, x_len, embed_sz]
        embedding = tf.nn.dropout(embedding, self.keep_prob)

        # LSTM
        cell = tf.contrib.rnn.GRUCell(self.rnn_sz)
        _, state = tf.nn.dynamic_rnn(cell, embedding, sequence_length=self.X_len, dtype=tf.float32)
        h_state = state                                             # Shape [None, lstm_sz]

        # ReLU Layer 1
        H1_W = tf.get_variable("H1_W", shape=[self.rnn_sz, self.h1_sz], dtype=tf.float32,
                               initializer=self.init)
        H1_B = tf.get_variable("H1_B", shape=[self.h1_sz], dtype=tf.float32,
                               initializer=self.init)
        h1 = tf.nn.relu(tf.matmul(h_state, H1_W) + H1_B)

        # ReLU Layer 2
        H2_W = tf.get_variable("H2_W", shape=[self.h1_sz, self.h2_sz], dtype=tf.float32,
                               initializer=self.init)
        H2_B = tf.get_variable("H2_B", shape=[self.h2_sz], dtype=tf.float32,
                               initializer=self.init)
        hidden = tf.nn.relu(tf.matmul(h1, H2_W) + H2_B)
        hidden = tf.nn.dropout(hidden, self.keep_prob)

        # Output Layer
        O_W = tf.get_variable("Output_W", shape=[self.h2_sz, len(self.commands)],
                              dtype=tf.float32, initializer=self.init)
        O_B = tf.get_variable("Output_B", shape=[len(self.commands)], dtype=tf.float32,
                              initializer=self.init)
        output = tf.matmul(hidden, O_W) + O_B
        return output

    def fit(self):
        """
        Train the model, with the specified batch size and number of epochs.
        """
        # Run through epochs
        chunk_size = (len(self.train_x) / self.bsz) * self.bsz
        for e in range(self.epochs):
            curr_loss, curr_acc, batches = 0.0, 0.0, 0.0
            for start, end in zip(range(0, len(self.train_x[:chunk_size]) - self.bsz, self.bsz),
                                  range(self.bsz, len(self.train_x[:chunk_size]), self.bsz)):
                loss, acc, _ = self.session.run([self.loss, self.accuracy, self.train_op],
                                                feed_dict={self.X: self.train_x[start:end],
                                                           self.X_len: self.lengths[start:end],
                                                           self.keep_prob: 0.5,
                                                           self.Y: self.train_y[start:end]})
                curr_loss, curr_acc, batches = curr_loss + loss, curr_acc + acc, batches + 1
            print 'Epoch %d\tAverage Loss: %.3f\tAverage Accuracy: %.3f' % (e, curr_loss / batches, curr_acc / batches)

    def eval(self):
        """
        Perform an evaluation epoch, running through the test data, returning accuracy.
        """
        num_correct = 0.0
        for (nl_command, rf) in self.test_pc:
            pred_rf, _ = self.score(nl_command)
            if pred_rf == rf:
                num_correct += 1

        acc = (num_correct / len(self.test_pc))
        print "Test Accuracy: %.3f" % acc
        return acc


    def score(self, nl_command):
        """
        Given a natural language command, return predicted output and score.

        :return: List of tokens representing predicted command, and score.
        """
        seq, seq_len = np.zeros((max(self.lengths))), len(nl_command)
        for i in range(min(len(nl_command), len(seq))):
            seq[i] = self.word2id.get(nl_command[i], UNK_ID)
        y = self.session.run(self.probs, feed_dict={self.X: [seq], self.X_len: [seq_len],
                                                    self.keep_prob: 1.0})
        [pred_command] = np.argmax(y, axis=1)
        return pred_command, y[0][pred_command]
