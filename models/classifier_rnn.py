"""
classifier_rnn.py

Model definition file for an RNN-based classifier sorting language into "low-level/action-specifying" and "high-level/goal-specifying language"
Basically the same as the single-rnn model
"""
import numpy as np
import tensorflow as tf

PAD, PAD_ID = "<<PAD>>", 0
UNK, UNK_ID = "<<UNK>>", 1

class ClassifierRNN():
    def __init__(self, train_path, test_path, embedding_size=30, rnn_size=50, h1_size=60,
                 h2_size=50, epochs=10, batch_size=16, num_categories=2):
        """
        Instantiate RNN language classifier model.
        """

        self.rnn_sz, self.embed_sz, self.num_categories = rnn_size, embedding_size, num_categories
        self.h1_sz, self.h2_sz = h1_size, h2_size
        self.init, self.bsz, self.epochs = tf.truncated_normal_initializer(stddev=0.5), batch_size, epochs
        self.epochs = epochs
        self.session = tf.Session() #Sidd's altar of the dark god TensorFlow

        #Load data
        self.train_data, self.train_labels = self.read_data(train_path)
        self.test_data, self.test_labels = self.read_data(test_path)

        # Build vocabulary
        self.word2id, self.id2word = self.build_vocabulary()

        # Vectorize training and test data corpus
        self.train_lengths = [len(n) for n in self.train_data]
        self.train_x = self.vectorize(self.train_data, max(self.train_lengths))

        self.test_lengths = [len(n) for n in self.test_data]
        self.test_x = self.vectorize(self.test_data, max(self.train_lengths))

        # Setup Placeholders
        self.X = tf.placeholder(tf.int64, shape=[None, self.train_x.shape[-1]], name='NL_Command')
        self.Y = tf.placeholder(tf.int64, shape=[None], name='Prediction')
        self.X_len = tf.placeholder(tf.int64, shape=[None], name='NL_Length')
        self.keep_prob = tf.placeholder(tf.float32, name='Dropout_Prob')

        #build inference graph
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

    def read_data(self, path):
        """
        reads data from file.
        """

        data = []
        labels = []
        with open(path, 'r') as f:
            for line in f:
                d, l = tuple(line.split(":"))
                data.append(d)
                labels.append(int(l.strip()))

        return data, np.array(labels) #return as numpy array

    def build_vocabulary(self):
        """
        Builds the vocabulary from the training language, adding PAD and UNK.

        :return: Tuple of Word2Id, Id2Word Dictionaries.
        """
        vocab = set()
        for phrase in self.train_data:
            for word in phrase:
                vocab.add(word)

        id2word = [PAD, UNK] + list(vocab)
        word2id = {id2word[i]: i for i in range(len(id2word))}
        return word2id, id2word
    def vectorize(self, data, vec_len):
        """
        Convert each natural language phrase into a vector of word tokens.
        """
        x = []
        for nl in data:
            nvec = np.zeros((vec_len), dtype=np.int32)
            for i in range(len(nl)):
                nvec[i] = self.word2id.get(nl[i], UNK_ID)
            x.append(nvec)
        return np.array(x, dtype=np.int32)

    def inference(self):
        """
        Define GRU inference graph.
        """
        # Embedding
        E = tf.get_variable("Embedding", shape=[len(self.word2id), self.embed_sz],
                            dtype=tf.float32, initializer=self.init)
        embedding = tf.nn.embedding_lookup(E, self.X)               # Shape [None, x_len, embed_sz]
        embedding = tf.nn.dropout(embedding, self.keep_prob)

        # GRU Cell
        cell = tf.contrib.rnn.GRUCell(self.rnn_sz)
        _, state = tf.nn.dynamic_rnn(cell, embedding, sequence_length=self.X_len, dtype=tf.float32)
        h_state = state                                             # Shape [None, rnn_sz]

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
        O_W = tf.get_variable("Output_W", shape=[self.h2_sz, self.num_categories],
                              dtype=tf.float32, initializer=self.init)
        O_B = tf.get_variable("Output_B", shape=[self.num_categories], dtype=tf.float32,
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
                                                           self.X_len: self.train_lengths[start:end],
                                                           self.keep_prob: 0.5,
                                                           self.Y: self.train_labels[start:end]})
                curr_loss, curr_acc, batches = curr_loss + loss, curr_acc + acc, batches + 1
            print 'Epoch %d\tAverage Loss: %.3f\tAverage Accuracy: %.3f' % (e, curr_loss / batches, curr_acc / batches)

    def eval(self):
        """
        Evaluate the model against all test data
        """
        y = self.session.run(self.probs, feed_dict={self.X: self.test_x, self.X_len: self.test_lengths,
                                                    self.keep_prob: 1.0})
        pred = np.argmax(y, axis=1)
        accuracy = np.sum(np.equal(pred, self.test_labels))/float(len(self.test_labels))
        print "test correctness: {}".format(accuracy)

    def score(self, nl_command):
        """
        Given a natural language command, return predicted output and score.

        :return: List of tokens representing predicted command, and score.
        """
        seq, seq_len = np.zeros((max(self.train_lengths))), len(nl_command)
        for i in range(min(len(nl_command), len(seq))):
            seq[i] = self.word2id.get(nl_command[i], UNK_ID)
        y = self.session.run(self.probs, feed_dict={self.X: [seq], self.X_len: [seq_len],
                                                    self.keep_prob: 1.0})
        [pred_class] = np.argmax(y, axis=1)
        return pred_class, y[0][pred_class]
