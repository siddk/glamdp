"""
super_draggn.py
"""
from keras.layers import Dense
from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf


class SuperDRAGGN:
    def __init__(self, trainX, trainX_len, trainY, word2id, labels, embed_sz=30, rnn_sz=128, bsz=32,
                 init=tf.truncated_normal_initializer(stddev=0.1)):
        """
        Instantiate SuperDRAGGN Model with necessary hyperparameters.
        """
        self.trainX, self.trainX_len, self.trainY = trainX, trainX_len, trainY
        self.word2id, self.labels = word2id, labels
        self.embed_sz, self.rnn_sz, self.bsz, self.init = embed_sz, rnn_sz, bsz, init
        self.session = tf.Session()

        # Setup Placeholders
        self.X = tf.placeholder(tf.int64, shape=[None, trainX.shape[1]], name='Utterance')
        self.X_len = tf.placeholder(tf.int64, shape=[None], name='Utterance_Length')
        self.Y = tf.placeholder(tf.int64, shape=[None], name='Label')
        self.keep_prob = tf.placeholder(tf.float32, name='Dropout_Probability')

        # Build Inference Pipeline
        self.logits = self.inference()

        # Build Loss Computation
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.Y, self.logits)

        # Build Training Operation
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        # Build Accuracy Operation
        correct = tf.equal(tf.argmax(self.logits, 1), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="Accuracy")

        # Initialize all Variables
        self.session.run(tf.global_variables_initializer())

    def inference(self):
        # Create NL Embedding Matrix, with 0 Vector for PAD_ID (0) [Program Net]
        E = tf.get_variable("Embedding", [len(self.word2id), self.embed_sz], initializer=self.init)
        zero_mask = tf.constant([0 if i == 0 else 1 for i in range(len(self.word2id))],
                                dtype=tf.float32, shape=[len(self.word2id), 1])
        self.E = E * zero_mask

        # Embed Input
        embedding = tf.nn.embedding_lookup(self.E, self.X)
        embedding = tf.nn.dropout(embedding, self.keep_prob)

        # Feed through RNN
        self.encoder_gru = tf.contrib.rnn.GRUCell(self.rnn_sz)
        _, state = tf.nn.dynamic_rnn(self.encoder_gru, embedding, sequence_length=self.X_len, dtype=tf.float32)

        # Feed-Forward Layers
        hidden = Dense(self.rnn_sz, activation='relu')(state)
        hidden = tf.nn.dropout(hidden, self.keep_prob)
        logits = Dense(len(self.labels), activation='linear')(hidden)
        return logits

    def fit(self, epochs):
        chunk_size = (len(self.trainX) / self.bsz) * self.bsz
        for e in range(epochs):
            curr_loss, curr_acc, batches = 0.0, 0.0, 0.0
            for start, end in zip(range(0, chunk_size - self.bsz, self.bsz), range(self.bsz, chunk_size, self.bsz)):
                loss, acc, _ = self.session.run([self.loss, self.accuracy, self.train_op],
                                                feed_dict={self.X: self.trainX[start:end],
                                                           self.X_len: self.trainX_len[start:end],
                                                           self.Y: self.trainY[start:end],
                                                           self.keep_prob: 0.5})
                curr_loss, curr_acc, batches = curr_loss + loss, curr_acc + acc, batches + 1

            print 'Epoch %d\tAverage Loss: %.3f\tAverage Accuracy: %.3f' % \
                  (e + 1, curr_loss / batches, curr_acc / batches)

    def eval(self, testX, testX_len, testY):
        logits = self.session.run(self.logits, feed_dict={self.X: testX, self.X_len: testX_len, self.Y: testY,
                                                       self.keep_prob: 1.0})
        predictions = np.argmax(logits, axis=1)
        print 'Test F1 Score: %.3f' % f1_score(testY, predictions, average='micro')