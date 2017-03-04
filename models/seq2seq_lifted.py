"""
seq2seq_lifted.py

Implements sequence to sequence translation from natural language to lifted reward function.
Using GRU encoder and decoder.

"""

import numpy as np
import tensorflow as tf

PAD, PAD_ID = "<<PAD>>", 0
UNK, UNK_ID = "<<UNK>>", 1
#tokens for decoder inputs
GO, GO_ID = "<<GO>>", 2
EOS, EOS_ID = "<<EOS>>", 3

class seq2seq_lifted():
    def __init__(self, parallel_corpus, embedding_size=30, rnn_size=50, h1_size=60,
        h2_size=50, epochs=10, batch_size=16, verbose=1):
        """
        Instantiates and Trains Model using the given parallel corpus.

        :param parallel_corpus: List of Tuples, where each tuple has two elements:
                    1) List of source sentence tokens
                    2) List of target sentence tokens
        """
        self.pc, self.epochs, self.bsz, self.verbose = parallel_corpus, epochs, batch_size, verbose
        self.embedding_sz, self.rnn_sz, self.h1_sz, self.h2_sz = embedding_size, rnn_size, h1_size, h2_size
        #create initializer and session
        self.init = tf.truncated_normal_initializer(stddev=0.5)
        self.session = tf.Session()

        #build vocab from parallel corpus
        encoder_inputs, decoder_inputs = self.pc

        self.word2id_encoder, self.id2word_encoder = self.build_vocabulary(encoder_inputs, False)
        self.word2id_decoder, self.id2word_decoder = self.build_vocabulary(decoder_inputs, True)


        #vectorize both the encoder and decoder corpuses
        self.encoder_lengths = [len(n) for n in encoder_inputs]
        self.decoder_lengths = [len(n) for n in decoder_inputs]

        self.train_enc = self.vectorize(encoder_inputs, self.word2id_encoder, max(self.encoder_lengths), False)
        self.train_dec = self.vectorize(decoder_inputs, self.word2id_decoder, max(self.decoder_lengths), True)

        #create placeholders
        self.X = tf.placeholder(tf.int32, shape=[None, self.train_enc.shape[-1]], name='NL_Command')
        self.Y = tf.placeholder(tf.int32, shape=[None, self.train_dec.shape[-1]], name='ML_Command')
        self.weights = tf.placeholder(tf.float32, shape=[None, self.train_dec.shape[-1]], name="train_weights")

        self.X_len = tf.placeholder(tf.int32, shape=[None], name='NL_Length')
        self.Y_len = tf.placeholder(tf.int32, shape=[None], name='NL_Length')

        self.keep_prob = tf.placeholder(tf.float32, name='Dropout_Prob')

        self.logits = self.build_model_train()

        #build loss computation
        #use all ones for weights
        self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.Y, self.weights)

        # Build Training Operation
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        # Build Saver
        self.saver = tf.train.Saver()

        # Initialize all variables
        self.session.run(tf.global_variables_initializer())


    def build_vocabulary(self, corpus, is_decoder):
        """
        Builds the vocabulary from the given corpus.
        Adds PAD, UNK, GO, and EOS tokens if the corpus is the decoder corpus,
        otherwise adds PAD and UNK.

        :return: Tuple of Word2Id, Id2Word Dictionaries.
        """

        vocab = set()
        for seq in corpus:
            #print seq
            [vocab.add(word) for word in seq]

        #adding extra tokens
        tokens = [PAD, UNK, GO, EOS] if is_decoder else [PAD, UNK]

        id2word = tokens + list(vocab)
        word2id = {id2word[i]: i for i in range(len(id2word))}
        return word2id, id2word

    def vectorize(self, corpus, word2id, vec_length, is_decoder):
        """
        vectorizes the corpus, converting each sequence to vectors.
        """
        vecs = []
        for seq in corpus:
            #add extra space for GO and EOS tokens
            vec_len = vec_length + 2 if is_decoder else vec_length
            offst = 1 if is_decoder else 0 #offset for decoder vectors
            vec = np.zeros((vec_len,), dtype=np.int32)
            for i in range(len(seq)):
                vec[i + offst] = word2id.get(seq[i], UNK_ID) #handling unknown words

            if is_decoder:
                vec[i + 2] = EOS_ID

            vecs.append(vec)

        return np.array(vecs, dtype=np.int32)

    def build_model_train(self):
        """
        Compiles the GRU sequence to sequence model for training.
        """

        #embeddings for encoder
        E_encode = tf.get_variable("embedding_encode", shape=[len(self.word2id_encoder), self.embedding_sz],
            dtype=tf.float32, initializer=self.init)

        embedding_encode = tf.nn.embedding_lookup(E_encode, self.X)
        embedding_encode = tf.nn.dropout(embedding_encode, self.keep_prob)

        E_decode= tf.get_variable("embedding_decode", shape=[len(self.word2id_decoder), self.embedding_sz],
            dtype=tf.float32, initializer=self.init)

        embedding_decode = tf.nn.embedding_lookup(E_decode, self.Y)
        embedding_decode = tf.nn.dropout(embedding_decode, self.keep_prob)

        #set variable scope for cells to avoid name collision
        with tf.variable_scope('encode'):
            gru_encode = tf.contrib.rnn.GRUCell(self.rnn_sz)
            #create encoder RNN
            _, enc_state = tf.nn.dynamic_rnn(gru_encode, embedding_encode, sequence_length=self.X_len, dtype=tf.float32)


        with tf.variable_scope('decode'):
            gru_decode = tf.contrib.rnn.GRUCell(self.rnn_sz)
            #define decoder function for training
            decoder_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train(enc_state)
            #define inputs at training time
            final_outputs, final_state, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(gru_decode, decoder_fn_train, inputs=embedding_decode,
                sequence_length=self.Y_len)

        return final_outputs

    def fit(self, chunk_size):
        """
        Train the model, with the specified batch size and number of epochs.
        """
        # Run through epochs
        for e in range(self.epochs):
            curr_loss, batches = 0.0, 0.0
            for start, end in zip(range(0, len(self.train_enc[:chunk_size]) - self.bsz, self.bsz),
                                  range(self.bsz, len(self.train_enc[:chunk_size]), self.bsz)):

                #create weights
                #I have no idea if this will work
                print self.train_dec[start:end].shape
                print len(self.decoder_lengths[start:end])
                weights_decoder = np.ones(self.train_dec[start:end].shape)

                loss, _ = self.session.run([self.loss, self.train_op],
                                           feed_dict={self.X: self.train_enc[start:end],
                                                      self.X_len: self.encoder_lengths[start:end],
                                                      self.keep_prob: 0.5,
                                                      self.Y: self.train_dec[start:end],
                                                      self.Y_len:self.decoder_lengths[start:end],
                                                      self.weights:weights_decoder})
                curr_loss += loss
                batches += 1
            if self.verbose == 1:
                print 'Epoch %s Average Loss:' % str(e), curr_loss / batches
