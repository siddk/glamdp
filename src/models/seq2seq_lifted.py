"""
seq2seq_lifted.py

Implements sequence to sequence translation from natural language to lifted reward function.
Using GRU encoder and decoder.
"""
import numpy as np
import tensorflow as tf
import random as r
import time

r.seed(1)

# PAD + UNK Tokens
PAD, PAD_ID = "<<PAD>>", 0
UNK, UNK_ID = "<<UNK>>", 1

class Seq2Seq_Lifted():
    def __init__(self, parallel_corpus_train, parallel_corpus_test, embedding_size=30, rnn_size=64, h1_size=60,
        h2_size=50, epochs=10, batch_size=32, verbose=1, pct_test=0.2, save_to=None, restore_from=None):
        """
        Instantiates and Trains Model using the given parallel corpus.

        :params parallel_corpus_(train, test): List of Tuples, where each tuple has two elements:
                    1) List of source sentence tokens
                    2) List of target sentence tokens
        """
        self.pc_train, self.pc_test, self.epochs, self.bsz, self.verbose = parallel_corpus_train, parallel_corpus_test, epochs, batch_size, verbose
        self.embedding_sz, self.rnn_sz, self.h1_sz, self.h2_sz = embedding_size, rnn_size, h1_size, h2_size

        # Create initializer and session
        self.init = tf.truncated_normal_initializer(stddev=0.1)
        self.session = tf.Session()



        #Unpack training and test data
        encoder_inputs_train, decoder_inputs_train = self.pc_train
        encoder_inputs_test, ground_truth_test = self.pc_test

        print "{} training, {} test".format(len(decoder_inputs_train), len(ground_truth_test))

        # Build vocab from training parallel corpus

        self.word2id_encoder, self.id2word_encoder = self.build_vocabulary(encoder_inputs_train, True)
        self.word2id_decoder, self.id2word_decoder = self.build_vocabulary(decoder_inputs_train, False)

        self.GO_ID = self.word2id_decoder['<<GO>>']
        self.EOS_ID = self.word2id_decoder['<<EOS>>']

        # Vectorize both the encoder and decoder corpora + get length
        self.encoder_lengths_train = np.array([len(n) for n in encoder_inputs_train], dtype=np.int32)
        self.decoder_lengths_train = np.array([len(n) for n in decoder_inputs_train], dtype=np.int32)

        #Produce lengths and vectorizing
        self.encoder_lengths_test = np.array([len(n) for n in encoder_inputs_test], dtype=np.int32)
        self.ground_truth_length= np.array([len(n) for n in ground_truth_test], dtype=np.int32)

        #Get maximum length of training and test data
        self.max_encoder_length = max(max(self.encoder_lengths_train), max(self.encoder_lengths_test))
        self.max_decoder_length = max(max(self.decoder_lengths_train), max(self.ground_truth_length))

        self.train_enc = self.vectorize(encoder_inputs_train, self.word2id_encoder, self.max_encoder_length, flip=True)
        self.train_dec = self.vectorize(decoder_inputs_train, self.word2id_decoder, self.max_decoder_length)

        self.test_enc = self.vectorize(encoder_inputs_test, self.word2id_encoder, self.max_encoder_length, flip=True)
        self.test_dec = self.vectorize(ground_truth_test, self.word2id_decoder, self.max_decoder_length)

        # Create placeholders
        self.X = tf.placeholder(tf.int32, shape=[None, self.max_encoder_length], name='NL_Command')
        self.Y = tf.placeholder(tf.int32, shape=[None, self.max_decoder_length], name='ML_Command')

        self.X_len = tf.placeholder(tf.int32, shape=[None], name='NL_Length')
        #self.Y_len = tf.placeholder(tf.int32, shape=[None], name='ML_Length')
        #to fix tensor size issue when computing loss function
        #probably related to the ML sentence size technically not being the vector size - adding GO and EOS tokens?
        self.Y_len = tf.constant(7, shape=[batch_size], name='ML_Length')
        # self.keep_prob = tf.placeholder(tf.float32, name='Dropout_Prob')

        # Instantiate Network Weights
        self.instantiate_weights()

        # Build Inference Pipeline
        self.train_logits, self.inference_logits = self.inference()

        # Build loss computation
        self.loss_val = self.loss()

        # Build Training Operation
        self.train_op = tf.train.AdamOptimizer(.001).minimize(self.loss_val)

        # Build Saver
        self.saver = tf.train.Saver()

        self.save_to = save_to

        # Initialize all variables
        self.session.run(tf.global_variables_initializer())

        if restore_from:
            print "restoring from {}".format(restore_from)
            self.saver.restore(self.session, restore_from)

    def build_vocabulary(self, corpus, is_encoder):
        """
        Builds the vocabulary from the given corpus.

        :return: Tuple of Word2Id, Id2Word Dictionaries.
        """
        vocab = set()
        for seq in corpus:
            [vocab.add(word) for word in seq]

        id2word = [PAD, UNK] + list(vocab) if is_encoder else [PAD] + list(vocab) #maybe don't add UNK to decoder?
        word2id = {id2word[i]: i for i in range(len(id2word))}
        return word2id, id2word

    def vectorize(self, corpus, word2id, vec_length, flip=False):
        """
        vectorizes the corpus, converting each sequence to vectors.
        <<GO>> and <<EOS>> tokens added during preprocessing.
        """
        vecs = []
        for seq in corpus:
            vec = np.zeros((vec_length,), dtype=np.int32)
            for i in range(len(seq)):
                vec[i] = word2id.get(seq[i], UNK_ID) #handling unknown words
            vecs.append(vec)

        vecs = np.array(vecs, dtype=np.int32)
        if flip:
            return np.flip(vecs, 1)
        else:
            return vecs

    def instantiate_weights(self):
        """
        Instantiate Network Weights, including Embedding Layers for both Input/Output,
        as well as Encoder/Decoder GRU Cells.
        """
        # Set up Embeddings
        self.encoder_embedding = tf.get_variable("Encoder_Embeddings", shape=[len(self.word2id_encoder),
                                                                              self.embedding_sz], initializer=self.init)
        self.decoder_embedding = tf.get_variable("Decoder_Embeddings", shape=[len(self.word2id_decoder),
                                                                              self.embedding_sz], initializer=self.init)

    def inference(self):
        """
        Build the inference computation graph, going from the input (natural language) to the
        output (machine language).
        """
        # Embed Encoder Inputs
        encoder_embeddings = tf.nn.embedding_lookup(self.encoder_embedding, self.X)   # Shape: [None, max_nl, embed_sz]

        # Feed through encoder
        with tf.variable_scope("Encoder") as encoder_scope:
            # Feed through Encoder GRU
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=tf.contrib.rnn.GRUCell(self.rnn_sz),
                                                                inputs=encoder_embeddings,
                                                                sequence_length=self.X_len,
                                                                dtype=tf.float32,
                                                                scope=encoder_scope)
            # Build Attention States
            attention_states = encoder_outputs

        # Feed through decoder
        with tf.variable_scope("Decoder") as decoder_scope:
            # Prepare Attention Mechanism
            attn_keys, attn_vals, attn_score_fn, attn_construct_fn = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                        "luong", self.rnn_sz)

            # Setup Decoder Train Function
            decoder_train_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state=encoder_state,
                                                                             attention_keys=attn_keys,
                                                                             attention_values=attn_vals,
                                                                             attention_score_fn=attn_score_fn,
                                                                             attention_construct_fn=attn_construct_fn)




            # Setup Output Function
            output_fn = lambda x: tf.contrib.layers.fully_connected(x, len(self.word2id_decoder), activation_fn=None)

            # Embed Decoder Symbols
            decoder_inputs = tf.nn.embedding_lookup(self.decoder_embedding, self.Y)

            # Run through Decoder (Train)
            decoder_cell = tf.contrib.rnn.GRUCell(self.rnn_sz)
            decoder_out_train, decoder_state_train, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(cell=decoder_cell,
                                                            decoder_fn=decoder_train_fn, inputs=decoder_inputs,
                                                            sequence_length=self.Y_len, scope=decoder_scope)
            # Map to distribution over tokens
            decoder_outputs_train = output_fn(decoder_out_train)

            # Reuse Variable Scope
            decoder_scope.reuse_variables()

            # Setup Decoder Inference Function
            #NOTE: using max_decoder_length - 1 because that's what they did in the tests, I don't think they know why either
            decoder_inference_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn=output_fn,
                                        encoder_state=encoder_state, attention_keys=attn_keys,
                                        attention_values=attn_vals, attention_score_fn=attn_score_fn,
                                        attention_construct_fn=attn_construct_fn, embeddings=self.decoder_embedding,
                                        start_of_sequence_id=self.GO_ID, end_of_sequence_id=self.EOS_ID,
                                        maximum_length=self.max_decoder_length - 1, num_decoder_symbols=len(self.word2id_decoder),
                                        dtype=tf.int32)

            # Run through Decoder (Inference)
            decoder_outputs_inference, decoder_state_inference, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(cell=decoder_cell,
                                                                         decoder_fn=decoder_inference_fn, scope=decoder_scope)

        return decoder_outputs_train, decoder_outputs_inference

    def loss(self):
        """
        Build loss computation, dependent on weights.
        """
        #TODO: with const. lengths, is there unfair penalization for sentences out of order
        weights = tf.sequence_mask(self.Y_len, self.max_decoder_length, dtype=tf.float32)
        #print weights
        loss = tf.contrib.seq2seq.sequence_loss(self.train_logits, self.Y, weights)
        return loss

    def fit(self):

        #start = time.time()
        for e in range(self.epochs):
            curr_loss, batches = 0.0, 0
            for start, end in zip(range(0, len(self.train_enc) - self.bsz, self.bsz),
                                  range(self.bsz, len(self.train_enc), self.bsz)):

                loss, _ = self.session.run([self.loss_val, self.train_op],
                                           feed_dict={self.X: self.train_enc[start:end],
                                                      self.X_len: self.encoder_lengths_train[start:end],
                                                      self.Y: self.train_dec[start:end]})
                curr_loss += loss
                batches += 1
                if self.verbose == 1:
                    print 'Epoch %d Batch %d\tCurrent Loss: %.3f' % (e, batches, curr_loss / batches)

            print 'Epoch %s Average Loss:' % str(e), curr_loss / batches

        #end = time.time()
        #print "Training finished in {} seconds".format(end - start)

        #Save trained model
        if self.save_to:
            self.saver.save(self.session, self.save_to)
            print "saved trained model"

    def test(self, print_example=False):
        """
        Performs tests on validation data.
        """

        y = self.session.run(self.inference_logits, feed_dict={self.X: self.test_enc,
                                                    self.X_len: self.encoder_lengths_test})

        estimates = np.argmax(y, axis=2)
        if print_example:
            #Displaying example
            cmd_num = 0
            nl = [self.id2word_encoder[i] for i in self.test_enc[cmd_num]]
            estim = [self.id2word_decoder[i] for i in estimates[cmd_num]]
            truth = [self.id2word_decoder[i] for i in self.test_dec[cmd_num]]
            print "NL Command: {}".format(str(nl))
            print "Estimated RF: {} , {} tokens".format(str(estim), len(estim))
            print "True RF: {}, {} tokens".format(str(truth), len(truth))

        #Elementwise correctness
        num_correct = np.equal(estimates, self.test_dec)
        print "percent of tokens correct: {}".format(float(np.sum(num_correct))/np.prod(num_correct.shape))


    def translate(self, nl_command):
        """
        translates a natural lanuage command into machine language with the trained model.
        """
