"""
seq2seq_lifted.py

Implements sequence to sequence translation from natural language to lifted reward function.
Using GRU encoder and decoder.
"""
import numpy as np
import tensorflow as tf
import random as r

r.seed(1)

# PAD + UNK Tokens
PAD, PAD_ID = "<<PAD>>", 0
UNK, UNK_ID = "<<UNK>>", 1

# Tokens for decoder inputs
GO, GO_ID = "<<GO>>", 1
EOS, EOS_ID = "<<EOS>>", 2

class Seq2Seq_Lifted():
    def __init__(self, parallel_corpus, embedding_size=30, rnn_size=64, h1_size=60,
        h2_size=50, epochs=10, batch_size=32, verbose=1, pct_test=0.2):
        """
        Instantiates and Trains Model using the given parallel corpus.

        :param parallel_corpus: List of Tuples, where each tuple has two elements:
                    1) List of source sentence tokens
                    2) List of target sentence tokens
        """
        self.pc, self.epochs, self.bsz, self.verbose = parallel_corpus, epochs, batch_size, verbose
        self.embedding_sz, self.rnn_sz, self.h1_sz, self.h2_sz = embedding_size, rnn_size, h1_size, h2_size

        # Create initializer and session
        self.init = tf.truncated_normal_initializer(stddev=0.1)
        self.session = tf.Session()

        #Zipping parallel corpus
        zipped_data = zip(*self.pc)

        #Shuffle parallel corpus
        r.shuffle(zipped_data)

        #Split parallel corpus into training and test data
        num_test = int(len(zipped_data)*pct_test)

        print "{} training phrases, {} test".format(num_test, len(zipped_data) - num_test)

        train_data = zipped_data[:num_test]
        test_data = zipped_data[num_test:]

        encoder_inputs_train, decoder_inputs_train = zip(*train_data) #unzipping
        encoder_inputs_test, decoder_inputs_test = zip(*test_data)

        # Build vocab from training parallel corpus

        self.word2id_encoder, self.id2word_encoder = self.build_vocabulary(encoder_inputs_train, False)
        self.word2id_decoder, self.id2word_decoder = self.build_vocabulary(decoder_inputs_train, True)

        # Vectorize both the encoder and decoder corpora + get length
        self.encoder_lengths_train = np.array([len(n) for n in encoder_inputs_train], dtype=np.int32)
        self.decoder_lengths_train = np.array([len(n) for n in decoder_inputs_train], dtype=np.int32)

        #Produce lengths and vectorizing
        self.encoder_lengths_test = np.array([len(n) for n in encoder_inputs_test], dtype=np.int32)
        self.decoder_lengths_test = np.array([len(n) for n in decoder_inputs_test], dtype=np.int32)

        #Get maximum length of training and test data
        self.max_encoder_length = max(max(self.encoder_lengths_train), max(self.encoder_lengths_test))
        self.max_decoder_length = max(max(self.decoder_lengths_train), max(self.decoder_lengths_test))

        self.train_enc = self.vectorize(encoder_inputs_train, self.word2id_encoder, self.max_encoder_length, False)
        self.train_dec = self.vectorize(decoder_inputs_train, self.word2id_decoder, self.max_decoder_length, True)

        self.test_enc = self.vectorize(encoder_inputs_test, self.word2id_encoder, self.max_encoder_length, False)
        self.test_dec = self.vectorize(decoder_inputs_test, self.word2id_decoder, self.max_decoder_length, True)

        self.max_decoder_length += 2 #to account for GO and EOS tokens

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
            [vocab.add(word) for word in seq]

        # Adding extra tokens
        tokens = [PAD, GO, EOS] if is_decoder else [PAD, UNK]

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
                vec[0] = GO_ID
                vec[i + 2] = EOS_ID

            if not is_decoder:
                vecs.append(vec)
            else:
                vecs.append(vec)
        return np.array(vecs, dtype=np.int32)

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
            #print 'DECODER OUT:', decoder_out_train
            # Map to Distribution
            decoder_outputs_train = output_fn(decoder_out_train)
            #print "DECODER OUTPUT FUNCTION SAYS: ", decoder_outputs_train
            #decoder_outputs_train = tf.reshape(decoder_outputs_train, [-1, self.max_ml_len, len(self.word2id_decoder)])

            # Reuse Variable Scope
            decoder_scope.reuse_variables()

            # Setup Decoder Inference Function
            decoder_inference_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn=output_fn,
                                        encoder_state=encoder_state, attention_keys=attn_keys,
                                        attention_values=attn_vals, attention_score_fn=attn_score_fn,
                                        attention_construct_fn=attn_construct_fn, embeddings=self.decoder_embedding,
                                        start_of_sequence_id=GO_ID, end_of_sequence_id=EOS_ID,
                                        maximum_length=self.max_decoder_length, num_decoder_symbols=len(self.word2id_decoder),
                                        dtype=tf.int32)

            # Run through Decoder (Inference)
            decoder_outputs_inference, decoder_state_inference, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(cell=decoder_cell,
                                                                         decoder_fn=decoder_inference_fn, scope=decoder_scope)

        return decoder_outputs_train, decoder_outputs_inference

    def loss(self):
        """
        Build loss computation, dependent on weights.
        """
        #print self.train_logits
        #print self.Y
        weights = tf.sequence_mask(self.Y_len, self.max_decoder_length, dtype=tf.float32)
        #print weights
        loss = tf.contrib.seq2seq.sequence_loss(self.train_logits, self.Y, weights)
        return loss

    def fit(self):
        for e in range(self.epochs):
            curr_loss, batches = 0.0, 0
            for start, end in zip(range(0, len(self.train_enc) - self.bsz, self.bsz),
                                  range(self.bsz, len(self.train_enc), self.bsz)):

                #print self.train_dec[start:end][0]

                loss, _ = self.session.run([self.loss_val, self.train_op],
                                           feed_dict={self.X: self.train_enc[start:end],
                                                      self.X_len: self.encoder_lengths_train[start:end],
                                                      self.Y: self.train_dec[start:end]})
                curr_loss += loss
                batches += 1
                print 'Epoch %d Batch %d\tCurrent Loss: %.3f' % (e, batches, curr_loss / batches)
            if self.verbose == 1:
                print 'Epoch %s Average Loss:' % str(e), curr_loss / batches

    def test(self):
        """
        Performs tests on validation data.
        """

        y = self.session.run(self.inference_logits, feed_dict={self.X: self.test_enc,
                                                    self.X_len: self.encoder_lengths_test,
                                                        self.Y: self.test_dec})


        print y.shape()
