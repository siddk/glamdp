"""
segmenter_rnn.py 

Core class defining the two-layer Segmenter RNN Network.
"""
import numpy as np
import tensorflow as tf

PAD, PAD_ID = "<<PAD>>", 0
UNK, UNK_ID = "<<UNK>>", 1

class Segmenter():
    def __init__(self, segmented_path, embedding_size=30):
        """
        Initialize a Segmenter, with the given training data.
        """
        self.segmented_path = segmented_path
        self.embed_sz = embedding_size
        self.trX, self.trY, self.tsX, self.tsY, self.word2id = self.parse()

        # Setup Placeholders
        self.X = tf.placeholder(tf.int64, shape=[None, self.trX.shape[-1]])
        self.Y = tf.placeholder(tf.int64, shape=[None, self.trY.shape[-1]])
        self.keep_prob = tf.placeholder(tf.float32)

        # Build Inference Graph
        self.logits = self.inference()

    def parse(self):
        """
        Load, LOWERCASE, and Parse Training Data
        """
        with open(self.segmented_path, 'r') as f:
            lines = f.readlines()
        
        sentence_words, sentence_boundaries = [], []
        for line in lines:
            segments, words, boundaries = line.strip().split('|'), [], []
            for segment in segments:
                segment_words = segment.split()
                for w in segment_words[:-1]:
                    words.append(w.lower())
                    boundaries.append(0)
                words.append(segment_words[-1])
                boundaries.append(1)
            sentence_words.append(words)
            sentence_boundaries.append(boundaries)
        
        vocab = [PAD, UNK_ID] + list(set(reduce(lambda x, y: x + y, segment_words)))
        word2id = {w: i for i, w in enumerate(vocab)}

        max_len = max(map(lambda x: len(x), sentence_words))

        train_len = int(len(sentence_words) * .9)
        test_len = len(sentence_words) - train_len
        
        trainX, trainY = np.zeros([train_len, max_len], dtype=np.int32), np.zeros([train_len, max_len], dtype=np.int32)
        testX, testY = np.zeros([test_len, max_len], dtype=np.int32), np.zeros([test_len, max_len], dtype=np.int32)
        
        for i in range(train_len):
            for j in range(len(sentence_words[i])):
                trainX[i][j] = word2id[sentence_words[i][j]]
                trainY[i][j] = sentence_boundaries[i][j]
        
        for i in range(test_len):
            for j in range(len(sentence_words[train_len + i])):
                testX[i][j] = word2id[sentence_words[train_len + i][j]]
                testY[i][j] = sentence_boundaries[train_len + i][j]
        
        return trainX, trainY, testX, testY, word2id
    
    def inference(self):
        # Embedding
        E = tf.get_variable("Embedding", shape=[len(self.word2id), self.embedding_sz],
                            dtype=tf.float32, initializer=self.init)
        embedding = tf.nn.embedding_lookup(E, self.X)               # Shape [None, max_len, embed_sz]
        embedding = tf.nn.dropout(embedding, self.keep_prob)
