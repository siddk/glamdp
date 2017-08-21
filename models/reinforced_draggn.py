"""
reinforced_draggn.py
"""
from itertools import chain
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import time


class ReinforcedDRAGGN:
    def __init__(self, trainX, trainX_len, trainX_state, trainY, word2id, labels, embed_sz=30, rnn_sz=128,
                 submodule_sz=64, bsz=32, init=tf.truncated_normal_initializer(stddev=0.1),
                 gamma=0.99, lambda_=1.0, critic_discount=0.5, entropy_discount=0.0001):
        """
        Instantiate ReinforcedDRAGGN Model with necessary hyperparameters.
        """
        self.trainX, self.trainX_len, self.trainX_state, self.trainY = trainX, trainX_len, trainX_state, trainY
        self.word2id, self.labels = word2id, labels
        self.embed_sz, self.rnn_sz, self.submodule_sz, self.bsz, self.init = embed_sz, rnn_sz, submodule_sz, bsz, init
        self.num_actions = len(self.labels)
        self.gamma, self.lambda_, self.vf_, self.ent_ = gamma, lambda_, critic_discount, entropy_discount
        self.session = tf.Session()

        # Setup Placeholders
        self.X = tf.placeholder(tf.int64, shape=[None, trainX.shape[1]], name='Utterance')
        self.X_len = tf.placeholder(tf.int64, shape=[None], name='Utterance_Length')
        self.X_state = tf.placeholder(tf.float32, shape=[None, trainX_state.shape[1]])
        self.action = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='Action')
        self.rewards = tf.placeholder(tf.float32, shape=[None, 1], name='Reward')
        self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name='Advantage')
        self.keep_prob = tf.placeholder(tf.float32, name='Dropout_Probability')

        # Compute Policy, Value Estimate
        self.policy, self.value = self.forward()

        # Compute Actor-Critic Loss
        self.loss_val = self.loss()

        # Build Training Operation
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_val)

        # Initialize all Variables
        self.session.run(tf.global_variables_initializer())

    def forward(self):
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

        # Encode State with Feed-Forward Layer
        state_encoding = Dense(32)(self.X_state)

        # Concatenate state + state_encoding
        state = tf.concat([state, state_encoding], axis=1)

        # Feed-Forward Layers
        hidden = Dense(self.rnn_sz, activation='relu')(state)
        hidden = tf.nn.dropout(hidden, self.keep_prob)

        # Policy Submodule
        p_hidden = Dense(self.submodule_sz, activation='relu')(hidden)
        policy = Dense(self.num_actions, activation='softmax')(p_hidden)

        # Value Submodule
        v_hidden = Dense(self.submodule_sz, activation='relu')(hidden)
        value = Dense(1, activation='linear')(v_hidden)

        return policy, value

    def loss(self):
        # Policy Gradient (Actor) Loss
        actor_loss = -tf.reduce_sum(tf.log(self.policy) * self.action * self.advantage)

        # Value Function (Critic) Loss
        critic_loss = 0.5 * tf.reduce_sum(tf.square((self.value - self.rewards)))

        # Entropy Term (to encourage exploration)
        entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))

        return actor_loss + self.vf_ * critic_loss + self.ent_ * entropy

    def predict(self, state, state_len, initial_state, dropout=1.0):
        return self.session.run([self.policy, self.value], feed_dict={self.X: state, self.X_len: state_len,
                                                                      self.X_state: initial_state,
                                                                      self.keep_prob: dropout})

    def act(self, policies):
        return [np.random.choice(self.num_actions, p=p) for p in policies]

    def train_step(self, env_xs, env_xs_len, env_xs_state, env_as, env_rs, env_vs):
        # Flatten Observations into 2D Tensor
        xs = np.vstack(list(chain.from_iterable(env_xs)))
        xs_len = np.vstack(list(chain.from_iterable(env_xs_len))).squeeze()
        xs_state = np.vstack(list(chain.from_iterable(env_xs_state))).squeeze()

        # One-Hot Actions
        as_ = np.zeros((len(xs), self.num_actions))
        as_[np.arange(len(xs)), list(chain.from_iterable(env_as))] = 1

        # Compute Discounted Rewards + Advantages
        drs, advs = [], []
        for i in range(len(env_vs)):
            # Compute discounted rewards with a 'bootstrapped' final value.
            rs_bootstrap = [] if env_rs[i] == [] else env_rs[i] + [env_vs[i][-1]]
            drs.extend(self._discount(rs_bootstrap, self.gamma)[:-1])

            # Compute advantages via GAE - Schulman et. al 2016
            delta_t = env_rs[i] + self.gamma * np.array(env_vs[i][1:]) - np.array(env_vs[i][:-1])  # (Eq 11)
            advs.extend(self._discount(delta_t, self.gamma * self.lambda_))                        # (Eq 16)

        # Expand drs, advs to be 2D Tensors
        drs, advs = np.array(drs)[:, np.newaxis], np.array(advs)[:, np.newaxis]

        # Perform Training Update
        self.session.run(self.train_op, feed_dict={self.X: xs, self.X_len: xs_len, self.X_state: xs_state,
                                                   self.action: as_, self.rewards: drs, self.advantage: advs,
                                                   self.keep_prob: 1.0})

    @staticmethod
    def _discount(x, gamma):
        return [sum(gamma ** i * r for i, r in enumerate(x[t:])) for t in range(len(x))]

    def fit(self, episodes, validate=False):
        start, running_reward = time.time(), None
        for e in range(episodes):
            tic = time.time()
            env_xs, env_as = [[] for _ in range(self.bsz)], [[] for _ in range(self.bsz)]
            env_xs_len, env_xs_state = [[] for _ in range(self.bsz)], [[] for _ in range(self.bsz)]
            env_rs, env_vs = [[] for _ in range(self.bsz)], [[] for _ in range(self.bsz)]
            episode_rs = np.zeros(self.bsz, dtype=np.float)

            # Get Observations from all Environments (bsz)
            idx = np.random.choice(len(self.trainX), size=self.bsz)
            step_xs, step_xs_len, step_xs_state = self.trainX[idx], self.trainX_len[idx], self.trainX_state[idx]

            # Get Policies/Actions and Values for all Environments in Single Pass
            step_ps, step_vs = self.predict(step_xs, step_xs_len, step_xs_state)
            step_as = self.act(step_ps)

            # Perform Action in Every Environment
            for i in range(self.bsz):
                if validate:
                    # Compute Reward (Validation Function)
                    r = 1 if step_as[i] in self.trainY[idx[i]] else 0
                else:
                    # Compute Reward (Equality Check with Actual Label)
                    r = 1 if step_as[i] == self.trainY[idx[i]] else 0

                # Record the observation, action, value, and reward in the buffers.
                env_xs[i].append(step_xs[i])
                env_xs_len[i].append(step_xs_len[i])
                env_xs_state[i].append(step_xs_state[i])
                env_as[i].append(step_as[i])
                env_vs[i].append(step_vs[i][0])
                env_rs[i].append(r)
                episode_rs[i] += r

                # Add 0 as State Value (because Done!)
                env_vs[i].append(0.0)

            # Perform Train Step
            self.train_step(env_xs, env_xs_len, env_xs_state, env_as, env_rs, env_vs)

            # Print Statistics
            for er in episode_rs:
                running_reward = er if running_reward is None else (0.99 * running_reward + 0.01 * er)

            if e % 10 == 0:
                print 'Batch %d complete (%.2fs) (%.1fs elapsed) (episode %d), batch total reward: %.2f, running reward: %.3f' % (e, time.time() - tic, time.time() - start, (e + 1) * self.bsz, sum(episode_rs), running_reward)

    def eval(self, testX, testX_len, testX_state, testY, validate=False):
        # Compute Policy
        step_ps, _ = self.predict(testX, testX_len, testX_state)
        step_as = np.argmax(step_ps, axis=1)

        # Compute Reward
        if validate:
            r = []
            for i in range(len(step_as)):
                if step_as[i] in testY[i]:
                    r.append(1.0)
                else:
                    r.append(0.0)
        else:
            r = (step_as == testY).astype(int)

        # Accuracy
        print 'Test Accuracy: %.3f' % np.mean(r)