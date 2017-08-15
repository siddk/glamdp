"""
reinforced_multi_draggn.py
"""
from itertools import chain
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import time


class ReinforcedMultiDRAGGN:
    def __init__(self, trainX, trainX_len, trainY, word2id, programs, arguments, embed_sz=30, rnn_sz=128,
                 submodule_sz=64, bsz=32, init=tf.truncated_normal_initializer(stddev=0.1),
                 gamma=0.99, lambda_=1.0, critic_discount=0.5, entropy_discount=0.0001):
        """
        Instantiate ReinforcedDRAGGN Model with necessary hyperparameters.
        """
        self.trainX, self.trainX_len, self.trainY = trainX, trainX_len, trainY
        self.word2id, self.programs, self.arguments = word2id, programs, arguments
        self.embed_sz, self.rnn_sz, self.submodule_sz, self.bsz, self.init = embed_sz, rnn_sz, submodule_sz, bsz, init
        self.num_programs, self.num_arguments = len(self.programs), len(self.arguments)
        self.gamma, self.lambda_, self.vf_, self.ent_ = gamma, lambda_, critic_discount, entropy_discount
        self.session = tf.Session()

        # Setup Placeholders
        self.X = tf.placeholder(tf.int64, shape=[self.bsz, trainX.shape[1]], name='Utterance')
        self.X_len = tf.placeholder(tf.int64, shape=[self.bsz], name='Utterance_Length')
        self.action = tf.placeholder(tf.float32, shape=[self.bsz, self.num_programs * self.num_arguments], name='Act')
        self.rewards = tf.placeholder(tf.float32, shape=[self.bsz, 1], name='Reward')
        self.advantage = tf.placeholder(tf.float32, shape=[self.bsz, 1], name='Advantage')
        self.keep_prob = tf.placeholder(tf.float32, name='Dropout_Probability')

        # Compute Policy, Value Estimate
        self.program_policy, self.argument_policy, self.value = self.forward()

        # Compute Actual Policy by Taking Outer Product times Valid Mask
        program_splits = tf.split(self.program_policy, self.bsz, axis=0)
        argument_splits = tf.split(self.argument_policy, self.bsz, axis=0)
        policy_split = tf.stack([tf.transpose(p) * a for p, a, in zip(program_splits, argument_splits)])
        self.policy = tf.reshape(policy_split, shape=[self.bsz, self.num_programs * self.num_arguments])

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
        state.set_shape([self.bsz, self.rnn_sz])

        # Feed-Forward Layers
        hidden = Dense(self.rnn_sz, activation='relu')(state)
        hidden = tf.nn.dropout(hidden, self.keep_prob)

        # Program Policy Submodule
        p_prog_hidden = Dense(self.submodule_sz, activation='relu')(hidden)
        prog_policy = Dense(self.num_programs, activation='softmax')(p_prog_hidden)

        # Argument Policy Submodule
        p_arg_hidden = Dense(self.submodule_sz, activation='relu')(hidden)
        arg_policy = Dense(self.num_arguments, activation='softmax')(p_arg_hidden)

        # Value Submodule
        v_hidden = Dense(self.submodule_sz, activation='relu')(hidden)
        value = Dense(1, activation='linear')(v_hidden)

        return prog_policy, arg_policy, value

    def loss(self):
        # Policy Gradient (Actor) Loss
        actor_loss = -tf.reduce_sum(tf.log(self.policy) * self.action * self.advantage)

        # Value Function (Critic) Loss
        critic_loss = 0.5 * tf.reduce_sum(tf.square((self.value - self.rewards)))

        return actor_loss + self.vf_ * critic_loss

    def predict(self, state, state_len, dropout=1.0):
        return self.session.run([self.policy, self.value], feed_dict={self.X: state, self.X_len: state_len,
                                                                      self.keep_prob: dropout})

    def act(self, policies):
        return [np.random.choice(self.num_programs * self.num_arguments, p=p) for p in policies]

    def train_step(self, env_xs, env_xs_len, env_as, env_rs, env_vs):
        # Flatten Observations into 2D Tensor
        xs = np.vstack(list(chain.from_iterable(env_xs)))
        xs_len = np.vstack(list(chain.from_iterable(env_xs_len))).squeeze()

        # One-Hot Actions
        as_ = np.zeros((len(xs), self.num_programs * self.num_arguments))
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
        self.session.run(self.train_op, feed_dict={self.X: xs, self.X_len: xs_len, self.action: as_,
                                                   self.rewards: drs, self.advantage: advs, self.keep_prob: 1.0})

    @staticmethod
    def _discount(x, gamma):
        return [sum(gamma ** i * r for i, r in enumerate(x[t:])) for t in range(len(x))]

    def fit(self, episodes, validate=False):
        start, running_reward = time.time(), None
        for e in range(episodes):
            tic = time.time()
            env_xs, env_as = [[] for _ in range(self.bsz)], [[] for _ in range(self.bsz)]
            env_xs_len = [[] for _ in range(self.bsz)]
            env_rs, env_vs = [[] for _ in range(self.bsz)], [[] for _ in range(self.bsz)]
            episode_rs = np.zeros(self.bsz, dtype=np.float)

            # Get Observations from all Environments (bsz)
            idx = np.random.choice(len(self.trainX), size=self.bsz)
            step_xs, step_xs_len = self.trainX[idx], self.trainX_len[idx]

            # Get Policies/Actions and Values for all Environments in Single Pass
            step_ps, step_vs = self.predict(step_xs, step_xs_len)
            step_as = self.act(step_ps)

            # Perform Action in Every Environment
            for i in range(self.bsz):
                # Compute Reward (Equality Check with Actual Program/Argument Label)
                program, argument = step_as[i] / self.num_programs, step_as[i] % self.num_programs
                r = 1.0 if (program == self.trainY[idx[i]][0]) and (argument == self.trainY[idx[i]][1]) else 0

                # Record the observation, action, value, and reward in the buffers.
                env_xs[i].append(step_xs[i])
                env_xs_len[i].append(step_xs_len[i])
                env_as[i].append(step_as[i])
                env_vs[i].append(step_vs[i][0])
                env_rs[i].append(r)
                episode_rs[i] += r

                # Add 0 as State Value (because Done!)
                env_vs[i].append(0.0)

            # Perform Train Step
            self.train_step(env_xs, env_xs_len, env_as, env_rs, env_vs)

            # Print Statistics
            for er in episode_rs:
                running_reward = er if running_reward is None else (0.99 * running_reward + 0.01 * er)

            if e % 10 == 0:
                print 'Batch %d complete (%.2fs) (%.1fs elapsed) (episode %d), batch total reward: %.2f, running reward: %.3f' % (e, time.time() - tic, time.time() - start, (e + 1) * self.bsz, sum(episode_rs), running_reward)

    def eval(self, testX, testX_len, testY, validate=False):
        correct, total = 0, 0
        for start, end in zip(range(0, len(testX) - self.bsz, self.bsz), range(self.bsz, len(testX), self.bsz)):
            # Compute Policy
            step_ps, _ = self.predict(testX[start:end], testX_len[start:end])
            step_as = np.argmax(step_ps, axis=1)

            # Get Programs, Arguments
            step_progs, step_args = step_as / self.num_programs, step_as % self.num_programs
            prog_corr, arg_corr = step_progs == testY[start:end, 0], step_args == testY[start:end, 1]
            joint_corr = (prog_corr * arg_corr).astype(int)
            correct, total = correct + sum(joint_corr), total + len(joint_corr)

        # Compute Final
        step_ps, _ = self.predict(testX[-self.bsz:], testX_len[-self.bsz:])
        step_as = np.argmax(step_ps, axis=1)[-(len(testX) - end):]
        step_progs, step_args = step_as / self.num_programs, step_as % self.num_programs
        prog_corr, arg_corr = step_progs == testY[end:, 0], step_args == testY[end:, 1]
        joint_corr = (prog_corr * arg_corr).astype(int)
        correct, total = correct + sum(joint_corr), total + len(joint_corr)

        # Accuracy
        print 'Test Accuracy: %.3f' % (float(correct) / float(total))