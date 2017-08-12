"""
parallel_a3c.py

Implementation of Parallel A3C with GAE for CartPole-v1.
"""
from itertools import chain
from keras.layers import Dense
import gym
import numpy as np
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("seed", 21, "Random seed, for reproducibility.")
tf.app.flags.DEFINE_integer("num_environments", 4, "Number of environments to run in parallel.")
tf.app.flags.DEFINE_integer("episodes", 1000, "Number of episodes to train for.")

tf.app.flags.DEFINE_integer("t_max", 50, "Number of time steps before performing parameter update.")


class A3C:
    def __init__(self, state_size, num_actions, hidden_sz=256, gamma=0.99, lambda_=1.0, critic_discount=0.5,
                 entropy_discount=0.01):
        """
        Initialize an GAE A3C Agent with the given state size and number of discrete actions.

        :param state_size: Size of the state space of the given environment.
        :param num_actions: Number of discrete actions for the agent to choose from.
        :param hidden_sz: Number of units in actor-critic hidden layer.
        :param gamma: Discount factor, for computing future reward.
        :param lambda_: Parameter for GAE, controls bias-variance trade-off.
        :param critic_discount: Discount for Value Function (Critic) Term in Loss.
        :param entropy_discount: Discount for Entropy Term in Loss.
        """
        self.state_size, self.num_actions, self.hidden_sz = state_size, num_actions, hidden_sz
        self.gamma, self.lambda_, self.vf_, self.ent_ = gamma, lambda_, critic_discount, entropy_discount
        self.session = tf.Session()

        # Setup Placeholders
        self.state = tf.placeholder(tf.float32, shape=[None, self.state_size], name='State')
        self.action = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='Action')
        self.rewards = tf.placeholder(tf.float32, shape=[None, 1], name='Reward')
        self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name='Advantage')

        # Compute Policy, Value Estimate
        self.policy, self.value = self.forward()

        # Compute Actor-Critic Loss
        self.loss_val = self.loss()

        # Build Training Operation
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_val)

        # Initialize all Variables
        self.session.run(tf.global_variables_initializer())

    def forward(self):
        # Encode State
        state_encoding = Dense(self.hidden_sz, activation='relu')(self.state)

        # Compute Policy
        policy = Dense(self.num_actions, activation='softmax')(state_encoding)

        # Compute Value
        value = Dense(1, activation='linear')(state_encoding)
        return policy, value

    def loss(self):
        # Policy Gradient (Actor) Loss
        actor_loss = -tf.reduce_sum(tf.log(self.policy) * self.action * self.advantage)

        # Value Function (Critic) Loss
        critic_loss = 0.5 * tf.reduce_sum(tf.square((self.value - self.rewards)))

        # Entropy Term (to encourage exploration)
        entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))

        return actor_loss + self.vf_ * critic_loss + self.ent_ * entropy

    def predict(self, state):
        return self.session.run([self.policy, self.value], feed_dict={self.state: state})

    def act(self, policies):
        return [np.random.choice(self.num_actions, p=p) for p in policies]

    def train_step(self, env_xs, env_as, env_rs, env_vs):
        # Flatten Observations into 2D Tensor
        xs = np.vstack(list(chain.from_iterable(env_xs)))

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
        self.session.run(self.train_op, feed_dict={self.state: xs, self.action: as_, self.rewards: drs,
                                                   self.advantage: advs})

    @staticmethod
    def _discount(x, gamma):
        return [sum(gamma ** i * r for i, r in enumerate(x[t:])) for t in range(len(x))]


def train_step(envs, agent):
    env_xs, env_as = [[] for _ in range(len(envs))], [[] for _ in range(len(envs))]
    env_rs, env_vs = [[] for _ in range(len(envs))], [[] for _ in range(len(envs))]
    episode_rs = np.zeros(len(envs), dtype=np.float)

    # Get Observations from all Environments
    observations = [env.reset() for env in envs]
    done, all_done, t = np.array([False for _ in range(len(envs))]), False, 1

    # Run Episode Loop
    while not all_done:
        # Stack all Observations into a Single Matrix
        step_xs = np.vstack(observations)

        # Get Policies/Actions and Values for all Environments in Single Pass
        step_ps, step_vs = agent.predict(step_xs)
        step_as = agent.act(step_ps)

        # Perform Action in every Environment, Update Observations
        for i, env in enumerate(envs):
            if not done[i]:
                env.render()
                obs, r, done[i], _ = env.step(step_as[i])

                # Record the observation, action, value, and reward in the buffers.
                env_xs[i].append(step_xs[i])
                env_as[i].append(step_as[i])
                env_vs[i].append(step_vs[i][0])
                env_rs[i].append(r)
                episode_rs[i] += r

                # Add 0 as the state value when done.
                if done[i]:
                    env_vs[i].append(0.0)
                else:
                    observations[i] = obs

        # Perform Update every T-MAX Steps (as per A3C Paper)
        if t == FLAGS.t_max:
            # If episode not done, bootstrap final return with estimated value for current state.
            _, extra_vs = agent.predict(np.vstack(observations).reshape(len(envs), -1))
            for i in range(len(envs)):
                if not done[i]:
                    env_vs[i].append(extra_vs[i][0])

            # Perform Parameter Update
            agent.train_step(env_xs, env_as, env_rs, env_vs)

            # Clear Buffers
            env_xs, env_as = [[] for _ in range(len(envs))], [[] for _ in range(len(envs))]
            env_rs, env_vs = [[] for _ in range(len(envs))], [[] for _ in range(len(envs))]
            t = 0

        all_done = np.all(done)
        t += 1

    # Perform a final update when all episodes are finished.
    if len(env_xs[0]) > 0:
        agent.train_step(env_xs, env_as, env_rs, env_vs)

    return episode_rs


def main(_):
    # Create Parallel Environments
    envs = [gym.make('CartPole-v1') for _ in range(FLAGS.num_environments)]
    for i, env in enumerate(envs):
        env.seed(i + FLAGS.seed)

    # Create A3C Agent
    agent = A3C(envs[0].observation_space.shape[0], envs[0].action_space.n)

    # Train
    start, running_reward = time.time(), None
    for e in range(FLAGS.episodes):
        tic = time.time()
        episode_rs = train_step(envs, agent)

        for er in episode_rs:
            running_reward = er if running_reward is None else (0.99 * running_reward + 0.01 * er)

        if e % 10 == 0:
            print 'Batch %d complete (%.2fs) (%.1fs elapsed) (episode %d), batch avg. reward: %.2f, running reward: %.3f' % (e, time.time() - tic, time.time() - start, (e + 1) * FLAGS.num_environments, np.mean(episode_rs), running_reward)

if __name__ == "__main__":
    tf.app.run()
