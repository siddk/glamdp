"""
cartpole_a2c.py

Implementation of A2C for Cartpole-V0 => OpenAI Gym Backend.
"""
from collections import namedtuple
from keras.layers import Dense
import gym
import numpy as np
import tensorflow as tf

Replay = namedtuple("Replay", ["state", "action", "reward", "next_state", "done"])


class CartPoleA2C:
    def __init__(self, env, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, render=True):
        """
        Build an CartPole A2C Agent, given the OpenAI Gym environment.

        :param env: Environment to build A2C Agent on.
        """
        self.env, self.action_space, self.state_space = env, env.action_space.n, env.observation_space.shape[0]
        self.actor_lr, self.critic_lr, self.gamma, self.render = actor_lr, critic_lr, gamma, render
        self.session = tf.Session()

        # Setup Placeholders
        self.state = tf.placeholder(tf.float32, shape=[self.state_space], name='State')
        self.action_taken = tf.placeholder(tf.int32, name='Action')
        self.target = tf.placeholder(tf.float32, name='Target')

        # Build Actor (Policy Estimator)
        self.build_actor()

        # Build Critic (Value Estimator)
        self.build_critic()

        # Initialize all Variables
        self.session.run(tf.global_variables_initializer())

    def build_actor(self):
        # Compute Policy given State
        hidden = Dense(32, activation='relu')(tf.expand_dims(self.state, axis=0))
        self.policy = tf.squeeze(Dense(self.action_space, activation='softmax')(hidden))

        # Build Loss
        self.p_action_taken = tf.gather(self.policy, self.action_taken)
        self.actor_loss = -tf.log(self.p_action_taken) * self.target

        # Training Operation
        self.actor_op = tf.train.AdamOptimizer(learning_rate=self.actor_lr).minimize(self.actor_loss)

    def predict_action(self, state):
        policy = self.session.run(self.policy, feed_dict={self.state: state})
        return np.random.choice(self.action_space, p=policy)

    def actor_update(self, state, action, target):
        loss, _ = self.session.run([self.actor_loss, self.actor_op], feed_dict={self.state: state,
                                                                                self.action_taken: action,
                                                                                self.target: target})
        return loss

    def build_critic(self):
        # Compute Value Function Estimate given State
        hidden = Dense(32, activation='relu')(tf.expand_dims(self.state, axis=0))
        self.value = tf.squeeze(Dense(1, activation='linear')(hidden))

        # Build Loss (Minimize MSE between Predicted Value and "True" Reward)
        self.critic_loss = tf.squared_difference(self.value, self.target)

        # Training Operation
        self.critic_op = tf.train.AdamOptimizer(learning_rate=self.critic_lr).minimize(self.critic_loss)

    def predict_value(self, state):
        return self.session.run(self.value, feed_dict={self.state: state})

    def critic_update(self, state, target):
        loss, _ = self.session.run([self.critic_loss, self.critic_op], feed_dict={self.state: state,
                                                                                  self.target: target})
        return loss

    def train(self, episodes):
        for e in range(episodes):
            # Reset Environment, collect First State, set Score to 0
            state, done, score, episode = self.env.reset(), False, 0, []

            # Collect Experience
            while not done:
                # Render (if applicable)
                if self.render:
                    self.env.render()

                # Take Action, Collect Observation from Environment
                action = self.predict_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # Compute Reward => -100 if Episode is over
                reward = reward if not done or score == 499 else -100

                # Update Score
                score += reward

                # Update Episode Replay
                episode.append(Replay(state, action, reward, next_state, done))

                # Update State
                state = next_state

            # Update Parameters
            for t, replay in enumerate(episode):
                # Compute Total Reward observed after time t
                forward_return = sum(self.gamma ** i * r.reward for i, r in enumerate(episode[t:]))

                # Calculate Advantage
                value = self.predict_value(replay.state)
                advantage = forward_return - value

                # Update Policy Estimator (Actor)
                self.actor_update(replay.state, replay.action, advantage)

                # Update Value Function Estimator
                self.critic_update(replay.state, forward_return)

            # Print Episode Statistics
            print 'Episode %d\tTotal Steps: %d\tTotal Reward: %d' % (e + 1, len(episode), score)


if __name__ == "__main__":
    # Build Environment, Get State Size, Action Size
    environment = gym.make('CartPole-v1')

    # Build A2C Agent
    agent = CartPoleA2C(environment)

    # Train for 5000 Episodes
    agent.train(5000)
