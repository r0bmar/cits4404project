import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np

import gym

# class ProbabilityDistribution(tf.keras.Model):
#     def call(self, logits):
#         return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

# class Actor(tf.keras.Model):
#     def __init__(self, num_actions):
#         super().__init__('mlp_policy')

#         self.h1 = kl.Dense(48, activation='relu')
#         self.h2 = kl.Dense(num_actions, activation='relu', name='policy_logits')
#         self.dist = ProbabilityDistribution()

#     def call(self, inputs):
#         x = tf.convert_to_tensor(inputs, dtype=tf.float32)
#         x = self.h1(x)
#         return self.h2(x)

#     def action_value(self, obs):
#         # executes call() under the hood
#         logits = self.predict(obs)
#         action = self.dist.predict(logits)
#         # a simpler option, will become clear later why we don't use it
#         # action = tf.random.categorical(logits, 1)
#         return np.squeeze(action, axis=-1)

class ActorCritic(object):
    """ActorCritic model for continous observation space, and discrete action space
    
    Arguments:
        object {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    def __init__(self, env, **kwargs):
        self.env=env

        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.gamma = kwargs.get('gamma', 0.99)

        self.actor_model  = self._create_actor_model()
        self.critic_model = self._create_critic_model()

    def _create_actor_model(self):
        model = tf.keras.Sequential([
            kl.Dense(48, activation='relu', input_shape=self.env.observation_space.shape),
            kl.Dense(128, activation='relu'),
            kl.Dense(self.env.action_space.n, activation='relu', name='policy_logits')
        ])
        model.compile(loss=self._actor_loss, optimizer='adam')

        return model

    def _create_critic_model(self):
        model = tf.keras.Sequential([
            kl.Dense(48, activation='relu', input_shape=self.env.observation_space.shape),
            kl.Dense(128, activation='relu'),
            kl.Dense(1, activation='relu')
        ])
        model.compile(loss=self._critic_loss, optimizer='adam')

        return model

    def _actor_loss(self, acts_and_advs, logits):
        return 0.0

    def _critic_loss(self, returns, value):
        return 0.0

    def train(self):
        pass
        # self.actor_model.train_on_batch(observations, acts_and_advs)
        # self.critic_model.train_on_batch(observations, returns)

    def predict_action(self, obs):
        logits = self.actor_model.predict(obs[None, :])
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        return np.squeeze(action, axis=-1)


if __name__=="__main__":
    env = gym.make("CartPole-v1")

    ac = ActorCritic(env)
    obs = env.reset()
    for _ in range(5):
        print(ac.predict_action(obs))


