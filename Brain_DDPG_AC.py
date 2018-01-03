import tensorflow as tf
import numpy as np

learning_rate_Actor = 0.02    # learning rate for actor
learning_rate_Citic = 0.02    # learning rate for critic
Discount_of_reward = 0.9     # reward discount
Soft_replace_factor = 0.01      # soft replacement
Storage_samples = 5000
Batch_size = 1024
Epsilon=0.85       # probability of applying greed-algorithm


class brain(object):
    def __init__(self, action_N, state_N, action_limit):
        self.memory = np.zeros((Storage_samples, state_N * 2 + action_N + 1), dtype=np.float32)
        self.is_memory_full = False
        self.pointer = 0
        self.Actor_replace_counter = 0
        self.Citic_replace_counter = 0
        self.sess = tf.Session()

        self.action_N, self.state_N = action_N, state_N
        self.state = tf.placeholder(tf.float32, [None, state_N], 'state_before')
        self.state_ = tf.placeholder(tf.float32, [None, state_N], 'state_after')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')

        with tf.variable_scope('Actor'):
            self.action = self.Actor_network(self.state, scope='evaluation', trainable=True)
            action_target = self.Actor_network(self.state_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.action = action in memory when calculating Q for td_error,
            # otherwise the self.action is from Actor when updating Actor
            Q_value = self.Citic_network(self.state, self.action, scope='evaluation', trainable=True)
            Q_value_target = self.Citic_network(self.state_, action_target, scope='target', trainable=False)

        # networks parameters
        self.Actor_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/evaluation')
        self.Actor_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.Citic_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/evaluation')
        self.Citic_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(actor_tar, (1 - Soft_replace_factor) * actor_tar + Soft_replace_factor * actor_eval),
                              tf.assign(citic_tar, (1 - Soft_replace_factor) * citic_tar + Soft_replace_factor * citic_eval)]
                             for actor_tar, actor_eval, citic_tar, citic_eval in zip(self.Actor_target_params, self.Actor_eval_params, self.Citic_target_params, self.Citic_eval_params)]

        self.hard_replace=[[tf.assign(actor_tar, actor_eval), tf.assign(citic_tar, citic_eval)]
                      for actor_tar, actor_eval, citic_tar, citic_eval in zip(self.Actor_target_params, self.Actor_eval_params, self.Citic_target_params, self.Citic_eval_params)]

        Q_target = self.reward + Discount_of_reward * Q_value_target
        # in the feed_dic for the td_error, the self.action should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=Q_target, predictions=Q_value)
        self.Citic_train = tf.train.AdamOptimizer(learning_rate_Citic).minimize(td_error, var_list=self.Citic_eval_params)

        action_loss = - tf.reduce_mean(Q_value)    # maximize the Q
        self.Actor_train = tf.train.AdamOptimizer(learning_rate_Actor).minimize(action_loss, var_list=self.Actor_eval_params)

        self.sess.run(tf.global_variables_initializer())

    def Actor_network(self, state, scope, trainable):
        with tf.variable_scope(scope):
            hidden_layer_Actor = tf.layers.dense(state, 32, activation=tf.nn.relu, name='hidden_layer_Actor', trainable=trainable)
            action_Actor = tf.layers.dense(hidden_layer_Actor, self.action_N, activation=tf.nn.tanh, name='action_Actor', trainable=trainable)
            return action_Actor

    def Citic_network(self, state, action, scope, trainable):
        with tf.variable_scope(scope):
            neuron_N = 32
            state_weight_Citic = tf.get_variable('state_weight_Citic', [self.state_N, neuron_N], trainable=trainable)
            action_weight_Citic = tf.get_variable('action_weight_Citic', [self.action_N, neuron_N], trainable=trainable)
            bias_Citic = tf.get_variable('bias_Citic', [1, neuron_N], trainable=trainable)
            eval_Citic = tf.nn.relu(tf.matmul(state, state_weight_Citic) + tf.matmul(action, action_weight_Citic) + bias_Citic)
            return tf.layers.dense(eval_Citic, self.action_N, trainable=trainable)  # Q(s,a)

    def choose_action(self, state):
        if np.random.uniform() < Epsilon:
            # apply greedy algorithm
            action=self.sess.run(self.action, {self.state: state[None, :]})[0]
        else:
            action=2.*np.random.rand(1)*np.pi    # sample action
            #action=action[0]
        return action

    def learning(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(Storage_samples, size=Batch_size)
        batch = self.memory[indices, :]
        batch_state = batch[:, :self.state_N]
        batch_action = batch[:, self.state_N: self.state_N + self.action_N]
        batch_reward = batch[:, -self.state_N - 1: -self.state_N]
        batch_state_ = batch[:, -self.state_N:]

        self.sess.run(self.Actor_train, {self.state: batch_state})
        self.sess.run(self.Citic_train, {self.state: batch_state, self.action: batch_action, self.reward: batch_reward, self.state_: batch_state_})

    def store(self, state, action, reward, state_):
        state = np.array(state).flatten()
        action = np.array(action).flatten()
        reward = np.array(reward).flatten()
        state_ = np.array(state_).flatten()
        transition = np.hstack((state, action, reward, state_))
        index = self.pointer % Storage_samples  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > Storage_samples:      # indicator for learning
            self.is_memory_full = True


    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './params')