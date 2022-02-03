import numpy as np
import tensorflow as tf
import time
from spinup.algos.tf1.sac import core2
from spinup.algos.tf1.sac.core2 import get_vars
from spinup.utils.logx import EpochLogger
from sklearn import preprocessing
import pandas as pd
from keras.models import load_model



config = tf.ConfigProto()
#
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
#
session = tf.Session(config=config)
#####################
#Discrete and Continuous  SAC Version
##################

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """


    def __init__(self, obs_dim, act_d_dim, act_c_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_c_buf = np.zeros([size, act_c_dim], dtype=np.float32)
        self.acts_d_buf = np.zeros([size, act_d_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.scalar_data = None

    def store(self, obs, act_c,act_d, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_c_buf[self.ptr] = act_c
        self.acts_d_buf[self.ptr] = act_d
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts_c=self.acts_c_buf[idxs],
                    acts_d=self.acts_d_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def sample_batch_test(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts_c=self.acts_c_buf[idxs],
                    acts_d=self.acts_d_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

        return np.array(batch['obs1']), idxs



    #数据预处理
    def load_from_csv(self, column_dict):
        self.scalar_data = column_dict['c_state'].to_numpy()

    def retransform(self, inputdata):

        array_scalar = preprocessing.StandardScaler().fit(self.scalar_data)
        return array_scalar.inverse_transform(inputdata)


    def reset(self):
        array_scalar = preprocessing.StandardScaler().fit(self.scalar_data)
        state = np.zeros([1, 14], dtype=np.float32)
        low_threshold = [[15.0, 9.0, 40.0, 10, 105.0, 95.0, 110.0, 90.0, 7.39, 0.9, 36.9, 24.5, 68.0, 1536163200]]
        high_threhold = [[25.0, 11.0, 43.0, 29.0, 119.0, 100, 120.0, 100.0, 7.40, 1.2, 37.1, 25.5, 71.5, 5099990400]]
        low_threshold = array_scalar.transform(low_threshold)
        high_threhold = array_scalar.transform(high_threhold)

        for i in range(14):
            state[:, i] = np.random.uniform(low=low_threshold[:,i], high=high_threhold[:,i])

        return state

def action_sample():
    a_c = np.zeros(5, dtype=np.float32)
    low_threshold = [0, 0, 0, 0, 0]
    high_thrshold = [100.0, 54.0, 100.0, 883.0, 100.0]

    for i in range(5):
        a_c[i] = np.random.uniform(low=low_threshold[i], high=high_thrshold[ i])

    a_c = a_c/1000
    a = np.random.randint(low=0, high=32)
    a_d =np.zeros(32)
    a_d[a] = 1
    return a_c, a_d

def sac(actor_critic=core2.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=1000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.9, lr=1e-4, alpha_c=0.1,alpha_d=0.1, batch_size=100, start_steps=2500,
        update_after=250, update_every=25, num_test_episodes=5, max_ep_len=100,
        logger_kwargs=dict(), save_freq=1):
    """0
    Soft Actor-Critic (SAC)  replay_size=int(1e6)


    Args:

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.


        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)  #熵

        batch_size (int): Minibatch size for SGD.（随机梯度下降）

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    #读取数据的格式
    tf.set_random_seed(seed)
    np.random.seed(seed)

    train_data = pd.read_csv("D:\\spinningup-master\\data_reshape.CSV",low_memory=False)
    c_state_column = ['STATE' + str(i) for i in range(14)]
    print(c_state_column)
    c_next_state_column = ['NEXT_STATE' + str(i) for i in range(14)]

    reward_column = ['REWARD']
    action_c_column = ['ACTION' + str(i)  for i in range(5)]
    action_d_column = ['ACTION5']

    column_dict = {'c_state': train_data[c_state_column],
                   'c_next_state': train_data[c_next_state_column],
                   'reward': train_data[reward_column],
                   'action_c': train_data[action_c_column],
                   'action_d':train_data[action_d_column]}


    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    c_obs_dim = len(c_state_column)
    act_c_dim = len(action_c_column)         #输出的action 的维度
    act_d_dim = len(train_data['ACTION5'].unique())

    #repla_buffer
    replay_buffer = ReplayBuffer(obs_dim=c_obs_dim, act_c_dim=act_c_dim, act_d_dim = act_d_dim, size=replay_size)
    replay_buffer.load_from_csv(column_dict)


    x_ph,a_c_ph, a_d_ph,x2_ph, r_ph, d_ph = core2.placeholders(c_obs_dim, act_c_dim, act_d_dim, c_obs_dim, None, None)


    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, mu_d, pi_d, logp_pi_d, q1_a, q2_a = actor_critic(x_ph, a_c_ph, a_d_ph, **ac_kwargs)

    with tf.variable_scope('main', reuse=True):
        # compose q with pi, for pi-learning
        _, _, _, _, _, _, q1_pi,q2_pi  = actor_critic(x_ph, pi, pi_d, **ac_kwargs)

        # get actions and log probs of actions for next states, for Q-learning
        _, pi_next, logp_pi_next, _, pi_d_next, logp_pi_d_next, _,_ = actor_critic(x2_ph, a_c_ph, a_d_ph, **ac_kwargs)

    with tf.variable_scope('target'):
        _, _, _, _, _, _, q1_pi_targ, q2_pi_targ = actor_critic(x2_ph, pi_next, pi_d_next, **ac_kwargs)

    # Count variables
    var_counts = tuple(core2.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

    # Min Double-Q:
    #continue
    min_q_pi = tf.minimum(q1_pi, q2_pi)
    min_q_pi_targ = tf.minimum(q1_pi_targ, q2_pi_targ)

    pi_loss = tf.reduce_mean(alpha_c * logp_pi + alpha_d * logp_pi_d - min_q_pi)
    q_backup = r_ph + gamma * (1 - d_ph) *tf.stop_gradient( (min_q_pi_targ - alpha_c * logp_pi_next - alpha_d * logp_pi_d_next))

    q1_loss = 0.5*tf.reduce_mean((q_backup - q1_a)**2)
    q2_loss = 0.5*tf.reduce_mean((q_backup - q2_a)**2)
    value_loss = q1_loss + q2_loss


    # Policy train op
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))


    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q')
    # value_params = get_vars('main/q') + get_vars('vae_encoder')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)



    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, q1_a, q2_a, logp_pi,
                train_pi_op, train_value_op, target_update]
    # loss_ops = [vae_loss]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])


    sess = tf.Session()
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a_c': a_c_ph,'a_d':a_d_ph},
                                outputs={'mu': mu, 'pi': pi, 'pi_d':pi_d, 'q1': q1_a, 'q2': q2_a})

    def get_action(o, deterministic = False):
        act_op = mu if deterministic else pi


        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    def get_action_d(o):
        act_op = pi_d
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    def test_envstep(model,action, action_d, obser):
        a = np.argmax(action_d,axis=-1)
        b = np.zeros_like(action_d)
        b[a] = 1
        action_d = b
        inputstate = np.concatenate((obser, np.array(action).reshape(1,5) ,np.array(action_d).reshape(1,32)), axis=1)
        inputstate = inputstate.reshape((inputstate.shape[0], 1, inputstate.shape[1]))

        next_state = model.predict(inputstate, verbose=0)
        next_state[:,13] = obser[:,13] #weigtht and age were not changed
        next_state[:, 12] = obser[:, 12]

        next_state_trans = replay_buffer.retransform(next_state)

        done =  next_state_trans[:,0] < 5.0 \
                or next_state_trans[:,0] > 35.0 \
                or next_state_trans[:, 2] < 32.0 \
                or next_state_trans[:, 2] > 50.0 \
                or next_state_trans[:, 3] < 0 \
                or next_state_trans[:, 3] > 39.0 \
                or next_state_trans[:, 5] < 95\
                or next_state_trans[:, 5] > 100 \
                or next_state_trans[:, 6] < 90.0 \
                or next_state_trans[:, 6] > 140.0 \
                or next_state_trans[:, 7] < 50.0 \
                or next_state_trans[:, 7] > 140.0 \
                or next_state_trans[:, 8] < 7.35 \
                or next_state_trans[:, 8] > 7.45 \
                or next_state_trans[:, 9] < 0.5 \
                or next_state_trans[:, 9] > 1.6 \
                or next_state_trans[:, 10] < 36.0 \
                or next_state_trans[:, 10] > 38.0 \
                or next_state_trans[:, 11] < 21.8 \
                or next_state_trans[:, 11] > 26.2 \

        done = bool(done)

        if not done:
            reward = 1.0
        else:
            reward = 0.0

        return next_state,reward, done

    def test_agent(model):
        for j in range(num_test_episodes):
            o = replay_buffer.reset()
            d = False
            ep_ret, ep_len = 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions amax_ep_lent test tim
                o, r, d = test_envstep(model, get_action(o,True), get_action_d(o),o)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o = replay_buffer.reset()
    print(o.shape)
    model = load_model('lstm_model_2.h5')
    ep_ret, ep_len = 0, 0
    total_steps = steps_per_epoch * epochs

    #
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps): #40000
        # print(t)
        if t > start_steps:
            a_c = get_action(o)
            a_d = get_action_d(o)
        else:
            a_c, a_d = action_sample()

        o2, r, d = test_envstep(model, a_c, a_d, o)
        ep_ret += r
        ep_len += 1
        d = False if ep_len==max_ep_len else d
        replay_buffer.store(o, a_c, a_d, r, o2, d)

        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o = replay_buffer.reset()
            ep_ret, ep_len = 0, 0

        # Update handling#更新的部分
        if t >= update_after and  t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_c_ph: batch['acts_c'],
                             a_d_ph: batch['acts_d'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                            }
                outs = sess.run(step_ops, feed_dict)
                logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                             Q1Vals=outs[3], Q2Vals=outs[4], LogPi=outs[5])

        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            test_agent(model)
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet',average_only=True)
            logger.log_tabular('TestEpRet', average_only=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', average_only=True)
            logger.log_tabular('Q2Vals', average_only=True)
            logger.log_tabular('LogPi', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='final_version 0.1(2) 200batch 1e4 100max')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    i = 1
    tf.reset_default_graph()
    sac(actor_critic=core2.mlp_actor_critic,
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
