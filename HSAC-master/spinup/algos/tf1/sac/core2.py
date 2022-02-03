import numpy as np
import tensorflow as tf

EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def vae_encoder(x, hidden_sizes, activation=tf.nn.relu, output_activation=None):
    with tf.variable_scope('vae_encoder'):
        net = mlp(x, list(hidden_sizes[:-1]), activation, activation)
        mu = tf.layers.dense(net, hidden_sizes[-1], activation=output_activation)
        log_std = tf.layers.dense(net, hidden_sizes[-1], activation=output_activation)
        z = mu + tf.random_normal(tf.shape(mu)) * tf.exp(log_std)
    return z, mu, log_std  #z为输出的向量


def vae_decoder(z, hidden_sizes, output_dim, activation=tf.nn.relu, output_activation=None):
    with tf.variable_scope('vae_decoder'):
        net = mlp(z, list(hidden_sizes), activation, activation)
        net = tf.layers.dense(net, output_dim, activation=output_activation)
    return net

def embedding_init(data):
    with tf.variable_scope('embedding_init'):
        data = [i for item in data for i in item]
        TAG_SET = list(set(data))
        embedding_params = tf.Variable(tf.truncated_normal([len(TAG_SET), 2]))  #2 为 embedding outpu的维度
    return TAG_SET, embedding_params

def embedding(data,TAG_SET,params):
    with tf.variable_scope('embedding'):
        data = [i for item in data for i in item]
        table = tf.contrib.lookup.index_table_from_tensor(
            mapping=TAG_SET, default_value=-1)
        tags = tf.SparseTensor(indices=[[i] for i in range(0, len(data))],
                               values=table.lookup(tf.constant(data)),
                               dense_shape=[len(data)])
        embedded_tags = tf.nn.embedding_lookup_sparse(params, sp_ids=tags, sp_weights=None)
        # return embedded_tags

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        sess.run(embedded_tags)
        return embedded_tags.eval()

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


"""
Policies
"""

LOG_STD_MAX = 1
LOG_STD_MIN = -1

def mlp_gaussian_policy(x, a, a_d, hidden_sizes, activation, output_activation):
    #share the same hidden layer
    hidden_sizes1 = (256*2,256*2,256*2,256*2,)
    net = mlp(x, list(hidden_sizes1), activation, activation)

    #continue part
    act_dim = a.shape.as_list()[-1]
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    log_std = tf.layers.dense(net, act_dim, activation=None)
    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

    std = tf.exp(log_std)  #parameterlized deviation
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)


    #gb_discrete
    act_d_dim = a_d.shape.as_list()[-1]
    log_its = tf.layers.dense(net, act_d_dim, activation=None)
    # action and log action probabilites (log_softmax covers numerical problems)
    action_probs = tf.nn.softmax(log_its, axis=-1)
    log_action_probs = tf.nn.log_softmax(log_its, axis=-1)
    # policy with no noise
    mu_d = tf.argmax(log_its, axis=-1)
    pi_d_dist = tf.contrib.distributions.RelaxedOneHotCategorical(temperature=1.0, logits= log_its)
    pi_d = pi_d_dist.sample()

    logp_pi_d = -tf.reduce_sum( action_probs* log_action_probs, axis=-1)

    return mu, pi, logp_pi, mu_d, pi_d, logp_pi_d

def apply_squashing_func(mu, pi, logp_pi):
    # Adjustment to log prob
    # NOTE: This formula is a little bit magic. To get an understanding of where it
    # comes from, check out the original SAC paper (arXiv 1801.01290) and look in
    # appendix C. This is a more numerically-stable equivalent to Eq 21.
    # Try deriving it yourself as a (very difficult) exercise. :)
    logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, a_d, hidden_sizes=(256,256), activation=tf.nn.relu,
                     output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    # policy
    with tf.variable_scope('pi'):
        mu, pi, logp_pi, mu_d, pi_d,  logp_pi_d = policy(x, a, a_d, hidden_sizes, activation, output_activation)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
        # print('log_action_probs:',log_action_probs)
        # print('action_probs:',action_probs)
        # print('logp_pi:',logp_pi)

    # make sure actions are in correct range
    action_scale = tf.constant([[0.05, 0.025, 0.05, 0.5, 0.05]])
    action_bias = tf.constant([[0.05, 0.025, 0.05, 0.5, 0.05]])
    mu = tf.multiply(mu,action_scale) + action_bias
    pi = tf.multiply(pi,action_scale) + action_bias

    hiddensizes = (256*2,256*2,256*2,256*2,256*2)

    # vfs
    with tf.variable_scope('q1'):
        q1 = tf.squeeze(mlp(tf.concat([x,a,a_d], axis=-1), list(hiddensizes)+[1], activation, None), axis=1)

    with tf.variable_scope('q2'):
        q2 = tf.squeeze(mlp(tf.concat([x,a,a_d], axis=-1), list(hiddensizes)+[1], activation, None), axis=1)

    return mu, pi, logp_pi, mu_d, pi_d, logp_pi_d, q1, q2

