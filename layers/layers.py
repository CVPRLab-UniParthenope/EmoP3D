import tensorflow as tf
from models.pyranet import *
from .variables import *


def ws3d_layer(input, out_filters, rf=(3, 4, 4), strides=(1, 1, 1, 1, 1), act_fn=lrelu,
               initializer=None, weight_decay=None, padding="VALID",
               data_format="NDHWC", log=False, reuse=False, name="weighted_sum_3d_layer"):
    with tf.variable_scope(name, reuse=reuse):
        _, d, h, w, c = map(int, input.shape)
        if not initializer:
            initializer = tf.truncated_normal_initializer()

        weights = ws3d_weight_initializer("weights", shape=(rf[0], h, w, c, out_filters),
                                          initializer=initializer, weight_decay=weight_decay)

        net = ws3d(input, weights, rf=rf, strides=strides,
                   padding=padding, data_format=data_format, name="ws3d")

        bias = ws3d_bias_initializer_like("bias", tensor=net, initializer=initializer)

        net = tf.add(net, bias, name="bias_add")

        if act_fn:
            net = act_fn(net)

        if log:
            tf.logging.info("{} | RF: {} - strides: {}".format(name, rf, strides[1:-1]))
            tf.logging.info("\t{} {}".format(weights.name, weights.shape))
            tf.logging.info("\t{} {}".format(bias.name, bias.shape))
            tf.logging.info("\t{} {}".format(net.name, net.shape))

        return net


def pool3d_layer(input_data, weight_depth=3, rf=(3, 2, 2), strides=(1, 1, 2, 2, 1), act_fn=lrelu,
                 initializer=None, weight_decay=None, pool_type=max_pool3d,
                 padding="VALID", data_format="NDHWC", reuse=False, log=False, name="pooling_3d_layer"):
    with tf.variable_scope(name, reuse=reuse):
        if not initializer:
            initializer = tf.truncated_normal_initializer()

        net = pool_type(input_data, weight_depth=weight_depth, rf=rf, strides=strides,
                           padding=padding, data_format=data_format)

        weights = pool3d_weight_initializer_like("weights", tensor=net,
                                                 initializer=initializer, weight_decay=weight_decay)
        bias = pool3d_bias_initializer_like("bias", tensor=net, initializer=initializer)

        net = tf.multiply(net, weights, name="mul_weights")
        net = tf.add(net, bias, name="bias_add")

        if act_fn:
            net = act_fn(net)

        if log:
            tf.logging.info("{} | Weight depth: {} - strides: {}".format(name, weight_depth, strides[1:-1]))
            tf.logging.info("\t{} {}".format(weights.name, weights.shape))
            tf.logging.info("\t{} {}".format(bias.name, bias.shape))
            tf.logging.info("\t{} {}".format(net.name, net.shape))

        return net


def normalization_layer(x, axes=(2, 3), epsilon=1e-8, name="norm_layer"):
    with tf.name_scope(name):
        beta = tf.constant(0., dtype=tf.float32)
        gamma = tf.constant(1., dtype=tf.float32)
        mean, var = tf.nn.moments(x, axes, keep_dims=True)
        normalized = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name="standardization")
        normalized = tf.where(tf.is_finite(normalized), normalized, tf.zeros_like(normalized))  # avoid nan and inf
        return normalized


def fc_layer(inputs, weight_size, name, act_fn=None, weight_decay=None):
    with tf.variable_scope(name):
        if len(inputs.shape) > 2:
            inputs = tf.reshape(inputs, [int(inputs.shape[0]), -1])

        w = get_variable_with_decay("weights", shape=[int(inputs.shape[1]), weight_size],
                                      initializer=tf.contrib.
                                      layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                          uniform=True, dtype=tf.float32),
                                      weight_decay=weight_decay)
        b = get_variable_with_decay("bias", shape=[weight_size],
                                      initializer=tf.contrib.
                                      layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                          uniform=True, dtype=tf.float32))

        net = tf.add(tf.matmul(inputs, w), b, name="mat_mul_bias_add")
        if act_fn:
            net = act_fn(net)

        return net
