import tensorflow as tf


def get_variable_with_decay(name, shape, initializer, weight_decay=None):
    with tf.device("/cpu:0"):
        var = tf.get_variable(name, shape=shape, initializer=initializer)

    if weight_decay:
        wd = tf.nn.l2_loss(var) * weight_decay
        tf.add_to_collection("weight_decay", wd)

    return var


def ws3d_weight_initializer(name, shape, initializer, weight_decay=None):
    return get_variable_with_decay(name, shape, initializer, weight_decay)


def ws3d_bias_initializer_like(name, tensor, initializer):
    _, d, h, w, f = tensor.shape.as_list()

    return get_variable_with_decay(name, shape=(d, h, w, f), initializer=initializer)


def pool3d_weight_initializer_like(name, tensor, initializer, weight_decay=None):
    _, d, h, w, f = tensor.shape.as_list()

    return get_variable_with_decay(name=name, shape=(1, h, w, f), initializer=initializer, weight_decay=weight_decay)


def pool3d_bias_initializer_like(name, tensor, initializer):
    _, d, _, _, f = tensor.shape.as_list()

    return get_variable_with_decay(name, shape=(d, 1, 1, f), initializer=initializer)
