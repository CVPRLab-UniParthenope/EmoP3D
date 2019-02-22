import tensorflow as tf
from .layers import ws3d_layer, pool3d_layer, normalization_layer
from models import pyranet


def strict_norm_net(inputs, feature_maps=3, act_fn=pyranet.lrelu, weight_decay=None,
                    log=False, name="STRICT_3DPYRANET"):
    with tf.name_scope(name):
        net = ws3d_layer(inputs, feature_maps, name="L1WS",
                         act_fn=act_fn, weight_decay=weight_decay,
                         initializer=tf.contrib.
                         layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                             uniform=True, dtype=tf.float32),
                         log=log)
        net = normalization_layer(net, axes=(2, 3), name="NORM_2")

        net = pool3d_layer(net, name="L3P", act_fn=act_fn, weight_decay=weight_decay,
                           initializer=tf.contrib.
                           layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                               uniform=True, dtype=tf.float32),
                           log=log)
        net = normalization_layer(net, axes=(2, 3), name="NORM_4")

        net = ws3d_layer(net, feature_maps, name="L5WS",
                         act_fn=act_fn, weight_decay=weight_decay,
                         initializer=tf.contrib.
                         layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                             uniform=True, dtype=tf.float32),
                         log=True)
        net = normalization_layer(net, axes=(2, 3), name="NORM_6")

        return net
