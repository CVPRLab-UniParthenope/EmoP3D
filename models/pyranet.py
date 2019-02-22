import tensorflow as tf
import numpy as np


def lrelu(input, alpha=0.01, name="lrelu"):
    with tf.name_scope(name):
        leaky_input = alpha * input
        return tf.maximum(leaky_input, input, name=name)


def ws3d_weight_initializer(name, shape, initializer, weight_decay=None):
    with tf.device("/cpu:0"):
        weights = tf.get_variable(name, shape=shape, initializer=initializer)

    if weight_decay:
        decay = tf.nn.l2_loss(weights) * weight_decay
        tf.add_to_collection("weight_decay", decay)

    return weights


def ws3d_bias_initializer_like(name, tensor, initializer):
    _, d, h, w, f = tensor.shape.as_list()

    with tf.device("/cpu:0"):
        bias = tf.get_variable(name=name,
                               shape=(d, h, w, f), initializer=initializer)

    return bias


def ws3d_layer_output_shape(input_shape, rf=(3, 4, 4), strides=(1, 1, 1, 1, 1), padding="VALID"):
    padding = padding.upper()
    input_shape = list(map(float, input_shape))
    if padding == "VALID":
        output_depth = np.round((input_shape[0] - rf[0] + 1.) / strides[1])
        output_height = np.round((input_shape[1] - rf[1] + 1.) / strides[2])
        output_width = np.round((input_shape[2] - rf[2] + 1.) / strides[3])
    elif padding == "SAME":
        output_depth, output_height, output_width = [np.round(s / strides[i])
                                                     for i, s in zip(strides[1:-1], input_shape)]
    else:
        raise NotImplementedError("{} is not a valid padding type".format(padding))

    return output_depth, output_height, output_width


def pool3d_bias_initializer_like(name, tensor, initializer):
    _, d, h, w, f = tensor.shape.as_list()

    with tf.device("/cpu:0"):
        bias = tf.get_variable(name=name, shape=(d, 1, 1, f), initializer=initializer)

    return bias


def pool3d_weight_initializer_like(name, tensor, initializer, weight_decay=None):
    _, d, h, w, f = tensor.shape.as_list()

    with tf.device("/cpu:0"):
        weights = tf.get_variable(name=name, shape=(1, h, w, f), initializer=initializer)

    if weight_decay:
        decay = tf.nn.l2_loss(weights) * weight_decay
        tf.add_to_collection("weight_decay", decay)

    return weights


def pool3d_layer_output_shape(input_shape, rf=(2, 2, 2), strides=(1, 2, 2, 2, 1), padding="VALID"):
    padding = padding.upper()
    input_shape = list(map(float, input_shape))
    if padding == "VALID":
        output_depth = np.round((input_shape[0] - rf[0] + 1) / strides[1])
        output_height = np.round((input_shape[1] - rf[1] + 1) / strides[2])
        output_width = np.round((input_shape[2] - rf[2] + 1) / strides[3])
    elif padding == "SAME":
        output_depth, output_height, output_width = [np.round(s / strides[i])
                                                     for i, s in zip(strides[1:-1], input_shape)]
    else:
        raise NotImplementedError("{} is not a valid padding type".format(padding))

    return output_depth, output_height, output_width


def ws3d(input_tensor, weights, rf=(3, 4, 4), strides=(1, 1, 1, 1, 1),
         padding="VALID", data_format="NDHWC", name="ws3d"):
    """
    PyraNet uses a VALID padding type.

    :param input_tensor:
    :param weights:
    :param rf: a list containing the receptive field sizes [height, width], it is the r_l in the original paper
    :param strides: overlap of receptive fields [1, stride_h, stride_w, 1], it is the o_l in the original paper
    :param padding: only VALID is admitted
    :param data_format:
    :param name:
    :return:
    """

    with tf.name_scope(name):

        input_shape = list(input_tensor.shape)
        if data_format == "NDHWC":
            n, d, h, w, c = map(int, input_shape)
        else:
            n, c, d, h, w = map(int, input_shape)

        out_channels = int(weights.shape[-1])

        output_depth, _, _ = map(int, ws3d_layer_output_shape((d, h, w), rf=rf, strides=strides, padding=padding))
        correlation = []

        # Bias and conv need to be intern to weighting
        with tf.name_scope("Input_Weighting_Op"):
            conv_weights = tf.constant(1.0, tf.float32, shape=(rf[0], rf[1], rf[2], c, 1),
                                       name="{}/conv_kernel".format(name))

            for fm in range(out_channels):
                assign_ops = []
                for cd in range(output_depth):
                    s = cd * strides[1]
                    out_mul = tf.multiply(input_tensor[:, s:s + rf[0], :, :, :], weights[:, :, :, :, fm])


                    with tf.name_scope("Correlation_Op"):
                        corr = tf.nn.conv3d(out_mul, conv_weights,
                                            padding=padding, strides=strides, name="xcorr3d")
                    assign_ops.append(corr)

                correlation.append(assign_ops)
            corr_axis_sorted = tf.transpose(correlation, [2, 1, 3, 4, 5, 0, 6], name="xcorr_sorted")
            return tf.squeeze(corr_axis_sorted, axis=[2, 6], name="xcorr_output")


def max_pool3d(input_data, weight_depth=3, rf=(3, 2, 2), strides=(1, 1, 2, 2, 1),
               padding='VALID', data_format='NDHWC', name="pool3D"):
    with tf.name_scope(name):
        _, d, h, w, fms = map(int, input_data.shape)

        out_depth, _, _ = map(int, ws3d_layer_output_shape(input_shape=(d, h, w), rf=rf, strides=strides))
        pool = tf.nn.max_pool3d(input_data, strides, strides, padding=padding,
                                data_format=data_format, name="max_pooling3d")

        output = []
        for depth in range(out_depth):
            output.append(tf.reduce_max(pool[:, depth:depth + weight_depth], axis=1))

        output = tf.transpose(output, [1, 0, 2, 3, 4])
        return output


def avg_pool3d(input_data, weight_depth=3, rf=(3, 2, 2), strides=(1, 1, 2, 2, 1),
               padding='VALID', data_format='NDHWC', name="pool3D"):
    with tf.name_scope(name):
        _, d, h, w, fms = map(int, input_data.shape)

        out_depth, _, _ = map(int, ws3d_layer_output_shape(input_shape=(d, h, w), rf=rf, strides=strides))
        pool = tf.nn.avg_pool3d(input_data, strides, strides, padding=padding,
                                data_format=data_format, name="avg_pooling3d")

        output = []
        for depth in range(out_depth):
            output.append(tf.reduce_mean(pool[:, depth:depth + weight_depth], axis=1))

        output = tf.transpose(output, [1, 0, 2, 3, 4])
        return output
