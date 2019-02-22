import tensorflow as tf
import numpy as np
from datasets import input_data
import layers
import os
import time
from tqdm import trange
from packaging import version

flags = tf.app.flags

# Checkpoint settings
flags.DEFINE_float("evaluate_every", 1, "Number of epoch for each evaluation (decimals allowed)")
flags.DEFINE_string("test_milestones", "15,20,25,30,35,40,45,50,75,100", "Each epoch where performs test")
flags.DEFINE_boolean("save_checkpoint", False, "Flag to save checkpoint or not")
flags.DEFINE_string("checkpoint_name", "3dpyranet.ckpt", "Name of checkpoint file")

# Input settings
dataset_path = "path/to/dataset"
flags.DEFINE_string("train_path",
                    os.path.join(dataset_path, "Training.npy"),
                    "Path to npy training set")
flags.DEFINE_string("train_labels_path",
                    os.path.join(dataset_path, "Training_label.npy"),
                    "Path to npy training set labels")
flags.DEFINE_string("val_path",
                    os.path.join(dataset_path, "TestVal.npy"),
                    "Path to npy val/test set")
flags.DEFINE_string("val_labels_path",
                    os.path.join(dataset_path, "TestVal_label.npy"),
                    "Path to npy val/test set labels")
flags.DEFINE_string("save_path", "train_dir",
                    "Path where to save network model")
flags.DEFINE_boolean("random_run", False, "Set usage of random data for debug purpose")

# Input parameters
flags.DEFINE_integer("batch_size", 100, "Batch size")
flags.DEFINE_integer("depth", 16, "Number of consecutive samples")
flags.DEFINE_integer("height", 100, "Samples height")
flags.DEFINE_integer("width", 100, "Samples width")
flags.DEFINE_integer("in_channels", 1, "Samples channels")
flags.DEFINE_integer("num_classes", 6, "Number of classes")

# Preprocessing
flags.DEFINE_boolean("normalize", True, "Normalize image in range 0-1")

# Hyper-parameters settings
flags.DEFINE_integer("feature_maps", 3, "Number of maps to use (strict model shares the number of maps in each layer)")
flags.DEFINE_float("learning_rate", 0.00015, "Learning rate")
flags.DEFINE_integer("decay_steps", 15, "Number of iteration for each decay")
flags.DEFINE_float("decay_rate", 0.1, "Learning rate decay")
flags.DEFINE_integer("max_steps", 50, "Maximum number of epoch to perform")
flags.DEFINE_float("weight_decay", None, "L2 regularization lambda")

# Optimization algorithm
opt_type = ["GD", "MOMENTUM", "ADAM"]
flags.DEFINE_string("optimizer", opt_type[1], "Optimization algorithm")
flags.DEFINE_boolean("use_nesterov", False, "Use Nesterov Momentum")

params_str = ""
FLAGS = flags.FLAGS
if version.parse(tf.__version__) < version.parse('1.4'):
    FLAGS._parse_flags()
    print("Parameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        params_str += "{} = {}\n".format(attr.upper(), value)
        print("{} = {}".format(attr.upper(), value))
    print("")
else:
    params_str = "NOT SUPPORTED, UNABLE TO TRY ON TF VERSION > 1.2"


def compute_loss(name_scope, logits, labels):
    with tf.name_scope("Loss_{}".format(name_scope)):
        cross_entropy_mean = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )

        tf.summary.scalar(
            name_scope + '_cross_entropy',
            cross_entropy_mean
        )

        weight_decay_loss = tf.get_collection('weight_decay')

        if len(weight_decay_loss) > 0:
            tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss))

            # Calculate the total loss for the current tower.
            total_loss = cross_entropy_mean + weight_decay_loss
            tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss))
        else:
            total_loss = cross_entropy_mean

        return total_loss


def compute_accuracy(logits, labels):
    with tf.name_scope("Accuracy"):
        correct_pred = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
        return accuracy


def prepare_dataset():
    train_x, train_y, val_x, val_y = input_data.read_dataset(
        FLAGS.train_path,
        FLAGS.train_labels_path,
        FLAGS.val_path,
        FLAGS.val_labels_path
    )

    batch_step = train_x.shape[0] // FLAGS.batch_size
    test_batch_step = val_x.shape[0] // FLAGS.batch_size

    FLAGS.decay_steps *= batch_step
    FLAGS.max_steps *= batch_step
    FLAGS.evaluate_every = int(FLAGS.evaluate_every * batch_step)

    if FLAGS.normalize:
        train_x = input_data.normalize(train_x, name="training set")
        val_x = input_data.normalize(val_x, name="val set")

    train_batch = input_data.generate_batch(train_x, train_y, batch_size=FLAGS.batch_size, shuffle=True)
    val_batch = input_data.generate_batch(val_x, val_y, batch_size=FLAGS.batch_size, shuffle=True)
    test_batch = input_data.generate_batch(val_x, val_y, batch_size=FLAGS.batch_size, shuffle=False)

    return batch_step, test_batch_step, train_batch, val_batch, test_batch


def random_dataset():
    train_x = np.random.rand(FLAGS.batch_size, FLAGS.depth, FLAGS.height, FLAGS.width, FLAGS.in_channels)
    train_y = np.random.random_integers(0, FLAGS.num_classes, FLAGS.batch_size)
    val_x = np.random.rand(FLAGS.batch_size, FLAGS.depth, FLAGS.height, FLAGS.width, FLAGS.in_channels)
    val_y = np.random.random_integers(0, FLAGS.num_classes, FLAGS.batch_size)

    batch_step = train_x.shape[0] // FLAGS.batch_size
    test_batch_step = val_x.shape[0] // FLAGS.batch_size

    FLAGS.decay_steps *= batch_step
    FLAGS.max_steps *= batch_step
    FLAGS.evaluate_every = int(FLAGS.evaluate_every * batch_step)

    if FLAGS.normalize:
        train_x = input_data.normalize(train_x, name="training set")
        val_x = input_data.normalize(val_x, name="val set")

    train_batch = input_data.generate_batch(train_x, train_y, batch_size=FLAGS.batch_size, shuffle=True)
    val_batch = input_data.generate_batch(val_x, val_y, batch_size=FLAGS.batch_size, shuffle=True)
    test_batch = input_data.generate_batch(val_x, val_y, batch_size=FLAGS.batch_size, shuffle=False)

    return batch_step, test_batch_step, train_batch, val_batch, test_batch


def train():
    if FLAGS.random_run:
        batch_step, test_batch_step, train_batch, val_batch, test_batch = random_dataset()
    else:
        batch_step, test_batch_step, train_batch, val_batch, test_batch = prepare_dataset()

    global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)

    input_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.depth,
                                                          FLAGS.height, FLAGS.width, FLAGS.in_channels),
                                       name="input_placeholder")
    labels_placeholder = tf.placeholder(tf.int64, shape=[FLAGS.batch_size], name="label_placeholder")

    lr_decay = tf.train.exponential_decay(FLAGS.learning_rate, global_step=global_step,
                                          decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate,
                                          staircase=True)

    optimizer = None
    if FLAGS.optimizer == "GD":
        optimizer = tf.train.GradientDescentOptimizer(lr_decay)
    elif FLAGS.optimizer == "ADAM":
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    elif FLAGS.optimizer == "MOMENTUM":
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_decay, momentum=0.9, use_nesterov=FLAGS.use_nesterov)

    net = layers.strict_norm_net(input_placeholder, feature_maps=FLAGS.feature_maps, weight_decay=FLAGS.weight_decay)

    logits = layers.fc_layer(net, weight_size=FLAGS.num_classes, act_fn=None,
                                     name="FC_OUT", weight_decay=FLAGS.weight_decay)

    with tf.name_scope("Loss"):
        loss = compute_loss("Dataset_Name", logits, labels_placeholder)

    accuracy = compute_accuracy(logits, labels_placeholder)

    train_op = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        current_exec = time.time()
        train_dir = FLAGS.save_path
        model_save_dir = os.path.join(train_dir, str(current_exec))

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        with open(os.path.join(model_save_dir, "params_settings"), "w+") as f:
            f.write(params_str)

        # Create summary writer
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(model_save_dir, 'train'), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(model_save_dir, 'val'), sess.graph)
        total_steps = trange(FLAGS.max_steps)

        test_milestones = map(int, FLAGS.test_milestones.split(","))

        t_acc, v_acc, t_loss, v_loss = .0, .0, .0, .0
        for step in total_steps:
            train_images, train_labels = next(train_batch)
            _, t_loss = sess.run([train_op, loss], feed_dict={
                input_placeholder: train_images,
                labels_placeholder: train_labels
            })

            t_loss = float(np.mean(t_loss))
            total_steps.set_description(
                'Loss: {:.4f}/{:.4f} - t_acc: {:.3f} - v_acc: {:.3f}'.format(
                    t_loss, v_loss, t_acc, v_acc)
            )

            # Evaluate the model periodically.
            if step % FLAGS.evaluate_every == 0 or (step + 1) == FLAGS.max_steps:

                summary, t_acc = sess.run(
                    [merged, accuracy],
                    feed_dict={
                        input_placeholder: train_images,
                        labels_placeholder: train_labels
                    })
                train_writer.add_summary(summary, step)

                val_images, val_labels = next(val_batch)
                summary, v_loss, v_acc = sess.run(
                    [merged, loss, accuracy],
                    feed_dict={
                        input_placeholder: val_images,
                        labels_placeholder: val_labels
                    })
                val_writer.add_summary(summary, step)

                v_loss = float(np.mean(v_loss))
                total_steps.set_description(
                    'Loss: {:.4f}/{:.4f} - t_acc: {:.3f} - v_acc: {:.3f}'.format(
                        t_loss, v_loss, t_acc, v_acc)
                )

                # Test on the whole validation or test set
                if (step / batch_step) in test_milestones or (step + 1) == FLAGS.max_steps:
                    if FLAGS.save_checkpoint:
                        saver.save(sess, os.path.join(model_save_dir, FLAGS.checkpoint_name), global_step=step)
                    print("testing...")

                    test_acc_list, test_loss_list = [], []
                    for _ in trange(test_batch_step):
                        test_images, test_labels = next(test_batch)
                        test_acc, test_loss = sess.run(
                            [accuracy, loss],
                            feed_dict={
                                input_placeholder: test_images,
                                labels_placeholder: test_labels
                            })
                        test_acc_list.append(test_acc)
                        test_loss_list.append(test_loss)

                    print("Epoch {} - Acc: {} - Loss {}".format((step / batch_step), np.mean(test_acc_list),
                                                                np.mean(test_loss_list)))

                    with open(os.path.join(model_save_dir, "test_results"), "a") as f:
                        f.write("Epoch: {}\n".format(step / batch_step))
                        f.write("\tMean train accuracy: {}\n".format(t_acc))
                        f.write("\tMean train loss: {}\n\n".format(t_loss))
                        f.write("\tMean test accuracy: {}\n".format(np.mean(test_acc_list)))
                        f.write("\tMean test loss: {}\n\n".format(np.mean(test_loss_list)))


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
