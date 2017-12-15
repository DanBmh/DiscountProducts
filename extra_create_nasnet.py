import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

sys.path.append('models/research/slim')
from nets.nasnet import nasnet

# =================================================================================================


def create(size):
    inputs = tf.placeholder(tf.float32, shape=(
        None, size, size, 3), name="InputHolder")
    tf.train.get_or_create_global_step()

    with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
        logits, end_points = nasnet.build_nasnet_large(
            inputs, 1001,  is_training=False)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, 'pretrained/large/model.ckpt')

        pb_visual_writer = tf.summary.FileWriter('training')
        pb_visual_writer.add_graph(sess.graph)

        saver.save(sess, "pretrained/large/nasnet_dp.ckpt")

# =================================================================================================


create_large(180)
