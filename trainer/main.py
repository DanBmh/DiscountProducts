import os
import shutil
import numpy as np
import tensorflow as tf
import time
import argparse

tf.logging.set_verbosity(tf.logging.INFO)

DELETE_OLD = False
EPOCH_NUMBER = 8
BATCH_SIZE = 2048 * 2

# gcloud ml-engine local train --module-name="trainer.main" --package-path="./trainer" -- --datapath="data/" --trainpath="traindata/"

# BUCKET_NAME="video-enhancer-bucket"
# JOB_NAME="dispro_learn_$(date +%Y%m%d_%H%M%S)"
# JOB_DIR="gs://$BUCKET_NAME/modeldata/$JOB_NAME"
# REGION="europe-west1"
# gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://$BUCKET_NAME/modeldata/$JOB_NAME --runtime-version="1.4" --scale-tier="BASIC" --module-name="trainer.main" --package-path="./trainer" --region $REGION -- --datapath="gs://$BUCKET_NAME/data/" --trainpath="gs://$BUCKET_NAME/traindata/"

# gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://$BUCKET_NAME/modeldata/$JOB_NAME --runtime-version="1.4" --scale-tier="BASIC_GPU" --module-name="trainer.main" --package-path="./trainer" --region $REGION -- --datapath="gs://$BUCKET_NAME/data/" --trainpath="gs://$BUCKET_NAME/traindata/"
# gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://$BUCKET_NAME/modeldata/$JOB_NAME --runtime-version="1.4" --scale-tier="CUSTOM" --config="config.yaml" --module-name="trainer.main" --package-path="./trainer" --region $REGION -- --datapath="gs://$BUCKET_NAME/data/" --trainpath="gs://$BUCKET_NAME/traindata2/"

# =================================================================================================


def model_fn_small(features, labels, training):
    """Simple fully connected prediction layer"""

    # Input Layer
    input1 = tf.identity(features["image_1"], name="input1")
    input2 = tf.identity(features["image_2"], name="input2")
    input3 = tf.identity(features["image_3"], name="input3")
    input4 = tf.identity(features["image_4"], name="input4")
    input5 = tf.identity(features["level1_category"], name="input5")
    input6 = tf.identity(features["level2_category"], name="input6")
    input7 = tf.identity(features["pid"], name="input7")
    input8 = tf.identity(True, name="training")

    input_layer = tf.concat([tf.layers.flatten(input1), tf.layers.flatten(
        input2), tf.layers.flatten(input3), tf.layers.flatten(input4)], 1)

    net = tf.layers.dropout(input_layer, training=input8)
    net = tf.layers.dense(net, 5720)

    # Output Layer
    output_layer = tf.nn.softmax(net, name="imageprediction")

    # Calculate Loss
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=net))
    accuracy = tf.reduce_mean(tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(output_layer, 1)))

    # Summaries
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.histogram("label", labels)
    tf.summary.histogram("prediction", output_layer)
    tf.summary.histogram("input", input_layer)

    return loss, accuracy
# =================================================================================================


def model_fn_large(features, labels, training):
    """Prediction layers to predict all category levels"""

    # Input Layer
    input1 = tf.identity(features["image_1"], name="input1")
    input2 = tf.identity(features["image_2"], name="input2")
    input3 = tf.identity(features["image_3"], name="input3")
    input4 = tf.identity(features["image_4"], name="input4")
    input5 = tf.identity(features["level1_category"], name="input5")
    input6 = tf.identity(features["level2_category"], name="input6")
    input7 = tf.identity(features["pid"], name="input7")
    input8 = tf.identity(True, name="training")

    input_layer = tf.concat([tf.layers.flatten(input1), tf.layers.flatten(
        input2), tf.layers.flatten(input3), tf.layers.flatten(input4)], 1)

    # Network to predict category layers
    net = tf.layers.dropout(input_layer, training=input8)
    level1 = tf.layers.dense(net, 49)
    soft1 = tf.nn.softmax(level1)

    net = tf.concat([soft1, input_layer], 1)
    net = tf.layers.dropout(net, training=input8)
    level2 = tf.layers.dense(net, 483)
    soft2 = tf.nn.softmax(level2)

    net = tf.concat([soft1, soft2, input_layer], 1)
    net = tf.layers.dropout(net, training=input8)
    level3 = tf.layers.dense(net, 5270)

    # Output Layer
    output_layer = tf.nn.softmax(level3, name="imageprediction")

    # Calculate Loss
    loss1 = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input5, logits=level1))
    loss2 = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input6, logits=level2))
    loss3 = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=level3))

    loss = loss1 * 0.15 + loss2 * 0.15 + loss3 * 0.7
    accuracy = tf.reduce_mean(tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(output_layer, 1)))

    # Summaries
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.histogram("label", labels)
    tf.summary.histogram("prediction", output_layer)
    tf.summary.histogram("input", input_layer)

    return loss, accuracy
# =================================================================================================


def adjust_tensor(tensor):
    """Fill tensors to 4032 values if it has less"""

    tensor = tf.pad(tensor, tf.constant([[0, 4031, ], [0, 0]]), "CONSTANT")
    tensor = tf.slice(tensor, [0, 0], [4032, 1])
    return tensor


def dataset_input_fn(filenames):
    dataset = tf.data.TFRecordDataset(filenames)

    # Parse single example
    def parser(record):
        keys_to_features = {
            'pid': tf.FixedLenFeature((), tf.int64, default_value=0),
            'level3_category': tf.FixedLenFeature((), tf.int64, default_value=0),
            'level2_category': tf.FixedLenFeature((), tf.int64, default_value=0),
            'level1_category': tf.FixedLenFeature((), tf.int64, default_value=0),
            'images': tf.FixedLenFeature((), tf.int64, default_value=0),
            'length': tf.FixedLenFeature((), tf.int64, default_value=0),
            'image_1': tf.VarLenFeature(dtype=tf.float32),
            'image_2': tf.VarLenFeature(dtype=tf.float32),
            'image_3': tf.VarLenFeature(dtype=tf.float32),
            'image_4': tf.VarLenFeature(dtype=tf.float32)
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Preprocess images
        image_1 = tf.sparse_tensor_to_dense(parsed["image_1"], default_value=0)
        image_1 = tf.reshape(image_1, shape=(4032, 1))

        image_2 = tf.sparse_tensor_to_dense(parsed["image_2"], default_value=0)
        image_2 = tf.reshape(image_2, shape=(-1, 1))
        image_2 = adjust_tensor(image_2)

        image_3 = tf.sparse_tensor_to_dense(parsed["image_3"], default_value=0)
        image_3 = tf.reshape(image_3, shape=(-1, 1))
        image_3 = adjust_tensor(image_3)

        image_4 = tf.sparse_tensor_to_dense(parsed["image_4"], default_value=0)
        image_4 = tf.reshape(image_4, shape=(-1, 1))
        image_4 = adjust_tensor(image_4)

        images = [tf.cast(parsed["images"], tf.int32)]
        length = [tf.cast(parsed["length"], tf.int32)]
        pid = [tf.cast(parsed["pid"], tf.int32)]
        level3_category = tf.cast(parsed["level3_category"], tf.int32)
        level2_category = tf.cast(parsed["level2_category"], tf.int32)
        level1_category = tf.cast(parsed["level1_category"], tf.int32)

        return {"image_1": image_1, "image_2": image_2, "image_3": image_3, "image_4": image_4,
                "images": images, "length": length, "pid": pid,
                "level2_category": level2_category, "level1_category": level1_category}, level3_category

    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=40000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat(EPOCH_NUMBER)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    return features, labels

# ====================================================================================================


if __name__ == '__main__':

    # Arguments are needed for running at google cloudml
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath',
        help='GCS or local paths to training data',
        required=True
    )
    parser.add_argument(
        '--trainpath',
        help='GCS or local paths for trained data',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='GCS or local paths to jobdir',
        required=False
    )
    args = parser.parse_args()
    arguments = args.__dict__
    datapath = arguments.pop('datapath')
    trainpath = arguments.pop('trainpath')

    # Delete old training files
    if (DELETE_OLD and os.path.isdir("traindata")):
        shutil.rmtree("traindata")

    print("\n ===== Starting Training ===== \n")

    next_example, next_label = dataset_input_fn([
        datapath + "train-00000-of-00008.tfrecords",
        datapath + "train-00001-of-00008.tfrecords",
        datapath + "train-00002-of-00008.tfrecords",
        datapath + "train-00003-of-00008.tfrecords",
        datapath + "train-00004-of-00008.tfrecords",
        datapath + "train-00005-of-00008.tfrecords",
        datapath + "train-00006-of-00008.tfrecords",
        datapath + "train-00007-of-00008.tfrecords"
    ])
    # datapath + "example-00000-of-00002.tfrecords",
    # datapath + "example-00001-of-00002.tfrecords"])

    # lossf, accf = model_fn_small(next_example, next_label, True)
    lossf, accf = model_fn_large(next_example, next_label, True)

    training_op = tf.train.AdamOptimizer().minimize(
        lossf, global_step=tf.train.get_or_create_global_step())

    with tf.train.MonitoredTrainingSession(checkpoint_dir=trainpath, save_summaries_steps=100, log_step_count_steps=10) as sess:

        while not sess.should_stop():

            losso, acco, _ = sess.run([lossf, accf, training_op])

            print('Step:', sess.run(tf.train.get_global_step()),
                  '- Loss:', losso, '- Accuracy:', acco)

            # exit()
