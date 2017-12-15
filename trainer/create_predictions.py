import os
import shutil
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import argparse
from tensorflow.python.lib.io import file_io

tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 1024

# gcloud ml-engine local train --module-name="trainer.create_predictions" --package-path="./trainer" -- --datapath="data/" --trainpath="traindata/"

# BUCKET_NAME="video-enhancer-bucket"
# JOB_NAME="dispro_predict_$(date +%Y%m%d_%H%M%S)"
# JOB_DIR="gs://$BUCKET_NAME/modeldata/$JOB_NAME"
# REGION="europe-west1"
# gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://$BUCKET_NAME/modeldata/$JOB_NAME --runtime-version="1.4" --scale-tier="BASIC" --module-name="trainer.create_predictions" --package-path="./trainer" --region $REGION -- --datapath="gs://$BUCKET_NAME/data/" --trainpath="gs://$BUCKET_NAME/traindata2/"

# gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://$BUCKET_NAME/modeldata/$JOB_NAME --runtime-version="1.4" --scale-tier="BASIC_GPU" --module-name="trainer.create_predictions" --package-path="./trainer" --region $REGION -- --datapath="gs://$BUCKET_NAME/data/" --trainpath="gs://$BUCKET_NAME/traindata2/"
# gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://$BUCKET_NAME/modeldata/$JOB_NAME --runtime-version="1.4" --scale-tier="CUSTOM" --config="config.yaml" --module-name="trainer.create_predictions" --package-path="./trainer" --region $REGION -- --datapath="gs://$BUCKET_NAME/data/" --trainpath="gs://$BUCKET_NAME/traindata2/"

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
    dataset = dataset.shuffle(buffer_size=1)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    return features, labels


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


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

    ckpt = tf.train.get_checkpoint_state(trainpath)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

    next_example, next_label = dataset_input_fn(
        [datapath + 'test-00000-of-00002.tfrecords', datapath + 'test-00001-of-00002.tfrecords'])
    # [datapath + "example-00000-of-00002.tfrecords", datapath + "example-00001-of-00002.tfrecords"])

    # Restore original category numbers
    categories = pd.read_csv(file_io.FileIO(
        datapath + 'category_numbers.csv', mode='r'))

    # Write predictions to tfrecord (I had some problems in google cloudml with writing to csv directly)
    writer_prediction = tf.python_io.TFRecordWriter(
        datapath + 'prediction.tfrecords')

    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)

        input1 = sess.graph.get_tensor_by_name("input1:0")
        input2 = sess.graph.get_tensor_by_name("input2:0")
        input3 = sess.graph.get_tensor_by_name("input3:0")
        input4 = sess.graph.get_tensor_by_name("input4:0")
        input7 = sess.graph.get_tensor_by_name("input7:0")
        input8 = sess.graph.get_tensor_by_name("training:0")
        predlayer = sess.graph.get_tensor_by_name("imageprediction:0")

        count = 0

        while True:
            try:
                features, label = sess.run([next_example, next_label])
                index = features["pid"][0][0]

                if(count % 1000 == 0):
                    print('Count:', count)
                    print('Index:', index)

                pred, ido = sess.run([predlayer, input7], feed_dict={input1: features["image_1"], input2: features["image_2"],
                                                                     input3: features["image_3"], input4: features["image_4"],
                                                                     input7: features["pid"], input8: False})

                prediction = sess.run(tf.argmax(pred, 1))

                # Walk through batch and write single examples for predictions
                for i, iio in enumerate(prediction):
                    line = categories.loc[categories['category_level3'] == iio]
                    catid = line.iloc[0]['category_id']

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'pid': _int64_feature(ido[i][0]),
                        'category': _int64_feature(catid)}))
                    writer_prediction.write(example.SerializeToString())

                    count = count + 1

            except tf.errors.OutOfRangeError:
                print("run finished")
                break

    writer_prediction.close()
