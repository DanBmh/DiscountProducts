import time
import argparse
import pandas as pd
import numpy as np
import cv2 as cv
import bson
import tensorflow as tf
from tensorflow.python.lib.io import file_io

USE_PRETRAINED = True
IMAGE_SIZE = 90
SHARD_NUMBER = 1

# python .\create_records.py --datapath="../data/" --pretrainpath="../pretrained/"
# gcloud ml-engine local train --module-name="trainer.create_records" --package-path="./trainer" -- --datapath="data/" --pretrainpath="pretrained/"

# BUCKET_NAME="video-enhancer-bucket"
# JOB_NAME="dispro_train_$(date +%Y%m%d_%H%M%S)"
# JOB_NAME="dispro_test_$(date +%Y%m%d_%H%M%S)"
# JOB_DIR="gs://$BUCKET_NAME/modeldata/$JOB_NAME"
# REGION="europe-west1"
# gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://$BUCKET_NAME/$JOB_NAME --runtime-version="1.2" --module-name="trainer.create_records" --package-path="./trainer" --region $REGION -- --datapath="gs://$BUCKET_NAME/data/" --pretrainpath="gs://$BUCKET_NAME/pretrained/"

# gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://$BUCKET_NAME/modeldata/$JOB_NAME --runtime-version="1.2" --scale-tier="BASIC_GPU" --module-name="trainer.create_records" --package-path="./trainer" --region $REGION -- --datapath="gs://$BUCKET_NAME/data/" --pretrainpath="gs://$BUCKET_NAME/pretrained/"

# =================================================================================================


def process_image(str):
    """Preprocess image from imagestring"""

    nparr = np.fromstring(str, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    img = cv.resize(src=img, dsize=(IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv.INTER_AREA)

    img = np.array(img, dtype=np.float32)
    img = ((img / 255) - 0.5) * 2

    return img


def create_batch_examples(batchrows):
    """Run through set of data and calculate bottleneck features"""

    if(USE_PRETRAINED == True):
        length = 4032
    else:
        length = IMAGE_SIZE * IMAGE_SIZE * 3

    ids = np.zeros(len(batchrows) * 4) - 1
    images = np.zeros((len(batchrows) * 4, IMAGE_SIZE, IMAGE_SIZE, 3))
    index = 0
    batchexamples = []

    # Collect images and product ids
    for row in (batchrows):
        for img in (row['imgs']):
            ids[index] = row['_id']
            images[index] = process_image(img['picture'])
            index = index + 1

    images = images[:index]

    if(USE_PRETRAINED == True):
        # Bottleneck values
        images = sess.run(predlayer, feed_dict={inputlayer: images})

    # Preprocess row and put images and productdata together again
    for row in (batchrows):
        ident = row['_id']

        if 'category_id' in row:
            category_id = row['category_id']
        else:
            category_id = -1

        count = 0
        pictures = []
        for i, idt in enumerate(ids):
            if(idt == ident):
                pictures.append(images[i].flatten())
                count = count + 1

        for i in range(count, 4):
            pictures.append(np.zeros(1))

        batchexamples.append(create_example(
            ident, category_id, count, pictures, length))

    return batchexamples


def create_example(ident, category_id, count, images, length):
    """Create example from values"""

    if(category_id != -1):
        line = categories.loc[categories['category_id'] == category_id]
        category_level3 = line.iloc[0]['category_level3']
        category_level2 = line.iloc[0]['category_level2']
        category_level1 = line.iloc[0]['category_level1']
    else:
        category_level3 = category_level2 = category_level1 = -1

    example = tf.train.Example(features=tf.train.Features(feature={
        'pid': _int64_feature(ident),
        'level3_category': _int64_feature(category_level3),
        'level2_category': _int64_feature(category_level2),
        'level1_category': _int64_feature(category_level1),
        'images': _int64_feature(count),
        'image_4': _array_feature(images.pop()),
        'image_3': _array_feature(images.pop()),
        'image_2': _array_feature(images.pop()),
        'image_1': _array_feature(images.pop()),
        'length': _int64_feature(length)}))

    return example


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_examples(examples, totest):
    """"Write examples in record"""

    if(totest == False):
        for j, example in enumerate(examples):
            if (j % 10 == 0):
                writer_validation.write(example.SerializeToString())
            else:
                writer_train.write(example.SerializeToString())

    elif(totest == True):
        for j, example in enumerate(examples):
            writer_test.write(example.SerializeToString())

    else:
        for j, example in enumerate(examples):
            print(example)
            exit()
# =================================================================================================


if __name__ == '__main__':

    # Arguments are needed for running at google cloudml
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath',
        help='GCS or local paths to training data',
        required=True
    )
    parser.add_argument(
        '--pretrainpath',
        help='GCS or local paths to pretrained data',
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
    pretrainpath = arguments.pop('pretrainpath')

    # Load data
    traindata = bson.decode_file_iter(file_io.FileIO(
        datapath + 'train_example.bson', mode='rb'))
    testdata = bson.decode_file_iter(file_io.FileIO(
        datapath + 'train_example.bson', mode='rb'))
    categories = pd.read_csv(file_io.FileIO(
        datapath + 'category_numbers.csv', mode='r'))

    if(USE_PRETRAINED == True):
        # Load network to calculate bottleneck values
        ckpt = tf.train.get_checkpoint_state(pretrainpath)
        saver = tf.train.import_meta_graph(
            ckpt.model_checkpoint_path + '.meta')
        sess = tf.Session()
        saver.restore(sess, ckpt.model_checkpoint_path)
        inputlayer = sess.graph.get_tensor_by_name("InputHolder:0")
        predlayer = sess.graph.get_tensor_by_name("final_layer/Mean:0")

    # Prepare TFRecordWriters
    filename_train = datapath + 'train-0000' + \
        str(SHARD_NUMBER) + '-of-00008.tfrecords'
    filename_validation = datapath + 'validation-0000' + \
        str(SHARD_NUMBER) + '-of-00008.tfrecords'
    filename_test = datapath + 'test-0000' + \
        str(SHARD_NUMBER) + '-of-00002.tfrecords'
    writer_train = tf.python_io.TFRecordWriter(filename_train)
    writer_validation = tf.python_io.TFRecordWriter(filename_validation)
    writer_test = tf.python_io.TFRecordWriter(filename_test)

    batchsize = 2048
    batchrows = []
    start = time.time()

    # Run through traindata and accumulate batches
    for i, row in enumerate(traindata):
        if(i > (SHARD_NUMBER + 1) * 1000000):
            # Stop when enough data for tfrecord shard
            break

        if(i > SHARD_NUMBER * 1000000):
            if(i % 100 == 0):
                print(i)

            batchrows.append(row)

            if(i % batchsize == batchsize - 1):
                examples = create_batch_examples(batchrows)
                write_examples(examples, False)
                while len(batchrows) > 0:
                    batchrows.pop()

                end = time.time()
                print("Batchdauer:", end - start)
                start = time.time()

    # Catch those last where batch not full
    examples = create_batch_examples(batchrows)
    write_examples(examples, False)
    while len(batchrows) > 0:
        batchrows.pop()

    # Run through testdata and accumulate batches
    for i, row in enumerate(testdata):
        if(i >= (SHARD_NUMBER + 1) * 1000000 or i >= 2000000):
            # Stop when enough data for tfrecord shard, only two for testset are needed
            break

        if(i >= SHARD_NUMBER * 1000000):
            if(i % 100 == 0):
                print(i)

            batchrows.append(row)

            if(i % batchsize == batchsize - 1):
                examples = create_batch_examples(batchrows)
                write_examples(examples, True)
                while len(batchrows) > 0:
                    batchrows.pop()

                end = time.time()
                print("Batchdauer:", end - start)
                start = time.time()

    # Catch those last where batch not full
    examples = create_batch_examples(batchrows)
    write_examples(examples, True)
    while len(batchrows) > 0:
        batchrows.pop()

    # Close writers
    writer_train.close()
    writer_validation.close()
    writer_test.close()
