import numpy as np
import tensorflow as tf

# =================================================================================================


def dataset_input_fn(filenames):
    dataset = tf.data.TFRecordDataset(filenames)

    # Parse single example
    def parser(record):
        keys_to_features = {
            'pid': tf.FixedLenFeature((), tf.int64, default_value=0),
            'category': tf.FixedLenFeature((), tf.int64, default_value=0),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        pid = [tf.cast(parsed["pid"], tf.int32)]
        category = [tf.cast(parsed["category"], tf.int32)]

        return pid, category

    dataset = dataset.map(parser)
    dataset = dataset.batch(1)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    return features, labels

# =================================================================================================


next_example, next_label = dataset_input_fn('data/prediction.tfrecord')

file = open('data/submission.csv', mode='wb')
file.write('_id,category_id'.encode('UTF-8'))
file.write('\n'.encode('UTF-8'))

with tf.Session() as sess:

    while True:
        try:
            # Write examples from record to csv
            pid, cat = sess.run([next_example, next_label])

            file.write(str(pid[0][0]).encode('UTF-8'))
            file.write(','.encode('UTF-8'))
            file.write(str(cat[0][0]).encode('UTF-8'))
            file.write('\n'.encode('UTF-8'))

        except tf.errors.OutOfRangeError:
            print("run finished")
            break
