# DiscountProducts
Transfer learning with NasNet for Kaggle Competition on product classification

In this competition (https://www.kaggle.com/c/cdiscount-image-classification-challenge) there where 5270 classes for ~15M images with 180x180 resolution (~70GB).


#### About project
My approch used nasnet (https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) which has beaten lots of different network architectures in Imagenet.

Due to time and resource limitations I am only using 90*90 pixel images and I am only training the last prediction layers of the network. To save time I created the bottleneck features (features of images one layer before prediction layer) and saved them to a tfrecord file. For all images this where about 250GB and needed ~40h on Nvidia Tesla K80.
But this saved a lot of time in training as one epoch needed only about 2h. 

The training scripts were executed on Google CloudML. You can see commands for this on top of the files as comments (dont forget to change project names). Training costed me about 100€ (you get 300$ when first signing up).


#### Using the scripts
To create nasnet checkpoints clone https://github.com/tensorflow/models and run _extra_create_nasnet.py_.

To create bottleneck files run _create_records.py_, to train run _main.py_, for prediction run _create_predictions.py_ (you can find all files in trainer folder).

To view single files from the testset run _view_products.py_
