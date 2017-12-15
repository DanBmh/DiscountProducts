import numpy as np
import cv2 as cv
import bson
import tensorflow as tf

# =================================================================================================

data = bson.decode_file_iter(open('data/train_example.bson', 'rb'))

for index, row in enumerate(data):

    for i, ims in enumerate(row['imgs']):
        nparr = np.fromstring(ims['picture'], np.uint8)
        picture = cv.imdecode(nparr, cv.IMREAD_COLOR)

        # picture = cv.resize(picture,(90,90),cv.INTER_AREA)
        # picture = cv.resize(picture, (360, 360), cv.INTER_CUBIC)

        cv.imshow("test", picture)
        cv.waitKey()

    if(index > 4):
        exit()
