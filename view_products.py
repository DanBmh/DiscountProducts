import numpy as np
import cv2 as cv
import bson
import tensorflow as tf

# =================================================================================================

data = bson.decode_file_iter(open('data/train_example.bson', 'rb'))

for c, d in enumerate(data):

    for e, pic in enumerate(d['imgs']):
        nparr = np.fromstring(pic['picture'], np.uint8)
        picture = cv.imdecode(nparr, cv.IMREAD_COLOR)

        # picture = cv.resize(picture,(90,90),cv.INTER_AREA)
        # picture = cv.resize(picture, (360, 360), cv.INTER_CUBIC)

        cv.imshow("test", picture)
        cv.waitKey()

    if(c > 4):
        exit()
