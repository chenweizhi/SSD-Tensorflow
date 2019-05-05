import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
from model_fun import create_model_exp

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)
# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
# with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
#     predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
all_num_anchors_depth = [4,6,6,6,4,4]
data_format = 'channels_last'

cls_pred, location_pred = create_model_exp(image_4d, data_format, all_num_anchors_depth, 21, False)
prediction = [tf.nn.softmax(pred) for pred in cls_pred]

# Restore SSD model.
#ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
ckpt_filename = '/home/ai/DataDisk/wayze/tensorflow/ssd-tensorflow-exp/piecewise_new_preprocess/model.ckpt-90000'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, prediction, location_pred, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes



# Test on some demo image and visualize output.
# 测试的文件夹
path = '/home/ai/DataDisk/wayze/tensorflow/voc2007/test/'
result_folder = '/home/ai/DataDisk/wayze/tensorflow/voc2007/result/'
image_names = sorted(os.listdir(path))

colors = []
for i in range(21):
    colors.append([random.random()*255, random.random()*255, random.random()*255])

class_names = ['none',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
    'total'];

for i in range(512):
    # 文件夹中的第几张图，-1代表最后一张
    img = cv2.imread(path + image_names[i])
    rclasses, rscores, rbboxes = process_image(img)
    visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, colors, class_names)
    cv2.imwrite(os.path.join(result_folder,image_names[i]), img)
    # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
    #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

