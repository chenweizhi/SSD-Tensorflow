import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from nets import ssd_net
import tf_utils
from utility import scaffolds

slim = tf.contrib.slim

DATA_FORMAT = 'NCHW'



def create_model_exp(features, data_format, all_num_anchors_depth, num_classes):

    with tf.variable_scope('ssd300', default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        backbone = ssd_net.VGG16Backbone(data_format)
        feature_layers = backbone.forward(features, training=True)
        # print(feature_layers)
        location_pred, cls_pred = ssd_net.multibox_head(feature_layers,
                                                        num_classes,
                                                        all_num_anchors_depth,
                                                        data_format=data_format)

        if data_format == 'channels_first':
            cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
            location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]

        cls_pred = [tf.reshape(pred,
                               [tf.shape(pred)[0], tf.shape(pred)[1], tf.shape(pred)[2], all_num_anchors_depth[idx],
                                num_classes]) for (idx, pred) in enumerate(cls_pred)]
        location_pred = [
            tf.reshape(pred, [tf.shape(pred)[0], tf.shape(pred)[1], tf.shape(pred)[2], all_num_anchors_depth[idx], 4])
            for (idx, pred) in enumerate(location_pred)]

    return cls_pred, location_pred
