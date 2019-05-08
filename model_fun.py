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


def create_model_exp(features, data_format, all_num_anchors_depth, num_classes, isTraining=True):

    with tf.variable_scope('ssd300', default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        backbone = ssd_net.VGG16Backbone(data_format)
        feature_layers = backbone.forward(features, training=isTraining)
        # print(feature_layers)
        location_pred, cls_pred = ssd_net.multibox_head(feature_layers,
                                                        num_classes,
                                                        all_num_anchors_depth,
                                                        data_format=data_format)

        if data_format == 'channels_first':
            cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
            location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]

        cls_pred = [tf.reshape(pred,
                               [tf.shape(pred)[0], pred.shape[1], pred.shape[2], all_num_anchors_depth[idx],
                                num_classes]) for (idx, pred) in enumerate(cls_pred)]
        location_pred = [
            tf.reshape(pred, [tf.shape(pred)[0], pred.shape[1], pred.shape[2], all_num_anchors_depth[idx], 4])
            for (idx, pred) in enumerate(location_pred)]

    return cls_pred, location_pred

def modified_smooth_l1(bbox_pred, bbox_targets, bbox_inside_weights=1., bbox_outside_weights=1., sigma=1.):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    """
    with tf.name_scope('smooth_l1', [bbox_pred, bbox_targets]):
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul


def get_losses(features, cls_pred, location_pred,
               cls_targets,loc_targets, FLAGS):

    cls_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, FLAGS.num_classes]) for pred in cls_pred]
    location_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, 4]) for pred in location_pred]

    cls_pred = tf.concat(cls_pred, axis=1)
    location_pred = tf.concat(location_pred, axis=1)

    cls_pred = tf.reshape(cls_pred, [-1, FLAGS.num_classes])
    location_pred = tf.reshape(location_pred, [-1, 4])

    with tf.device('/cpu:0'):
        with tf.control_dependencies([cls_pred, location_pred]):
            with tf.name_scope('post_forward'):
                cls_targets = [tf.reshape(pred, [features.shape[0], -1]) for pred in cls_targets]
                loc_targets = [tf.reshape(pred, [features.shape[0], -1, 4]) for pred in loc_targets]

                cls_targets = tf.concat(cls_targets, axis=1)
                loc_targets = tf.concat(loc_targets, axis=1)

                flaten_cls_targets = tf.reshape(cls_targets, [-1])
                flaten_loc_targets = tf.reshape(loc_targets, [-1, 4])

                # each positive examples has one label
                positive_mask = flaten_cls_targets > 0

                batch_n_positives = tf.count_nonzero(cls_targets, -1)

                batch_negtive_mask = tf.equal(cls_targets,
                                              0)  # tf.logical_and(tf.equal(cls_targets, 0), match_scores > 0.)
                batch_n_negtives = tf.count_nonzero(batch_negtive_mask, -1)

                batch_n_neg_select = tf.cast(FLAGS.negative_ratio * tf.cast(batch_n_positives, tf.float32),
                                             tf.int32)
                batch_n_neg_select = tf.minimum(batch_n_neg_select, tf.cast(batch_n_negtives, tf.int32))

                # hard negative mining for classification
                predictions_for_bg = tf.nn.softmax(
                    tf.reshape(cls_pred, [tf.shape(features)[0], -1, FLAGS.num_classes]))[:, :, 0]
                prob_for_negtives = tf.where(batch_negtive_mask,
                                             0. - predictions_for_bg,
                                             # ignore all the positives
                                             0. - tf.ones_like(predictions_for_bg))
                topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=tf.shape(prob_for_negtives)[1])
                score_at_k = tf.gather_nd(topk_prob_for_bg,
                                          tf.stack([tf.range(tf.shape(features)[0]), batch_n_neg_select - 1], axis=-1))

                selected_neg_mask = prob_for_negtives >= tf.expand_dims(score_at_k, axis=-1)

                # include both selected negtive and all positive examples
                final_mask = tf.stop_gradient(
                    tf.logical_or(tf.reshape(tf.logical_and(batch_negtive_mask, selected_neg_mask), [-1]),
                                  positive_mask))
                #
                total_examples = tf.count_nonzero(final_mask)
                tf.summary.scalar('total_examples', total_examples)
                cls_pred = tf.boolean_mask(cls_pred, final_mask)
                location_pred = tf.boolean_mask(location_pred, tf.stop_gradient(positive_mask))
                flaten_cls_targets = tf.boolean_mask(tf.clip_by_value(flaten_cls_targets, 0, FLAGS.num_classes),
                                                     final_mask)
                flaten_loc_targets = tf.stop_gradient(tf.boolean_mask(flaten_loc_targets, positive_mask))


    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    # cross_entropy = tf.cond(n_positives > 0, lambda: tf.losses.sparse_softmax_cross_entropy(labels=flaten_cls_targets, logits=cls_pred), lambda: 0.)# * (params['negative_ratio'] + 1.)
    # flaten_cls_targets=tf.Print(flaten_cls_targets, [flaten_loc_targets],summarize=50000)
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=flaten_cls_targets, logits=cls_pred) #* (params['negative_ratio'] + 1.)
    # Create a tensor named cross_entropy for logging purposes.
    #tf.identity(cross_entropy, name='cross_entropy_loss')
    #tf.summary.scalar('cross_entropy_loss', cross_entropy)

    # loc_loss = tf.cond(n_positives > 0, lambda: modified_smooth_l1(location_pred, tf.stop_gradient(flaten_loc_targets), sigma=1.), lambda: tf.zeros_like(location_pred))
    loc_loss = modified_smooth_l1(location_pred, flaten_loc_targets, sigma=1.)
    # loc_loss = modified_smooth_l1(location_pred, tf.stop_gradient(gtargets))
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=-1), name='location_loss')
    #tf.summary.scalar('location_loss', loc_loss)
    tf.losses.add_loss(loc_loss)

    l2_loss_vars = []
    for trainable_var in tf.trainable_variables():
        if '_bn' not in trainable_var.name:
            if 'conv4_3_scale' not in trainable_var.name:
                l2_loss_vars.append(tf.nn.l2_loss(trainable_var))
            else:
                l2_loss_vars.append(tf.nn.l2_loss(trainable_var) * 0.1)
    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    total_loss = tf.add(cross_entropy + loc_loss,
                        tf.multiply(FLAGS.weight_decay, tf.add_n(l2_loss_vars), name='l2_loss'), name='total_loss')
    tf.summary.scalar('total_loss', total_loss)

    return total_loss


def flatten(x):
    result = []
    for el in x:
        if isinstance(el, tuple):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def split_encoder(array, all_anchors):

    indcies = [0] + [int(all_anchors[i][0].shape[0]*all_anchors[i][0].shape[1]*all_anchors[i][2].shape[0]) for i in range(len(all_anchors))]

    array_first = []
    index = 0
    for i in range(len(indcies) - 1):
        array_first.append(array[index:(index+indcies[i+1]), ...])
        index += indcies[i+1]
    print(len(array.shape))
    if len(array.shape) > 1:
        return [tf.reshape(array_first[i], (all_anchors[i][0].shape[0], all_anchors[i][0].shape[1], all_anchors[i][2].shape[0], array_first[i].shape[-1])) for i in range(len(all_anchors))]
    else:
        return [tf.reshape(array_first[i], (all_anchors[i][0].shape[0], all_anchors[i][0].shape[1], all_anchors[i][2].shape[0])) for i in range(len(all_anchors))]


