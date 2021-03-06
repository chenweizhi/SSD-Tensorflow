# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a SSD model using a given dataset."""
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import tf_utils
from utility import scaffolds
from model_fun import create_model_exp
from preprocessing import ssd_preprocessing
from utility import anchor_manipulator
from model_fun import split_encoder
import model_fun

slim = tf.contrib.slim

DATA_FORMAT = 'NCHW'

# =========================================================================== #
# SSD Network flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'loss_alpha', 1., 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold in the loss function.')

# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'ssd_300_vgg', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS




# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        # Config model_deploy. Keep TF Slim Models structure.
        # Useful if want to need multiple GPUs and/or servers in the future.
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=0,
            num_replicas=1,
            num_ps_tasks=0)
        # Create global_step.
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        # Select the dataset.
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        # Get the SSD network and its anchors.
        ssd_class = nets_factory.get_network(FLAGS.model_name)
        ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net_origin = ssd_class(ssd_params)
        ssd_shape = ssd_net_origin.params.img_shape
        ssd_anchors = ssd_net_origin.anchors(ssd_shape)

        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=True)

        tf_utils.print_configuration(FLAGS.__flags, ssd_params,
                                     dataset.data_sources, FLAGS.train_dir)


        out_shape = ssd_shape #[FLAGS.train_image_size] * 2
        anchor_creator = anchor_manipulator.AnchorCreator(out_shape,
                                                    layers_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                                                    anchor_scales = [(0.1,), (0.2,), (0.375,), (0.55,), (0.725,), (0.9,)],
                                                    extra_anchor_scales = [(0.1414,), (0.2739,), (0.4541,), (0.6315,), (0.8078,), (0.9836,)],
                                                    anchor_ratios = [(1., 2., .5), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., .5), (1., 2., .5)],
                                                    layer_steps = [8, 16, 32, 64, 100, 300])
        all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()

        num_anchors_per_layer = []
        for ind in range(len(all_anchors)):
            num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])

        anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(allowed_borders = [1.0] * 6,
                                                            positive_threshold = FLAGS.match_threshold,
                                                            ignore_threshold = 0.5, #FLAGS.neg_threshold,
                                                            prior_scaling=[0.1, 0.1, 0.2, 0.2])

        data_format = 'channels_first'

        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        with tf.device(deploy_config.inputs_device()):
            with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=FLAGS.num_readers,
                    common_queue_capacity=20 * FLAGS.batch_size,
                    common_queue_min=10 * FLAGS.batch_size,
                    shuffle=True)
            # Get for SSD network: image, labels, bboxes.
            [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                             'object/label',
                                                             'object/bbox'])
            # Pre-processing image, labels and bboxes.
            # image, glabels, gbboxes = \
            #     image_preprocessing_fn(image, glabels, gbboxes,
            #                            out_shape=ssd_shape,
            #                            data_format=DATA_FORMAT)
            out_shape = [ i for i in ssd_shape]
            image, glabels, gbboxes = ssd_preprocessing.preprocess_image(image,
                                                                         glabels,
                                                                         gbboxes,
                                                                         out_shape,
                                                                       is_training=True,
                                                                       data_format=data_format,
                                                                       output_rgb=True)



            # Encode groundtruth labels and bboxes.
            # gclasses, glocalisations, gscores = \
            #     ssd_net_origin.bboxes_encode(glabels, gbboxes, ssd_anchors)

            gt_targets, gt_labels, gt_scores = anchor_encoder_decoder.encode_all_anchors(glabels, gbboxes, all_anchors, all_num_anchors_depth,
                                                      all_num_anchors_spatial)

            gt_targets = split_encoder(gt_targets, all_anchors)
            gt_labels = split_encoder(gt_labels, all_anchors)
            gt_scores = split_encoder(gt_scores, all_anchors)

            batch_shape = [1] + [len(ssd_anchors)] * 3

            # Training batches and queue.
            # r = tf.train.batch(
            #     tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
            #     batch_size=FLAGS.batch_size,
            #     num_threads=FLAGS.num_preprocessing_threads,
            #     capacity=5 * FLAGS.batch_size)
            r = tf.train.batch(
                tf_utils.reshape_list([image, gt_labels, gt_targets, gt_scores]),
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)

            b_image, b_gclasses, b_glocalisations, b_gscores = \
                tf_utils.reshape_list(r, batch_shape)

            # Intermediate queueing: unique batch computation pipeline for all
            # GPUs running the training.
            batch_queue = slim.prefetch_queue.prefetch_queue(
                tf_utils.reshape_list([b_image, b_gclasses, b_glocalisations, b_gscores]),
                capacity=2 * deploy_config.num_clones)


        b_image, b_gclasses, b_glocalisations, b_gscores = \
                tf_utils.reshape_list(batch_queue.dequeue(), batch_shape)


        #new add
        all_num_anchors_depth = [len(ele[2]) for ele in ssd_anchors]

        # with tf.variable_scope('ssd300', default_name=None, values=[b_image], reuse=tf.AUTO_REUSE):
        #     backbone = ssd_net.VGG16Backbone(data_format)
        #     feature_layers = backbone.forward(b_image, training=True)
        #     # print(feature_layers)
        #     location_pred, cls_pred = ssd_net.multibox_head(feature_layers,
        #                                                     FLAGS.num_classes,
        #                                                     all_num_anchors_depth,
        #                                                     data_format=data_format)
        #
        #     if data_format == 'channels_first':
        #         cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
        #         location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]
        #
        #     cls_pred = [tf.reshape(pred, [tf.shape(pred)[0],tf.shape(pred)[1],tf.shape(pred)[2],  all_num_anchors_depth[idx], FLAGS.num_classes]) for (idx, pred) in enumerate(cls_pred)]
        #     location_pred = [tf.reshape(pred, [tf.shape(pred)[0],tf.shape(pred)[1],tf.shape(pred)[2], all_num_anchors_depth[idx], 4]) for (idx, pred) in enumerate(location_pred)]
        cls_pred, location_pred = create_model_exp(b_image, data_format, all_num_anchors_depth, FLAGS.num_classes)

        total_loss = model_fun.get_losses(b_image, cls_pred, location_pred, b_gclasses, b_glocalisations, FLAGS)

        # ssd_net_origin.losses(cls_pred, location_pred,
        #                       b_gclasses, b_glocalisations, b_gscores,
        #                       match_threshold=FLAGS.match_threshold,
        #                       negative_ratio=FLAGS.negative_ratio,
        #                       alpha=FLAGS.loss_alpha,
        #                       label_smoothing=FLAGS.label_smoothing)
        # =================================================================== #
        # Define the model running on every GPU.
        # =================================================================== #
        # def clone_fn(batch_queue):
        #     """Allows data parallelism by creating multiple
        #     clones of network_fn."""
        #     # Dequeue batch.
        #
        #
        #     # Construct SSD network.
        #     arg_scope = ssd_net_origin.arg_scope(weight_decay=FLAGS.weight_decay,
        #                                   data_format=DATA_FORMAT)
        #     with slim.arg_scope(arg_scope):
        #         predictions, localisations, logits, end_points = \
        #             ssd_net_origin.net(b_image, is_training=True)
        #     # Add loss function.
        #     ssd_net_origin.losses(logits, localisations,
        #                    b_gclasses, b_glocalisations, b_gscores,
        #                    match_threshold=FLAGS.match_threshold,
        #                    negative_ratio=FLAGS.negative_ratio,
        #                    alpha=FLAGS.loss_alpha,
        #                    label_smoothing=FLAGS.label_smoothing)
        #     return end_points
        #
        # # Gather initial summaries.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        #
        # # =================================================================== #
        # # Add summaries from first clone.
        # # =================================================================== #
        # clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        # first_clone_scope = deploy_config.clone_scope(0)
        # # Gather update_ops from the first clone. These contain, for example,
        # # the updates for the batch_norm variables created by network_fn.
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
        #
        # # Add summaries for end_points.
        # end_points = clones[0].outputs
        # for end_point in end_points:
        #     x = end_points[end_point]
        #     summaries.add(tf.summary.histogram('activations/' + end_point, x))
        #     summaries.add(tf.summary.scalar('sparsity/' + end_point,
        #                                     tf.nn.zero_fraction(x)))
        # Add summaries for losses and extra losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar(loss.op.name, loss))
        for loss in tf.get_collection('EXTRA_LOSSES'):
            summaries.add(tf.summary.scalar(loss.op.name, loss))
        #
        # # Add summaries for variables.
        # for variable in slim.get_model_variables():
        #     summaries.add(tf.summary.histogram(variable.op.name, variable))


        # =================================================================== #
        # Configure the moving averages.
        # =================================================================== #
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        # =================================================================== #
        # Configure the optimization procedure.
        # =================================================================== #
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = tf_utils.configure_learning_rate(FLAGS,
                                                             dataset.num_samples,
                                                             global_step)
            optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = tf_utils.get_variables_to_train(FLAGS)

        # losses = tf.get_collection(tf.GraphKeys.LOSSES)
        # total_loss = tf.add_n(losses)

        grads = optimizer.compute_gradients(total_loss)

        # and returns a train_tensor and summary_op
        # total_loss, clones_gradients = model_deploy.optimize_clones(
        #     clones,
        #     optimizer,
        #     var_list=variables_to_train)
        # # Add total_loss to summary.
        # summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(grads,
                                                 global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                          name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
        #                                    first_clone_scope))
        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # =================================================================== #
        # Kicks off the training.
        # =================================================================== #
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False,
                                gpu_options=gpu_options)
        saver = tf.train.Saver(max_to_keep=5,
                               keep_checkpoint_every_n_hours=1.0,
                               write_version=2,
                               pad_step_number=False)

        init_func = scaffolds.get_init_fn_for_scaffold(FLAGS.train_dir, "model/",
                                           'ssd300', 'vgg_16',
                                           'ssd300/multibox_head, ssd300/additional_layers, ssd300/conv4_3_scale', True,
                                           name_remap={'/kernel': '/weights', '/bias': '/biases'})

        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_dir,
            master='',
            is_chief=True,
            init_fn=init_func, #tf_utils.get_init_fn(FLAGS),
            summary_op=summary_op,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            saver=saver,
            save_interval_secs=FLAGS.save_interval_secs,
            session_config=config,
            sync_optimizer=None)


if __name__ == '__main__':
    tf.app.run()
