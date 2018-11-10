#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
import time
from distutils.version import LooseVersion
import project_tests as tests
import argparse
from moviepy.editor import VideoFileClip
import numpy as np
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('action',
                    help='which action do you want to run? (train, images, video, all)', nargs='?', default='all')
parser.add_argument('--epochs',
                    help='How many epochs to train?', type=int, default=20)
args = parser.parse_args()

print('Running action: {}, with {} epochs'.format(args.action, args.epochs))

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load the vgg tag
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # Get the default graph
    graph = tf.get_default_graph()

    # Get each tensor from the graph
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    l2r = tf.contrib.layers.l2_regularizer(1e-3)
    rni = tf.random_normal_initializer(stddev=1e-3)

    conv7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='SAME', kernel_regularizer=l2r,
                                 kernel_initializer=rni)
    # Upsample with deconvolution
    output = tf.layers.conv2d_transpose(conv7_1x1, num_classes, 4, 2, padding='SAME', kernel_regularizer=l2r,
                                        kernel_initializer=rni)

    conv4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='SAME', kernel_regularizer=l2r,
                                 kernel_initializer=rni)
    # Add the vgg layer 4 for the first skip-layer
    output = tf.add(output, conv4_1x1)
    # Upsample with deconvolution
    output = tf.layers.conv2d_transpose(output, num_classes, 4, 2, padding='SAME', kernel_regularizer=l2r,
                                        kernel_initializer=rni)

    conv3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='SAME', kernel_regularizer=l2r,
                                 kernel_initializer=rni)
    # Add the vgg layer 3 for the second skip-layer
    output = tf.add(output, conv3_1x1)
    # Upsample with deconvolution
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, padding='SAME', kernel_regularizer=l2r,
                                        kernel_initializer=rni)
    return output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    for epoch in range(epochs):
        start_time = time.time()
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                     feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 1e-4})
        print("Epoch: {}".format(epoch+1), " of {}".format(epochs))
        print("Loss: {:.3f}".format(loss))
        print("Duration: {}".format(time.time() - start_time))


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    epochs = args.epochs
    batch_size = 16

    correct_label = tf.placeholder(tf.float32, (None, None, None, num_classes))
    learning_rate = tf.placeholder(tf.float32)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # Note: see helper.augment_image

        # Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        final_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(final_layer, correct_label, learning_rate, num_classes)


        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()

        if args.action == 'train' or args.action == 'all':
            train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label,
                     keep_prob, learning_rate)

            save_path = saver.save(sess, os.path.join('saved_models', 'model_' + str(time.time()) + '.ckpt'))
            print("Saving model: {}".format(save_path))
        else:
            saver.restore(sess, tf.train.latest_checkpoint('./saved_models'))

        # Save inference data using helper.save_inference_samples
        if args.action == 'images' or args.action == 'all':
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video
        if args.action == 'video' or args.action == 'all':
            video_out_name = 'inference_video.mp4'
            print('Generating {}'.format(video_out_name))

            def save_inference_video(img):
                image = scipy.misc.imresize(img, image_shape)
                im_softmax = sess.run(
                    [tf.nn.softmax(logits)],
                    {keep_prob: 1.0, image_input: [image]})
                im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
                segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
                mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
                mask = scipy.misc.toimage(mask, mode="RGBA")
                street_im = scipy.misc.toimage(image)
                street_im.paste(mask, box=None, mask=mask)
                return np.array(street_im)

            clip = VideoFileClip('harder_challenge_video.mp4')
            new_clip = clip.fl_image(save_inference_video)
            print('Saving sample video to: {}'.format(video_out_name))
            new_clip.write_videofile(video_out_name, audio=False)

        sess.close()
        exit(0)


if __name__ == '__main__':
    run()
