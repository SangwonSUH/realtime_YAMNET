# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Core model definition of YAMNet."""

import csv

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

import yamnet.features as features_lib
import yamnet.params as params


def _batch_norm(name):
  def _bn_layer(layer_input):
    return layers.BatchNormalization(
      name=name,
      center=params.BATCHNORM_CENTER,
      scale=params.BATCHNORM_SCALE,
      epsilon=params.BATCHNORM_EPSILON)(layer_input)
  return _bn_layer


def _conv(name, kernel, stride, filters):
  def _conv_layer(layer_input):
    output = layers.Conv2D(name='{}/conv'.format(name),
                           filters=filters,
                           kernel_size=kernel,
                           strides=stride,
                           padding=params.CONV_PADDING,
                           use_bias=False,
                           activation=None)(layer_input)
    output = _batch_norm(name='{}/conv/bn'.format(name))(output)
    output = layers.ReLU(name='{}/relu'.format(name))(output)
    return output
  return _conv_layer


def _separable_conv(name, kernel, stride, filters):
  def _separable_conv_layer(layer_input):
    output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
                                    kernel_size=kernel,
                                    strides=stride,
                                    depth_multiplier=1,
                                    padding=params.CONV_PADDING,
                                    use_bias=False,
                                    activation=None)(layer_input)
    output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
    output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
    output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
                           filters=filters,
                           kernel_size=(1, 1),
                           strides=1,
                           padding=params.CONV_PADDING,
                           use_bias=False,
                           activation=None)(output)
    output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
    output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)
    return output
  return _separable_conv_layer


_YAMNET_LAYER_DEFS = [
    # (layer_function, kernel, stride, num_filters)
    (_conv,          [3, 3], 2,   32),
    (_separable_conv, [3, 3], 1,   64),
    (_separable_conv, [3, 3], 2,  128),
    (_separable_conv, [3, 3], 1,  128),
    (_separable_conv, [3, 3], 2,  256),
    (_separable_conv, [3, 3], 1,  256),
    (_separable_conv, [3, 3], 2,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 2, 1024),
    (_separable_conv, [3, 3], 1, 1024)
]


def yamnet(features):
  """Define the core YAMNet mode in Keras."""
  net = layers.Reshape(
    (params.PATCH_FRAMES, params.PATCH_BANDS, 1),
    input_shape=(params.PATCH_FRAMES, params.PATCH_BANDS))(features)
  for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
    net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters)(net)
  net = layers.GlobalAveragePooling2D()(net)
  logits = layers.Dense(units=params.NUM_CLASSES, use_bias=True)(net)
  predictions = layers.Activation(
    name=params.EXAMPLE_PREDICTIONS_LAYER_NAME,
    activation=params.CLASSIFIER_ACTIVATION)(logits)
  return predictions


def yamnet_frames_model(feature_params):
  """Defines the YAMNet waveform-to-class-scores model.

  Args:
    feature_params: An object with parameter fields to control the feature
    calculation.

  Returns:
    A model accepting (1, num_samples) waveform input and emitting a
    (num_patches, num_classes) matrix of class scores per time frame as
    well as a (num_spectrogram_frames, num_mel_bins) spectrogram feature
    matrix.
  """
  waveform = layers.Input(batch_shape=(1, None))
  # Store the intermediate spectrogram features to use in visualization.
  spectrogram = features_lib.waveform_to_log_mel_spectrogram(
    tf.squeeze(waveform, axis=0), feature_params)
  patches = features_lib.spectrogram_to_patches(spectrogram, feature_params)
  predictions = yamnet(patches)
  frames_model = Model(name='yamnet_frames', 
                       inputs=waveform, outputs=[predictions, spectrogram])
  return frames_model


def class_names(class_map_csv):
  """Read the class name definition file and return a list of strings."""
  with open(class_map_csv) as csv_file:
    reader = csv.reader(csv_file)
    next(reader)   # Skip header
    return np.array([display_name for (_, _, display_name) in reader])
