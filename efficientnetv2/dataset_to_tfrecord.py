# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

r"""Script to convert a custom Imagenet-like dataset into TFRecords.
Prepare the dataset files in the following format:
- Training images: train/classname/exampleX.jpg
- Validation Images: val/classname/exampleY.jpg
To run the script to convert the raw dataset into TFRecords,
run the following command:
```
python3 dataset_to_tfrecord.py \
  --raw_data_dir="path/to/dataset"
```
"""

import math
import os
import random
from typing import Iterable, List, Mapping, Union, Tuple
from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

flags.DEFINE_string(
    'local_scratch_dir', None, 'Scratch directory path for temporary files.')
flags.DEFINE_string(
    'raw_data_dir', None, 'Directory path for raw dataset. '
    'Should have train and validation subdirectories inside it.')

FLAGS = flags.FLAGS

TRAINING_SHARDS = 1024
VALIDATION_SHARDS = 128

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'val'


def _check_or_create_dir(directory: str):
  """Checks if directory exists otherwise creates it."""
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)


def _int64_feature(value: Union[int, Iterable[int]]) -> tf.train.Feature:
  """Inserts int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value: Union[bytes, str]) -> tf.train.Feature:
  """Inserts bytes features into Example proto."""
  if isinstance(value, str):
    value = bytes(value, 'utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename: str,
                        image_buffer: str,
                        label: int,
                        synset: str,
                        height: int,
                        width: int) -> tf.train.Example:
  """Builds an Example proto for an dataset example.
  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/synset': _bytes_feature(synset),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def decode_jpeg(self, image_data: bytes) -> tf.Tensor:
    """Decodes a JPEG image."""
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _process_image(
    filename: str, coder: ImageCoder) -> Tuple[str, int, int]:
  """Processes a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(
    coder: ImageCoder,
    output_file: str,
    filenames: Iterable[str],
    synsets: Iterable[Union[str, bytes]],
    labels: Mapping[str, int]):
  """Processes and saves a list of images as TFRecords.
  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    output_file: string, unique identifier specifying the data set.
    filenames: list of strings; each string is a path to an image file.
    synsets: list of strings; each string is a unique WordNet ID.
    labels: map of string to integer; id for all synset labels.
  """
  writer = tf.python_io.TFRecordWriter(output_file)

  for filename, synset in zip(filenames, synsets):
    image_buffer, height, width = _process_image(filename, coder)
    label = labels[synset]
    example = _convert_to_example(filename, image_buffer, label,
                                  synset, height, width)
    writer.write(example.SerializeToString())

  writer.close()


def _process_dataset(
    filenames: Iterable[str],
    synsets: Iterable[str],
    labels: Mapping[str, int],
    output_directory: str,
    prefix: str,
    num_shards: int) -> List[str]:
  """Processes and saves list of images as TFRecords.
  Args:
    filenames: iterable of strings; each string is a path to an image file.
    synsets: iterable of strings; each string is a unique WordNet ID.
    labels: map of string to integer; id for all synset labels.
    output_directory: path where output files should be created.
    prefix: string; prefix for each file.
    num_shards: number of chunks to split the filenames into.
  Returns:
    files: list of tf-record filepaths created from processing the dataset.
  """
  _check_or_create_dir(output_directory)
  chunksize = int(math.ceil(len(filenames) / num_shards))
  coder = ImageCoder()

  files = []

  for shard in range(num_shards):
    chunk_files = filenames[shard * chunksize : (shard + 1) * chunksize]
    chunk_synsets = synsets[shard * chunksize : (shard + 1) * chunksize]
    output_file = os.path.join(
        output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))
    _process_image_files_batch(coder, output_file, chunk_files,
                               chunk_synsets, labels)
    logging.info('Finished writing file: %s', output_file)
    files.append(output_file)
  return files


def convert_to_tf_records(
    raw_data_dir: str,
    local_scratch_dir: str) -> Tuple[List[str], List[str]]:
  """Converts the Imagenet-like dataset into TF-Record dumps."""

  # Shuffle training records to ensure we are distributing classes
  # across the batches.
  random.seed(0)
  def make_shuffle_idx(n):
    order = list(range(n))
    random.shuffle(order)
    return order

  # Glob all the training files
  training_files = tf.gfile.Glob(
      os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*', '*.jpg'))

  # Get training file synset labels from the directory name
  training_synsets = [
      os.path.basename(os.path.dirname(f)) for f in training_files]
  training_synsets = list(map(lambda x: bytes(x, 'utf-8'), training_synsets))

  training_shuffle_idx = make_shuffle_idx(len(training_files))
  training_files = [training_files[i] for i in training_shuffle_idx]
  training_synsets = [training_synsets[i] for i in training_shuffle_idx]

  # Glob all the validation files
  validation_files = sorted(tf.gfile.Glob(
      os.path.join(raw_data_dir, VALIDATION_DIRECTORY, '*', '*.jpg')))

  # Get training file synset labels from the directory name
  validation_synsets = [
      os.path.basename(os.path.dirname(f)) for f in validation_files]
  validation_synsets = \
    list(map(lambda x: bytes(x, 'utf-8'), validation_synsets))

  validation_shuffle_idx = make_shuffle_idx(len(validation_files))
  validation_files = [validation_files[i] for i in validation_shuffle_idx]
  validation_synsets = [validation_synsets[i] for i in validation_shuffle_idx]

  # Create unique ids for all synsets
  labels = {v: k + 1 for k, v in enumerate(
      sorted(set(validation_synsets + training_synsets)))}

  # Create training data
  logging.info('Processing the training data.')
  training_records = _process_dataset(
      training_files, training_synsets, labels,
      os.path.join(local_scratch_dir, TRAINING_DIRECTORY),
      TRAINING_DIRECTORY, TRAINING_SHARDS)

  # Create validation data
  logging.info('Processing the validation data.')
  validation_records = _process_dataset(
      validation_files, validation_synsets, labels,
      os.path.join(local_scratch_dir, VALIDATION_DIRECTORY),
      VALIDATION_DIRECTORY, VALIDATION_SHARDS)

  return training_records, validation_records

def run(raw_data_dir: str,
        local_scratch_dir: str):
  """Runs the ImageNet-like data conversion into TFRecords.
  Args:
    raw_data_dir: str, the path to the folder with raw ImageNet-like data.
    local_scratch_dir: str, the local directory path.
  """
  if raw_data_dir is None:
    raise AssertionError(
        'Please, provide the `raw_data_dir`.')

  # Convert the raw data into tf-records
  training_records, validation_records = convert_to_tf_records(
      raw_data_dir=raw_data_dir,
      local_scratch_dir=local_scratch_dir)


def main(_):
  run(raw_data_dir=FLAGS.raw_data_dir,
      local_scratch_dir=FLAGS.local_scratch_dir)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.disable_v2_behavior()
  app.run(main)
