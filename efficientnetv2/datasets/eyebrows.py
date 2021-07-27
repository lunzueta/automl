"""eyebrows dataset."""

import tensorflow_datasets as tfds

# TODO(eyebrows): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(eyebrows): BibTeX citation
_CITATION = """
"""


class Eyebrows(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for eyebrows dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(eyebrows): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(128, 128, 3)),
            'label': tfds.features.ClassLabel(names=['frown', 'neutral', 'rise']),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    extracted_path = dl_manager.extract('/host/data/eyebrows/eyebrows.zip')
    return {
        'train': self._generate_examples(path=extracted_path / 'train'),
        'val': self._generate_examples(path=extracted_path / 'val'),
    }

  def _generate_examples(self, path):
    """Generator of examples for each split."""
    for img_path in path.glob('*.png'):
      # Yields (key, example)
      yield img_path.name, {
          'image': img_path,
          'label': 'frown' if img_path.name.startswith('frown_') \
            else ('neutral' if img_path.name.startswith('neutral_') else 'rise'),
      }
