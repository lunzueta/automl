"""mouth dataset."""

import tensorflow_datasets as tfds

# TODO(mouth): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(mouth): BibTeX citation
_CITATION = """
"""


class Mouth(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for mouth dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(mouth): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(128, 128, 3)),
            'label': tfds.features.ClassLabel(names=['bite_lower_lip', \
                                                     'bite_upper_lip', \
                                                     'neutral', \
                                                     'kiss_left', \
                                                     'kiss_right', \
                                                     'kiss', \
                                                     'open', \
                                                     'press_lips']),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    extracted_path = dl_manager.extract('/host/data/mouth/mouth.zip')
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
          'label': 'bite_lower_lip' if img_path.name.startswith('bite_lower_lip_') \
            else ('bite_upper_lip' if img_path.name.startswith('bite_upper_lip_') \
            else ('neutral' if img_path.name.startswith('neutral_') \
            else ('kiss_left' if img_path.name.startswith('kiss_left_') \
            else ('kiss_right' if img_path.name.startswith('kiss_right_') \
            else ('kiss' if img_path.name.startswith('kiss_') \
            else ('open' if img_path.name.startswith('open_') \
            else ('press_lips'))))))),
      }
