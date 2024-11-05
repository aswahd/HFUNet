# fmt: off
import os
import sys

sys.path.insert(0, os.path.abspath("segmentation_models.pytorch"))
# fmt: on

from .build_model import build_model  # noqa
