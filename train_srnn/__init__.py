"""SRNN training package."""

from train_srnn import config as _config
from train_srnn import paths as _paths

_paths.register_resolvers()
_config.register_configs()
