# coding=utf-8

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main file for running paltax from command line."""

from importlib import import_module
import os
import sys
from typing import Any

from absl import app
from absl import flags
import jax
import jax.numpy as jnp

from paltax import utils
from paltax import train, train_snpe


FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'working directory.')
flags.DEFINE_string('config_path', None,
    'path to the training configuration.')
flags.DEFINE_string('target_image_path', None, 'path to the target image.')


def _get_config(config_path: str) -> Any:
    """Return config from provided path.

    Args:
        config_path: Path to configuration file.

    Returns:
        Loaded configuration file.
    """
    # Get the dictionary from the .py file.
    config_dir, config_file = os.path.split(os.path.abspath(config_path))
    sys.path.insert(0, config_dir)
    config_name, _ = os.path.splitext(config_file)
    config_module = import_module(config_name)
    return config_module.get_config()


def main(_: Any):
    """Train neural network model with configuration defined by flags."""
    config = _get_config(FLAGS.config_path)
    # The training configuration will tell us what configuration we want to
    # use to generate images.
    input_config = _get_config(config.input_config_path)
    rng = jax.random.PRNGKey(config.get('rng_key',0))

    if config.get('num_unique_batches',0) > 0:
        rng_list = jax.random.split(rng, FLAGS.num_unique_batches)
        rng = utils.random_permutation_iterator(rng_list, rng)

    if config.train_type == 'NPE':
        train.train_and_evaluate(config, input_config, FLAGS.workdir, rng)
    else:
        target_image = jnp.load(FLAGS.target_image_path)
        train_snpe.train_and_evaluate_snpe(
            config, input_config, FLAGS.workdir, rng, target_image
        )


if __name__ == '__main__':
    app.run(main)
