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

from typing import Any

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np

from paltax import utils
from paltax import train, train_snpe, train_nf


FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'working directory.')
flags.DEFINE_string(
    'target_image_path', None,
    'path to the target image. Only required for SNPE.'
)
flags.DEFINE_multi_string(
    'log_prob_batches_path', None,
    'List of paths to batches whose log probability will be logged.'
)
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training configuration.',
)


def main(_: Any):
    """Train neural network model with configuration defined by flags."""
    config = FLAGS.config
    # The training configuration will tell us what configuration we want to
    # use to generate images.
    input_config = train._get_config(config.input_config_path)
    rng = jax.random.PRNGKey(config.get('rng_key',0))

    if config.get('num_unique_batches', 0) > 0:
        if not (config.train_type == 'NPE'):
            raise ValueError('Cannot do finite batches with sequential.')
        rng_list = jax.random.split(rng, config.get('num_unique_batches'))
        rng = utils.random_permutation_iterator(rng_list, rng)

    # Load any batches we want to track during training.
    log_prob_batches = None
    if FLAGS.log_prob_batches_path is not None:
        log_prob_batches = {}
        for path in FLAGS.log_prob_batches:
            batch = np.load(path, allow_pickle=True).item()
            log_prob_batches[batch['name']] = {
                'truth': batch['truth'],
                'image': batch['image']
            }

    if config.train_type == 'NPE':
        train.train_and_evaluate(config, input_config, FLAGS.workdir, rng)
    elif config.train_type == 'SNPE':
        target_image = jnp.load(FLAGS.target_image_path)
        train_snpe.train_and_evaluate_snpe(
            config, input_config, FLAGS.workdir, rng, target_image
        )
    elif config.train_type == 'NF':
        target_image = jnp.load(FLAGS.target_image_path)
        train_nf.train_and_evaluate_nf(
            config, input_config, FLAGS.workdir, rng, target_image,
            log_prob_batches=log_prob_batches
        )
    else:
        raise ValueError(
            f'train_type {config.train_type} not a valid training configuration'
        )


if __name__ == '__main__':
    app.run(main)
