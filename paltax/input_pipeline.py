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
"""Code for drawing batches of example images used to train a neural
network.
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import jax

from paltax import cosmology_utils
from paltax import los
from paltax import subhalos
from paltax import image_simulation
from paltax import utils


NUM_NORMAL_DISTRIBUTIONS = 100
UNIFORM_ENCODING_START = 0
CONSTANT_ENCODING_START = 3
NORMAL_ENCODING_START = 4


def _generate_blank_encoding() -> jnp.ndarray:
    """Allocate a blank encoding to be used by the encode functions.

    Returns:
        Encoding with all values set to zero.

    Notes:
        The number of normal distributions is a global variable defined in the
        file.
    """
    # Number of parameters for uniform
    total_size = 3
    # Number of parameters for constant
    total_size += 1
    # Number of paramters for mixture of normal distributions
    total_size += 3 * NUM_NORMAL_DISTRIBUTIONS

    return jnp.zeros(total_size)


def _encode_normal(encoding: jnp.ndarray, mean: float,
                   std: float, weight: float) -> jnp.ndarray:
    """Update encoding to include a normal distribution.

    Args:
        encoding: Current encoding without a mean (should be either zeros or
            this should be called within add_normal_to_encoding).
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
        weight: Weight of the normal. Should be 1.0 if it is the only normal
            in the mixture.

    Returns:
        Updated encoding that represents a normal with given mean and
        standard deviation.
    """
    encoding = encoding.at[NORMAL_ENCODING_START].set(weight)
    encoding = encoding.at[NORMAL_ENCODING_START+1].set(mean)
    encoding = encoding.at[NORMAL_ENCODING_START+2].set(std)
    return encoding


def encode_normal(mean: float, std: float) -> jnp.ndarray:
    """Generate the jax array that encodes a normal distribution.

    Args:
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.

    Returns:
        Encoding that represents a normal with given mean and standard
        deviation.
    """
    # Encoding is currently [uniform: 3, constant: 1, normals:
    # 3 x NUM_NORMAL_DISTRIBUTIONS]. Each normal is encoded by [weight, mean,
    # std], so here we set 1 normal to have weight 1.0 and the given mean
    # and standard deviation.
    encoding = _generate_blank_encoding()
    return _encode_normal(encoding, mean, std, 1.0)


def encode_uniform(minimum: float, maximum: float) -> jnp.ndarray:
    """Generate the jax array that encodes a uniform distribution.

    Args:
        min: Minimum value of the uniform distribution.
        max: Maximum value of the uniform distribution.

    Returns:
        Encoding that represents a uniform with given min and max.
    """
    # Encoding is currently [uniform: 3, constant: 1, normals:
    # 3 x NUM_NORMAL_DISTRIBUTIONS]. The uniform parameters are [flag, minimum,
    # maximum].
    encoding = _generate_blank_encoding()
    encoding = encoding.at[UNIFORM_ENCODING_START].set(1.0)
    encoding = encoding.at[UNIFORM_ENCODING_START + 1].set(minimum)
    encoding = encoding.at[UNIFORM_ENCODING_START + 2].set(maximum)
    return encoding


def encode_constant(constant: float) -> jnp.ndarray:
    """Generate the jax array that encodes a constant value.

    Args:
        constant: Constant value.

    Returns:
        Encoding that represents a constant value.
    """
    # Encoding is currently [uniform: 3, constant: 1, normals:
    # 3 x NUM_NORMAL_DISTRIBUTIONS].
    encoding = _generate_blank_encoding()
    encoding = encoding.at[CONSTANT_ENCODING_START].set(constant)
    return encoding


def _get_normal_mean_indices() -> jnp.ndarray:
    """Get the indices for the mean values of the normal distributions.

    Returns:
        Indices for mean values of the normal distribution.
    """
    mean_indices = jnp.arange(NUM_NORMAL_DISTRIBUTIONS) * 3
    mean_indices += NORMAL_ENCODING_START + 1
    return mean_indices


def _get_normal_mean(encoding: jnp.ndarray) -> jnp.ndarray:
    """Return the array of mean values for the normal distributions

    Args:
        encoding: Encoded distribution.

    Returns:
        Mean for each normal in the encoded distribution.
    """
    # Three parameters per normal encoding and mean is the second parameter.
    mean_indices = _get_normal_mean_indices()
    return encoding[mean_indices]


def _get_normal_std_indices() -> jnp.ndarray:
    """Get the indices for the std values of the normal distributions.

    Returns:
        Indices for std values of the normal distribution.
    """
    std_indices = jnp.arange(NUM_NORMAL_DISTRIBUTIONS) * 3
    std_indices += NORMAL_ENCODING_START + 2
    return std_indices


def _get_normal_std(encoding: jnp.ndarray) -> jnp.ndarray:
    """Return the array of std values for the normal distributions

    Args:
        encoding: Encoded distribution.

    Returns:
        Standard deviation for each normal in the encoded distribution.
    """
    # Three parameters per normal encoding and mean is the third parameter.
    std_indices = _get_normal_std_indices()
    return encoding[std_indices]


def _get_normal_weights_indices() -> jnp.ndarray:
    """Get the indices for the weight values of the normal distributions.

    Returns:
        Indices for weight values of the normal distribution.
    """
    weight_indices = jnp.arange(NUM_NORMAL_DISTRIBUTIONS) * 3
    weight_indices += NORMAL_ENCODING_START
    return weight_indices


def _get_normal_weights(encoding: jnp.ndarray) -> jnp.ndarray:
    """Return the array of weight values for the normal distributions.

    Args:
        encoding: Encoded distribution.

    Returns:
        Weight of each normal in the encoded distribution.
    """
    # Three parameters per normal encoding and mean is the first parameter.
    weight_indices = _get_normal_weights_indices()
    return encoding[weight_indices]


def add_normal_to_encoding(encoding: jnp.ndarray, mean: float, std: float,
                           decay_factor: float) -> jnp.ndarray:
    """Add a new normal distribution to the mixture.

    Args:
        encoding: Current encoded distribution.
        mean: Mean of new normal distribution.
        std: Standard deviation of new normal distribution.
        decay_factor: Decay factor to apply on other distribution weights when
            adding the new distribution.

    Returns:
        Encoding for new mixture of Gaussian with appropriate decay factor
        applied.
    """
    cur_weights = _get_normal_weights(encoding)[:-1]
    cur_means = _get_normal_mean(encoding)[:-1]
    cur_std = _get_normal_std(encoding)[:-1]
    weight_indices = _get_normal_weights_indices()[1:]
    mean_indices = _get_normal_mean_indices()[1:]
    std_indices = _get_normal_std_indices()[1:]

    # Add Gaussian to the mixture following a queue mentality.
    # Normalization needs to factor in a previous Gaussian having been dropped.
    normalization = decay_factor / jnp.sum(cur_weights)
    encoding = encoding.at[weight_indices].set(cur_weights * normalization)
    encoding = encoding.at[mean_indices].set(cur_means)
    encoding = encoding.at[std_indices].set(cur_std)

    return _encode_normal(encoding, mean, std, 1 - decay_factor)


def average_normal_to_encoding(
        encoding: jnp.ndarray, mean: float, std: float
) -> jnp.ndarray:
    """Average a new normal distribution into the mixture.

    Args:
        encoding: Current encoded distribution.
        mean: Mean of new normal distribution.
        std: Standard deviation of new normal distribution.

    Returns:
        Encoding for new mixture of N Gaussian where each Gaussian has weight
        1/N.
    """
    cur_weights = _get_normal_weights(encoding)[:-1]
    cur_means = _get_normal_mean(encoding)[:-1]
    cur_std = _get_normal_std(encoding)[:-1]
    weight_indices = _get_normal_weights_indices()[1:]
    mean_indices = _get_normal_mean_indices()[1:]
    std_indices = _get_normal_std_indices()[1:]

    # Add Gaussian to the mixture following a queue mentality.
    # Normalization needs to factor in a previous Gaussian having been dropped.
    normalization = 1/(jnp.sum(cur_weights > 0.0) + 1)
    encoding = encoding.at[weight_indices].set(
        (cur_weights > 0.0) * normalization
    )
    encoding = encoding.at[mean_indices].set(cur_means)
    encoding = encoding.at[std_indices].set(cur_std)

    return _encode_normal(encoding, mean, std, normalization)


def decode_maximum(encoding: jnp.ndarray) -> float:
    """Decode the maximum value of the distribution defined by encoding.

    Args:
        encoding: Encoded distribution.

    Returns:
        Maximum value of encoded distribution. May be an approximation for
        distributions without a maximum.
    """
    # Encoding is currently [uniform: 3, constant: 1, normals:
    # 3 x NUM_NORMAL_DISTRIBUTIONS].
    # If not uniform will give 0.
    maximum = (
        encoding[UNIFORM_ENCODING_START] * encoding[UNIFORM_ENCODING_START + 2]
    )
    # If constant return constant
    maximum += encoding[CONSTANT_ENCODING_START]
    # If normal return largest mean + five sigma.
    normal_maximum = _get_normal_mean(encoding) + 5 * _get_normal_std(encoding)
    maximum += jnp.max(normal_maximum)

    return maximum


def decode_minimum(encoding: jnp.ndarray) -> float:
    """Decode the minimum value of the dsitribution defined by the encoding.

    Args:
        encoding: Encoded distribution.

    Returns:
        Minimum value of encoded distribution. May be an approximation for
        distributions without a minimum.
    """
    # Encoding is currently [uniform: 3, constant: 1, normals:
    # 3 x NUM_NORMAL_DISTRIBUTIONS].
    # If not uniform will give 0.
    minimum = (
        encoding[UNIFORM_ENCODING_START] * encoding[UNIFORM_ENCODING_START + 1]
    )
    # If constant return constant
    minimum += encoding[CONSTANT_ENCODING_START]
    # If normal return smallest mean - five sigma.
    minimum += encoding[3] * (encoding[4] - encoding[5]*5)
    # If constant return constant
    normal_minimum = _get_normal_mean(encoding) - 5 * _get_normal_std(encoding)
    minimum += jnp.min(normal_minimum)

    return minimum


def _draw_from_normals(encoding: jnp.ndarray, rng: Sequence[int]) -> float:
    """Draw from encoded normal distributions.

    Args:
        encoding: Encoded distribution.
        rng: jax PRNG key.

    Returns:
        A draw from the encoded normal distributions.
    """
    rng_index, rng_normal = jax.random.split(rng)
    weights = _get_normal_weights(encoding)
    means = _get_normal_mean(encoding)
    stds = _get_normal_std(encoding)

    # First select which normal will be drawn from.
    weight_cumulative = jnp.cumsum(weights)
    index_draw = jax.random.uniform(rng_index)
    normal_index = jnp.searchsorted(weight_cumulative, index_draw)

    # Now draw from that normal. Remember order is [weight, mean, std].
    draw = jax.random.normal(rng_normal) * stds[normal_index]
    draw += means[normal_index]
    return draw


def draw_from_encoding(encoding: jnp.ndarray, rng: Sequence[int]) -> float:
    """Draw from encoded distribution.

    Args:
        encoding: Encoded distribution.
        rng: jax PRNG key.

    Returns:
        A draw from the encoded distribution.
    """
    rng_uniform, rng_normal = jax.random.split(rng)

    # Encoding is currently [uniform: 3, constant: 1, normals:
    # 3 x NUM_NORMAL_DISTRIBUTIONS].
    # Start with the uniform component.
    draw = encoding[UNIFORM_ENCODING_START] * (
        jax.random.uniform(rng_uniform) * (encoding[UNIFORM_ENCODING_START + 2] -
        encoding[UNIFORM_ENCODING_START + 1]) +
        encoding[UNIFORM_ENCODING_START + 1]
    )

    # Now the normal component.
    normal_weights = _get_normal_weights(encoding)
    draw += jax.lax.select(
        jnp.sum(normal_weights) > 0.0,
        _draw_from_normals(encoding, rng_normal),
        0.0)

    # And finally the constant
    draw += encoding[CONSTANT_ENCODING_START]

    return draw


def _calculate_mixture_mean_std(encoding: jnp.ndarray) -> Tuple[float, float]:
    """Calculate the mean and standard deviation for the mixture of Gaussians.

    Args:
        encoding: Encoded distribution.

    Returns:
        Mean and standard deviation for the encoded mixture of Gaussians.
    """
    normal_weights = _get_normal_weights(encoding)
    normal_means = _get_normal_mean(encoding)
    normal_stds = _get_normal_std(encoding)

    # Calculate the mean and standard deviation of the mixture
    mixture_mean = jnp.sum(normal_weights * normal_means)
    mixture_std = jnp.sum(normal_weights *
                          (normal_means ** 2 + normal_stds ** 2))
    mixture_std -= mixture_mean ** 2
    mixture_std = jnp.sqrt(mixture_std)

    return mixture_mean, mixture_std


def normalize_param(parameter: float, encoding: jnp.ndarray) -> float:
    """Return parameter normalized by encoded distribution.

    Args:
        parameter: Parameter value to normalize.
        encoding: Encoded distribution.

    Returns:
        Normalized parameter.
    """
    # Encoding is currently [uniform: 3, constant: 1, normals:
    # 3 x NUM_NORMAL_DISTRIBUTIONS].
    # Normalize uniform distribution to be between 0 and 1.
    normalized_param = jax.lax.select(
        encoding[UNIFORM_ENCODING_START] > 0.0,
        (parameter - encoding[UNIFORM_ENCODING_START + 1]) /
            (encoding[UNIFORM_ENCODING_START + 2] -
            encoding[UNIFORM_ENCODING_START + 1]),
        0.0
    )

    # Normalize mixture of normal distributions to mean 0 and standard
    # deviation 1.
    normal_weights = _get_normal_weights(encoding)
    mixture_mean, mixture_std = _calculate_mixture_mean_std(encoding)

    normalized_param += jax.lax.select(
        jnp.sum(normal_weights) > 0.0,
        (parameter - mixture_mean) / mixture_std, 0.0)

    # Constant will be normalzied to 0.0 by default.

    return normalized_param

def unnormalize_param(normalized_parameter: float,
                      encoding: jnp.ndarray) -> float:
    """Return parameter with the normalization removed.

    Args:
        normalized_parameter: Parameter value that has been normalized.
        encoding: Encoded distribution.

    Returns:
        Parameter without normalization.
    """
    # This reverse the procedures specified in normalize_param.
    # Uniform distribution was normalized to be between 0 and 1.
    param = jax.lax.select(
        encoding[UNIFORM_ENCODING_START] > 0.0,
        normalized_parameter *
            (encoding[UNIFORM_ENCODING_START + 2] -
             encoding[UNIFORM_ENCODING_START + 1]) +
             encoding[UNIFORM_ENCODING_START + 1],
        0.0
    )

    # Normal distribution was given mean 0 and standard deviation 1.0.
    normal_weights = _get_normal_weights(encoding)
    mixture_mean, mixture_std = _calculate_mixture_mean_std(encoding)
    param += jax.lax.select(
        jnp.sum(normal_weights) > 0.0,
        normalized_parameter * mixture_std + mixture_mean, 0.0)

    # Constant value must be restored
    param += encoding[CONSTANT_ENCODING_START]

    return param

def generate_grids(
        config: Mapping[str, Mapping[str, jnp.ndarray]]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate the x- and y-grid on which to generate the lensing image.

    Args:
        config: Configuration dictionary from which the detector kwargs will be
            drawn.

    Returns:
        x- and y-grid in units of arcseconds as a tuple.
    """
    kwargs_detector = config['kwargs_detector']
    grid_x, grid_y = utils.coordinates_evaluate(kwargs_detector['n_x'],
        kwargs_detector['n_y'], kwargs_detector['pixel_width'],
        kwargs_detector['supersampling_factor'])
    return grid_x, grid_y


def initialize_cosmology_params(
        config: Mapping[str, Mapping[str, Union[Any,jnp.ndarray]]],
        rng: Sequence[int]
) -> Union[Dict[str, Union[float, int, jnp.ndarray]], Any]:
    """Initialize the cosmology parameters as needed by the config.

    Args:
        config: Configuration dictionary for input generation.
        rng: jax PRNG key.
        all_models: Tuple of model classes to consider for each component.

    Returns:
        Cosmological parameters with appropriate lookup table.

    Notes:
        Return type guarantee is stronger than Any, but Any is required since
        pytype doesn't know what is returned by calls to
        model.modify_cosmology_params.
    """
    max_source_z = decode_maximum(
        config['lensing_config']['source_params']['z_source'])
    dz = decode_minimum(config['lensing_config']['los_params']['dz'])
    m_max = max(
        decode_maximum(config['lensing_config']['subhalo_params']['m_max']),
        decode_maximum(config['lensing_config']['los_params']['m_max']))
    m_min = min(
        decode_minimum(config['lensing_config']['subhalo_params']['m_min']),
        decode_minimum(config['lensing_config']['los_params']['m_min']))

    cosmology_params_init = draw_sample(config['cosmology_params'], rng)

    # Initial bounds on lagrangian radius are just placeholders.
    cosmology_params = cosmology_utils.add_lookup_tables_to_cosmology_params(
        cosmology_params_init, max_source_z, dz / 2, 1e-4, 1e3, 2)
    r_min = cosmology_utils.lagrangian_radius(cosmology_params,
                                              m_min / 10)
    r_max = cosmology_utils.lagrangian_radius(cosmology_params,
                                              m_max * 10)
    cosmology_params = cosmology_utils.add_lookup_tables_to_cosmology_params(
        cosmology_params_init, max_source_z, dz / 2, r_min, r_max, 10000)
    extrenal_los_params = {'m_min': m_min, 'm_max': m_max, 'dz': dz}
    cosmology_params = los.add_los_lookup_tables_to_cosmology_params(
        extrenal_los_params, cosmology_params, max_source_z
    )

    # Add the additional parameters required by the lens and source models.
    for model_group in config['all_models']:
        for model in config['all_models'][model_group]:
            cosmology_params = model.modify_cosmology_params(cosmology_params)

    return cosmology_params


def initialize_lookup_tables(
    config: Mapping[str, Mapping[str, Union[Any,jnp.ndarray]]],
) -> Dict[str, Union[float, int, jnp.ndarray]]:
    """Initialize lookup tables from provided models.

    Args:
        all_models: Tuple of model classes to consider for each component.

    Returns:
        Lookup tables for all the included models.
    """
    lookup_tables = {}
    for model_group in config['all_models']:
        for model in config['all_models'][model_group]:
            lookup_tables = model.add_lookup_tables(lookup_tables)

    return lookup_tables


def draw_sample(
        encoded_configuration: Mapping[str, Union[float, Mapping[str, Any]]],
        rng: Sequence[int]
) -> Mapping[str, Union[float, Mapping[str, Any]]]:
    """Map an econded configuration into a configuration of randomly draws.

    Args:
        encoded_configuration: Configuration with encoded distribution for all
            leaves of the PyTree structure.
        rng: jax PRNG key.

    Returns:
        Configuration with encoded distribution replaced by draws from those
            distributions.
    """
    # Generate the rng keys we will need for each leaf.
    treedef = jax.tree_util.tree_structure(encoded_configuration)
    rng_keys = jax.random.split(rng, treedef.num_leaves)
    rng_tree = jax.tree_util.tree_unflatten(treedef, rng_keys)
    return jax.tree_util.tree_map(draw_from_encoding, encoded_configuration,
                                  rng_tree)


def extract_multiple_models(
        encoded_configuration: Mapping[str, Mapping[str, float]],
        rng: Sequence[int], n_models: int
) -> Dict[str, jnp.ndarray]:
    """Extract multiple models from a single configuration.

    Args:
        encoded_configuration: Encodings for each of the parameters of the
            model(s).
        rng: jax PRNG key.
        n_models: Number of models to draw parameters for.

    Returns:
        Draws for the parameters for all the models. For each parameters, first
        dimension will be the number of models.
    """
    draw_sample_vmap = jax.vmap(draw_sample, in_axes=[None, 0])
    rng_list = jax.random.split(rng, n_models)
    draws = draw_sample_vmap(encoded_configuration, rng_list)
    # Add the model index list.
    draws['model_index'] = jnp.arange(n_models)
    return draws


def extract_multiple_models_angular(
        encoded_configuration: Mapping[str, Mapping[str, float]],
        rng: Sequence[int],
        cosmology_params: Dict[str, Union[float, int, jnp.ndarray]],
        all_models: Sequence[Any]
) -> Dict[str, jnp.ndarray]:
    """Extract multiple models and translate them to angular coordinates.

    Args:
        encoded_configuration: Encodings for each of the parameters of the
            model(s).
        rng: jax PRNG key.
        all_models: Model classes to use for translation from physical to
            angular units.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.

    Returns:
        Draws for the parameters for all the models. For each parameters, first
        dimension will be the number of models. Is mutable.
    """
    n_models = len(all_models)

    # Start by drawing the parameters directly from the config
    draws = extract_multiple_models(encoded_configuration, rng, n_models)

    # Convert those parameters using the conversion function included in
    # each source model. This may often do nothing at all.
    # Since different models add different properties to the config, we cannot
    # use a switch statement here. Instead loop over all the models.
    for model in all_models:
        draws = jax.vmap(model.convert_to_angular, in_axes=[0, None])(
            draws, cosmology_params
        )
    return draws


def rotate_params(
    all_params: Dict[str, Dict[str, jnp.ndarray]],
    truth_parameters: Tuple[Sequence[str], Sequence[str], Sequence[int]],
    rotation_angle: float
) -> jnp.ndarray:
    """Rotate the parameter as required by the physical type of the parameter.

    Args:
        all_params: All of the parameters grouped by object. Must be mutable.
        truth_parameters: List of the lensing objects, corresponding
            parameters to extract, and main object index for each parameter.
        rotation_angle: Counterclockwise angle of rotation.

    Returns:
        All of the parameters grouped by object with the rotation applied.

    Notes:
        The relationship between parameter name and rotation is hard coded here.
        Also only supports there being one of each type of parameter. Pretty
        fragile code, but the rotations are only relevant for comparison
        exmperiments.
    """
    extract_objects, extract_keys, _ = truth_parameters

    # The basic rotation operation applied to each parameter pair.
    def _rotation_operation(param_one: str, param_two: str,
        rotation_angle: float):
        # Get the parameter indices.
        index_one = extract_keys.index(param_one)
        index_two = extract_keys.index(param_two)

        # Extract the values and rotate.
        param_one_val = all_params[extract_objects[index_one]][param_one]
        param_two_val = all_params[extract_objects[index_two]][param_two]
        param_one_val, param_two_val = utils.rotate_coordinates(
            param_one_val, param_two_val, rotation_angle
        )

        # Set the new parameter values.
        all_params[extract_objects[index_one]][param_one] = param_one_val
        all_params[extract_objects[index_two]][param_two] = param_two_val


    if 'center_x' in extract_keys or 'center_y' in extract_keys:
        _rotation_operation('center_x', 'center_y', rotation_angle)

    if 'ellip_x' in extract_keys or 'ellip_xy' in extract_keys:
        _rotation_operation('ellip_x', 'ellip_xy', 2 * rotation_angle)

    if 'gamma_one' in extract_keys or 'gamma_two' in extract_keys:
        _rotation_operation('gamma_one', 'gamma_two', 2 * rotation_angle)

    if 'angle' in extract_keys:
        index = extract_keys.index('angle')
        angle = all_params[extract_objects[index]]['angle']
        all_params[extract_objects[index]]['angle'] = angle + rotation_angle

    return all_params


def extract_truth_values(
        all_params: Dict[str, Dict[str, jnp.ndarray]],
        lensing_config: Mapping[str, Mapping[str, jnp.ndarray]],
        truth_parameters: Tuple[Sequence[str], Sequence[str], Sequence[int]],
        rotation_angle: Optional[float] = 0.0,
        normalize_truths: Optional[bool] = True
) -> jnp.ndarray:
    """Extract the truth parameters and normalize them according to the config.

    Args:
        all_params: All of the parameters grouped by object. Must be mutable.
        lensing_config: Distribution encodings for each of the parameters.
        truth_parameters: List of the lensing objects, corresponding
            parameters to extract, and main object index for each parameter.
        rotation_angle: Counterclockwise angle by which to rotate truths.
        normalize_truths: If true, normalize parameters according to the
            encoded distribtion.

    Returns:
        Truth values for each of the requested parameters.
    """
    extract_objects, extract_keys, extract_indices = truth_parameters

    # Begin by adding the rotation applied to the image.
    rotate_params(all_params, truth_parameters, rotation_angle)

    # Now normalize the parameters if requested.
    if normalize_truths:
        return jnp.array(jax.tree_util.tree_map(
            lambda x, y, z: normalize_param(all_params[x][y][z],
                                            lensing_config[x][y]),
            extract_objects, extract_keys, extract_indices))

    return jnp.array(jax.tree_util.tree_map(
        lambda x, y, z: all_params[x][y][z], extract_objects, extract_keys,
        extract_indices))


def replace_truth_values(
        truth: jnp.ndarray,
        all_params: Dict[str, Dict[str, jnp.ndarray]],
        lensing_config: Mapping[str, Mapping[str, jnp.ndarray]],
        truth_parameters: Tuple[Sequence[str], Sequence[str], Sequence[int]],
        normalize_truths: bool) -> jnp.ndarray:
    """Replace parameters with the unormalized specified truth.

    Args:
        truth: True values to insert in the parameters.
        all_params: All of the parameters grouped by object. Must be mutable.
        lensing_config: Distribution encodings for each of the parameters.
        truth_parameters: List of the lensing objects, corresponding
            parameters to extract, and main object index for each parameter.
        normalize_truths: If true, normalize parameters according to the
            encoded distribtion.

    Returns:
        Truth values for each of the requested parameters.
    """
    extract_objects, extract_keys, extract_indices = truth_parameters

    # Replace each truth parameter
    for truth_value, obj, key, index in zip(
        truth, extract_objects, extract_keys, extract_indices
    ):
        # Unnormalize the truth before passing them to parameters.
        if normalize_truths:
            truth_value = unnormalize_param(
                truth_value, lensing_config[obj][key]
            )

        all_params[obj][key] = all_params[obj][key].at[index].set(truth_value)

    return all_params


def _draw_all_params(
    lensing_config: Mapping[str, Mapping[str, jnp.ndarray]],
    cosmology_params: Dict[str, Union[float, int, jnp.ndarray]],
    rng: Sequence[int],
    all_models: Mapping[str, Sequence[Any]],
    kwargs_psf: Mapping[str, Union[float, int, jnp.ndarray]],
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Draw all of the parameters of lensing simulations.

    Args:
        lensing_config: Distribution encodings for each of the objects in the
            lensing system.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        rng: jax PRNG key.
        all_models: Tuple of model classes to consider for each component.
        kwargs_psf: Keyword arguments defining the point spread function. The
            psf is applied in the supersampled space, so the size of pixels
            should be defined with respect to the supersampled space.

    Returns:
        Parameters for the source, lens light, los, subhalo, and main deflector
        models.
    """
    # Draw an instance of the parameter values for each object in our lensing
    # system.
    rng_md, rng_source, rng_ll, rng_los, rng_sub, rng = jax.random.split(rng, 6)
    main_deflector_params = extract_multiple_models_angular(
        lensing_config['main_deflector_params'], rng_md, cosmology_params,
        all_models['all_main_deflector_models']
    )
    source_params = extract_multiple_models_angular(
        lensing_config['source_params'], rng_source, cosmology_params,
        all_models['all_source_models']
    )
    lens_light_params = extract_multiple_models_angular(
        lensing_config['lens_light_params'], rng_ll, cosmology_params,
        all_models['all_lens_light_models']
    )
    los_params = extract_multiple_models(
        lensing_config['los_params'], rng_los,
        len(all_models['all_los_models'])
    )
    subhalo_params = extract_multiple_models(
        lensing_config['subhalo_params'], rng_sub,
        len(all_models['all_subhalo_models'])
    )

    # PSF kwargs are allowed to be random.
    rng_psf, rng = jax.random.split(rng, 2)
    psf_params = extract_multiple_models(
        kwargs_psf, rng_psf, len(all_models['all_psf_models'])
    )

    # Repackage the parameters.
    all_params = {
        'source_params': source_params,
        'lens_light_params': lens_light_params,
        'los_params': los_params, 'subhalo_params': subhalo_params,
        'main_deflector_params': main_deflector_params,
        'psf_params': psf_params
    }
    return all_params


def _draw_image_and_truth(
    all_params: Mapping[str, Mapping[str, jnp.ndarray]],
    cosmology_params: Dict[str, Union[float, int, jnp.ndarray]],
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    rng: Sequence[int],
    rotation_angle: float,
    all_models: Mapping[str, Sequence[Any]],
    principal_model_indices: Mapping[str, int],
    kwargs_simulation: Mapping[str, int],
    kwargs_detector:  Mapping[str, Union[int, float]],
    normalize_image: bool,
    lookup_tables: Dict[str, Union[float, jnp.ndarray]]
) -> jnp.ndarray:
    """Draw image for a set of realization parameters.

    Args:
        all_params: Parameters for simulation.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        grid_x: x-grid in units of arcseconds.
        grid_y: y-grid in units of arcseconds.
        rng: jax PRNG key.
        rotation_angle: Counterclockwise angle by which to rotate images and
            truths.
        all_models: Tuple of model classes to consider for each component.
        principal_model_indices: Indices for the principal model of each
            lensing component.
        kwargs_simulation: Keyword arguments for the draws of the substructure.
        kwargs_detector: Keyword arguments defining the detector configuration.
        normalize_image: If True, the image will be normalized to have
            standard deviation 1.
        lookup_tables: Optional lookup tables for source and derivative
            functions.

    Returns:
        Image.
    """
    # Get the principal model for each lensing object.
    all_params_principal = {
        lens_obj: jax.tree_util.tree_map(
            lambda x: x[principal_model_indices[lens_obj]], all_params[lens_obj]
        ) for lens_obj in all_params
    }

    # Pull out the padding and bins we will use while vmapping our simulation.
    num_z_bins = kwargs_simulation['num_z_bins']
    los_pad_length = kwargs_simulation['los_pad_length']
    subhalos_pad_length = kwargs_simulation['subhalos_pad_length']
    subhalos_n_chunks = kwargs_simulation.get('subhalos_n_chunks', 1)
    sampling_pad_length = kwargs_simulation['sampling_pad_length']

    rng_los, rng_sub, rng_noise = jax.random.split(rng, 3)
    los_before_tuple, los_after_tuple = los.draw_los(
        main_deflector_params=all_params_principal['main_deflector_params'],
        source_params=all_params_principal['source_params'],
        los_params=all_params_principal['los_params'],
        cosmology_params=cosmology_params, rng=rng_los, num_z_bins=num_z_bins,
        los_pad_length=los_pad_length
    )
    subhalos_z, subhalos_kwargs = subhalos.draw_subhalos(
        main_deflector_params=all_params_principal['main_deflector_params'],
        source_params=all_params_principal['source_params'],
        subhalo_params=all_params_principal['subhalo_params'],
        cosmology_params=cosmology_params, rng=rng_sub,
        subhalos_pad_length=subhalos_pad_length,
        sampling_pad_length=sampling_pad_length
    )

    kwargs_lens_all = {
        'z_array_los_before': los_before_tuple[0],
        'kwargs_los_before': los_before_tuple[1],
        'z_array_los_after': los_after_tuple[0],
        'kwargs_los_after': los_after_tuple[1],
        'kwargs_main_deflector': all_params['main_deflector_params'],
        'z_array_main_deflector': all_params['main_deflector_params']['z_lens'],
        'z_array_subhalos': subhalos_z, 'kwargs_subhalos': subhalos_kwargs
    }
    z_source = all_params_principal['source_params']['z_source']

    # Apply the rotation angle to the image through the grid. This requires
    # rotating the coordinates by the negative angle.
    grid_x, grid_y = utils.rotate_coordinates(grid_x, grid_y, -rotation_angle)

    image_supersampled = image_simulation.generate_image(
        grid_x, grid_y, kwargs_lens_all, all_params['source_params'],
        all_params['lens_light_params'], all_params['psf_params'],
        cosmology_params, z_source, kwargs_detector, all_models, apply_psf=True,
        lookup_tables=lookup_tables, subhalos_n_chunks=subhalos_n_chunks
    )
    image = utils.downsample(
        image_supersampled, kwargs_detector['supersampling_factor']
    )
    image += image_simulation.noise_realization(
        image, rng_noise, kwargs_detector
    )
    # Normalize the image to have standard deviation 1.
    if normalize_image:
        image /= jnp.std(image)

    return image


def draw_image_and_truth(
    lensing_config: Mapping[str, Mapping[str, jnp.ndarray]],
    cosmology_params: Dict[str, Union[float, int, jnp.ndarray]],
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    rng: Sequence[int],
    rotation_angle: float,
    all_models: Mapping[str, Sequence[Any]],
    principal_model_indices: Mapping[str, int],
    kwargs_simulation: Mapping[str, int],
    kwargs_detector:  Mapping[str, Union[int, float]],
    kwargs_psf: Mapping[str, Union[float, int, jnp.ndarray]],
    truth_parameters: Tuple[Sequence[str], Sequence[str], Sequence[int]],
    normalize_image: Optional[bool] = True,
    normalize_truths: Optional[bool] = True,
    normalize_config: Optional[Mapping[str, Mapping[str, jnp.ndarray]]] = None,
    lookup_tables: Optional[Dict[str, Union[float, jnp.ndarray]]] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Draw image and truth values for a realization of the lensing config.

    Args:
        lensing_config: Distribution encodings for each of the objects in the
            lensing system.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        grid_x: x-grid in units of arcseconds.
        grid_y: y-grid in units of arcseconds.
        rng: jax PRNG key.
        rotation_angle: Counterclockwise angle by which to rotate images and
            truths.
        all_models: Tuple of model classes to consider for each component.
        principal_model_indices: Indices for the principal model of each
            lensing component.
        kwargs_simulation: Keyword arguments for the draws of the substructure.
        kwargs_detector: Keyword arguments defining the detector configuration.
        kwargs_psf: Keyword arguments defining the point spread function. The
            psf is applied in the supersampled space, so the size of pixels
            should be defined with respect to the supersampled space.
        truth_parameters: List of the lensing objects, corresponding
            parameters to extract, and main object index for each parameter.
        normalize_image: If True, the image will be normalized to have
            standard deviation 1.
        normalize_truths: If true, normalize parameters according to the
            encoded distribtion.
        normalize_config: The lensing config to use for normalization. If None
            will default to the lensing config.
        lookup_tables: Optional lookup tables for source and derivative
            functions.

    Returns:
        Image and corresponding truth values.

    Notes:
        To jit compile, every parameter after rng must be fixed.
    """
    # If no normalization config is specified, assume the input lensing
    # configuration.
    if normalize_config is None:
        normalize_config = lensing_config

    # Repackage the parameters.
    all_params = _draw_all_params(
        lensing_config, cosmology_params, rng, all_models, kwargs_psf
    )

    image = _draw_image_and_truth(
        all_params, cosmology_params, grid_x, grid_y, rng, rotation_angle,
        all_models, principal_model_indices, kwargs_simulation, kwargs_detector,
        normalize_image, lookup_tables
    )

    # Extract the truth values and normalize them.
    truth = extract_truth_values(
        all_params, normalize_config, truth_parameters,
        rotation_angle, normalize_truths
    )

    return image, truth


def draw_truth(
    lensing_config: Mapping[str, Mapping[str, jnp.ndarray]],
    cosmology_params: Dict[str, Union[float, int, jnp.ndarray]],
    rng: Sequence[int],
    rotation_angle: float,
    all_models: Mapping[str, Sequence[Any]],
    kwargs_psf: Mapping[str, Union[float, int, jnp.ndarray]],
    truth_parameters: Tuple[Sequence[str], Sequence[str], Sequence[int]],
    normalize_truths: Optional[bool] = True,
    normalize_config: Optional[Mapping[str, Mapping[str, jnp.ndarray]]] = None
) -> jnp.ndarray:
    """Draw truth values for a realization of the lensing config.

    Args:
        lensing_config: Distribution encodings for each of the objects in the
            lensing system.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        rng: jax PRNG key.
        rotation_angle: Counterclockwise angle by which to rotate images and
            truths.
        all_models: Tuple of model classes to consider for each component.
        kwargs_psf: Keyword arguments defining the point spread function. The
            psf is applied in the supersampled space, so the size of pixels
            should be defined with respect to the supersampled space.
        truth_parameters: List of the lensing objects, corresponding
            parameters to extract, and main object index for each parameter.
        normalize_truths: If true, normalize parameters according to the
            encoded distribtion.
        normalize_config: The lensing config to use for normalization. If None
            will default to the lensing config.

    Returns:
        Truth values.
    """
    # If no normalization config is specified, assume the input lensing
    # configuration.
    if normalize_config is None:
        normalize_config = lensing_config

    # Repackage the parameters.
    all_params = _draw_all_params(
        lensing_config, cosmology_params, rng, all_models, kwargs_psf
    )

    # Extract the truth values and normalize them.
    truth = extract_truth_values(
        all_params, normalize_config, truth_parameters,
        rotation_angle, normalize_truths
    )

    return truth


def draw_image(
    lensing_config: Mapping[str, Mapping[str, jnp.ndarray]],
    truth: jnp.ndarray,
    cosmology_params: Dict[str, Union[float, int, jnp.ndarray]],
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    rng: Sequence[int],
    all_models: Mapping[str, Sequence[Any]],
    principal_model_indices: Mapping[str, int],
    kwargs_simulation: Mapping[str, int],
    kwargs_detector:  Mapping[str, Union[int, float]],
    kwargs_psf: Mapping[str, Union[float, int, jnp.ndarray]],
    truth_parameters: Tuple[Sequence[str], Sequence[str], Sequence[int]],
    normalize_image: Optional[bool] = True,
    normalize_truths: Optional[bool] = True,
    normalize_config: Optional[Mapping[str, Mapping[str, jnp.ndarray]]] = None,
    lookup_tables: Optional[Dict[str, Union[float, jnp.ndarray]]] = None
) -> jnp.ndarray:
    """Draw image from lensing config with fixed truth parameters.

    Args:
        lensing_config: Distribution encodings for each of the objects in the
            lensing system.
        truth: Value of fixed truth parameters to draw simulation from.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        grid_x: x-grid in units of arcseconds.
        grid_y: y-grid in units of arcseconds.
        rng: jax PRNG key.
        all_models: Tuple of model classes to consider for each component.
        principal_model_indices: Indices for the principal model of each
            lensing component.
        kwargs_simulation: Keyword arguments for the draws of the substructure.
        kwargs_detector: Keyword arguments defining the detector configuration.
        kwargs_psf: Keyword arguments defining the point spread function. The
            psf is applied in the supersampled space, so the size of pixels
            should be defined with respect to the supersampled space.
        truth_parameters: List of the lensing objects, corresponding
            parameters to extract, and main object index for each parameter.
        normalize_image: If True, the image will be normalized to have
            standard deviation 1.
        normalize_truths: If true, normalize parameters according to the
            encoded distribtion.
        normalize_config: The lensing config to use for normalization. If None
            will default to the lensing config.
        lookup_tables: Optional lookup tables for source and derivative
            functions.

    Returns:
        Image.

    Notes:
        To jit compile, every parameter after rng must be fixed.
    """
    # If no normalization config is specified, assume the input lensing
    # configuration.
    if normalize_config is None:
        normalize_config = lensing_config

    # Repackage the parameters.
    all_params = _draw_all_params(
        lensing_config, cosmology_params, rng, all_models, kwargs_psf
    )

    # Replace the parameters that are fixed by the truth input.
    all_params = replace_truth_values(
        truth, all_params, normalize_config, truth_parameters,
        normalize_truths
    )

    rotation_angle = 0.0
    image = _draw_image_and_truth(
        all_params, cosmology_params, grid_x, grid_y, rng, rotation_angle,
        all_models, principal_model_indices, kwargs_simulation, kwargs_detector,
        normalize_image, lookup_tables
    )

    return image
