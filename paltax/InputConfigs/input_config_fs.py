# coding=utf-8

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
"""Configuration file for warm dark matter.
"""

from paltax.InputConfigs import input_config_wdm
from paltax.input_pipeline import encode_constant

def get_config():
    """Get the hyperparameter configuration"""
    config = input_config_wdm.get_config()

    config['lensing_config']['source_params']['galaxy_index'] = (
        encode_constant(0.32)
    )

    return config
