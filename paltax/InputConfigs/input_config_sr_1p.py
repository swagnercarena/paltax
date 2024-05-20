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
"""Configuration file using only 50 percent of sources.
"""

from  paltax.InputConfigs import input_config_br
from paltax.input_pipeline import encode_uniform

def get_config():
    """Get the hyperparameter configuration"""
    config = input_config_br.get_config()

    # Limit the number of unique batches.
    config['source_params']['galaxy_index'] = encode_uniform(
        minimum=0.0, maximum=0.01
    )

    return config
