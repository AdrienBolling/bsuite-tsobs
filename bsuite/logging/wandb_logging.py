# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Logging functionality for Weights and Biases (wandb) based experiments."""

from typing import Any, List, Dict

from bsuite import environments
from bsuite import sweep
from bsuite.logging import base
from bsuite.utils import wrappers

import dm_env
import wandb

SAFE_SEPARATOR = '-'
INITIAL_SEPARATOR = '_-_'
BSUITE_PREFIX = 'bsuite_id' + INITIAL_SEPARATOR


def wrap_environment(env: environments.Environment,
                     bsuite_id: str,
                     project_name: str,
                     project_entity: str,
                     project_group: str,
                     project_config: Dict[str, Any] = None,
                     project_tags: List[str] = None,
                     overwrite: bool = False,
                     log_by_step: bool = False) -> dm_env.Environment:
    """Returns a wrapped environment that logs using wandb."""
    logger = WandbLogger(bsuite_id, project_name, project_entity, project_group, project_config, project_tags,
                         overwrite)
    return wrappers.Logging(env, logger, log_by_step=log_by_step)


class WandbLogger(base.Logger):
    """Logs data to Weights and Biases (wandb).

  This simplified logger sends bsuite experiment metrics to wandb.
  Each bsuite_id logs under a specific project and experiment ID.
  The logger is initialized with a project name and bsuite_id.
  """

    def __init__(self,
                 bsuite_id: str,
                 project_name: str,
                 project_entity: str,
                 project_group: str,
                 project_config: Dict[str, Any] = None,
                 project_tags: List[str] = None,
                 overwrite: bool = False):
        """Initializes a new wandb logger."""

        # The default '/' symbol is dangerous for file systems!
        safe_bsuite_id = bsuite_id.replace(sweep.SEPARATOR, SAFE_SEPARATOR)
        experiment_name = f'{BSUITE_PREFIX}{safe_bsuite_id}'

        # Start a new wandb run
        wandb.init(project=project_name, entity=project_entity, group=project_group,
                   config=project_config, tags=project_tags, reinit=overwrite)

    def write(self, data: dict[str, Any]):
        """Logs data to wandb."""
        # Log the dictionary data to wandb as metrics
        wandb.log(data)
