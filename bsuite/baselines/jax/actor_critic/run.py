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
"""Run an actor-critic agent instance on a bsuite experiment."""

from absl import app
from absl import flags

import bsuite
from bsuite import sweep

from bsuite.baselines import experiment
from bsuite.baselines.jax import actor_critic
from bsuite.baselines.utils import pool

# Internal imports.

# Experiment flags.
flags.DEFINE_string(
    'bsuite_id', 'catch/0', 'BSuite identifier. '
                            'This global flag can be used to control which environment is loaded.')
flags.DEFINE_string('save_path', '/tmp/bsuite', 'where to save bsuite results')
flags.DEFINE_enum('logging_mode', 'wandb', ['csv', 'sqlite', 'terminal', 'wandb'],
                  'which form of logging to use for bsuite results')
flags.DEFINE_boolean('overwrite', False, 'overwrite csv logging if found')
flags.DEFINE_integer('num_episodes', None, 'Overrides number of training eps.')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')

# Weights and Biases flags.
flags.DEFINE_integer('max_episode_length', 1000, 'Maximum episode length')
flags.DEFINE_string('project_name', 'rl-with-ts-obs', 'Weights and Biases project name')
flags.DEFINE_string('project_entity', 'bolling-adrien', 'Weights and Biases project entity')
flags.DEFINE_string('project_group', 'ts-bsuite', 'Weights and Biases project group')
flags.DEFINE_list('project_tags', None, 'Weights and Biases project tags')

FLAGS = flags.FLAGS


def run(bsuite_id: str) -> str:
    """Runs an A2C agent on a given bsuite environment, logging to CSV."""

    env = bsuite.load_and_record(
        bsuite_id=bsuite_id,
        save_path=FLAGS.save_path,
        logging_mode=FLAGS.logging_mode,
        overwrite=FLAGS.overwrite,
    )

    agent = actor_critic.default_agent(
        env.observation_spec(), env.action_spec())

    num_episodes = FLAGS.num_episodes or getattr(env, 'bsuite_num_episodes')
    experiment.run(
        agent=agent,
        environment=env,
        num_episodes=num_episodes,
        verbose=FLAGS.verbose)

    return bsuite_id

def run_with_wandb_logging(bsuite_id: str) -> str:
    """
    Runs an A2C agent on a given bsuite environment, logging to Weights and Biases.
    """

    config = {
        'bsuite_id': bsuite_id,
        'num_episodes': FLAGS.num_episodes,
        'policy_algo': 'actor_critic',
    }

    env = bsuite.load_and_record_to_wandb(
        bsuite_id=bsuite_id,
        max_episode_length=FLAGS.max_episode_length,
        project_name=FLAGS.project_name,
        project_entity=FLAGS.project_entity,
        project_group=FLAGS.project_group,
        project_config=config,
        project_tags=FLAGS.project_tags,
        overwrite=FLAGS.overwrite,
    )

    agent = actor_critic.default_agent(
        env.observation_spec(), env.action_spec())

    num_episodes = FLAGS.num_episodes or getattr(env, 'bsuite_num_episodes')
    experiment.run(
        agent=agent,
        environment=env,
        num_episodes=num_episodes,
        verbose=FLAGS.verbose)


def main(_):
    # Parses whether to run a single bsuite_id, or multiprocess sweep.
    bsuite_id = FLAGS.bsuite_id

    if bsuite_id in sweep.SWEEP:
        print(f'Running single experiment: bsuite_id={bsuite_id}.')
        if FLAGS.logging_mode == 'wandb':
            run_with_wandb_logging(bsuite_id)
        else:
            run(bsuite_id)

    elif hasattr(sweep, bsuite_id):
        bsuite_sweep = getattr(sweep, bsuite_id)
        print(f'Running sweep over bsuite_id in sweep.{bsuite_sweep}')
        FLAGS.verbose = False
        pool.map_mpi(run, bsuite_sweep)

    else:
        raise ValueError(f'Invalid flag: bsuite_id={bsuite_id}.')


if __name__ == '__main__':
    app.run(main)
