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
"""A simple actor-critic agent implemented in JAX + Haiku."""

from typing import Any, Callable, NamedTuple, Tuple

from bsuite.baselines import base
from bsuite.baselines.utils import sequence

import dm_env
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax

Logits = jnp.ndarray
Value = jnp.ndarray
PolicyValueNet = Callable[[jnp.ndarray], Tuple[Logits, Value]]


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: Any


class ActorCritic(base.Agent):
    """Feed-forward actor-critic agent."""

    def __init__(
            self,
            obs_spec: specs.Array,
            action_spec: specs.DiscreteArray,
            network: PolicyValueNet,
            optimizer: optax.GradientTransformation,
            rng: hk.PRNGSequence,
            sequence_length: int,
            discount: float,
            td_lambda: float,
    ):
        # Define loss function.
        def loss(trajectory: sequence.Trajectory) -> jnp.ndarray:
            """"Actor-critic loss."""
            logits, values = network(trajectory.observations)
            td_errors = rlax.td_lambda(
                v_tm1=values[:-1],
                r_t=trajectory.rewards,
                discount_t=trajectory.discounts * discount,
                v_t=values[1:],
                lambda_=jnp.array(td_lambda),
            )
            critic_loss = jnp.mean(td_errors ** 2)
            actor_loss = rlax.policy_gradient_loss(
                logits_t=logits[:-1],
                a_t=trajectory.actions,
                adv_t=td_errors,
                w_t=jnp.ones_like(td_errors))

            return actor_loss + critic_loss

        # Transform the loss into a pure function.
        loss_fn = hk.without_apply_rng(hk.transform(loss)).apply

        # Define update function.
        @jax.jit
        def sgd_step(state: TrainingState,
                     trajectory: sequence.Trajectory) -> TrainingState:
            """Does a step of SGD over a trajectory."""
            gradients = jax.grad(loss_fn)(state.params, trajectory)
            updates, new_opt_state = optimizer.update(gradients, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)
            return TrainingState(params=new_params, opt_state=new_opt_state)

        # Initialize network parameters and optimiser state.
        init, forward = hk.without_apply_rng(hk.transform(network))
        dummy_observation = jnp.zeros((1, *obs_spec.shape), dtype=jnp.float32)
        initial_params = init(next(rng), dummy_observation)
        initial_opt_state = optimizer.init(initial_params)

        # Internalize state.
        self._state = TrainingState(initial_params, initial_opt_state)
        self._forward = jax.jit(forward)
        self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length)
        self._sgd_step = sgd_step
        self._rng = rng

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions according to a softmax policy."""
        key = next(self._rng)
        observation = timestep.observation[None, ...]
        logits, _ = self._forward(self._state.params, observation)
        action = jax.random.categorical(key, logits).squeeze()
        return int(action)

    def update(
            self,
            timestep: dm_env.TimeStep,
            action: base.Action,
            new_timestep: dm_env.TimeStep,
    ):
        """Adds a transition to the trajectory buffer and periodically does SGD."""
        self._buffer.append(timestep, action, new_timestep)
        if self._buffer.full() or new_timestep.last():
            trajectory = self._buffer.drain()
            self._state = self._sgd_step(self._state, trajectory)


class CNNTimeSeries(hk.Module):
    def __init__(self, num_channels: int, output_size: int, name: str = None):
        """
        Args:
        - num_channels: Number of variables in the multivariate time series (input channels).
        - output_size: Number of possible actions (output dimension).
        """
        super().__init__(name=name)
        self.num_channels = num_channels
        self.output_size = output_size

    def __call__(self, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
        - inputs: A [batch_size, sequence_length, num_channels] array representing
                  the multivariate time series.

        Returns:
        - logits: The policy logits for action selection.
        - value: The scalar value estimation for the current state.
        """
        # Apply 1D Convolution over the time dimension with multiple filters
        conv1 = hk.Conv1D(output_channels=32, kernel_shape=3, stride=1, padding="SAME")
        conv2 = hk.Conv1D(output_channels=64, kernel_shape=3, stride=1, padding="SAME")

        # First convolution
        x = conv1(inputs)
        x = jax.nn.relu(x)

        # Second convolution
        x = conv2(x)
        x = jax.nn.relu(x)

        # Global Average Pooling (reduces over the time dimension)
        x = jnp.mean(x, axis=1)  # [batch_size, num_filters]

        # Fully connected layers
        torso = hk.nets.MLP([128, 64])
        embedding = torso(x)

        # Policy and Value heads
        policy_head = hk.Linear(self.output_size)
        value_head = hk.Linear(1)

        logits = policy_head(embedding)
        value = jnp.squeeze(value_head(embedding), axis=-1)

        return logits, value


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray,
                  seed: int = 0) -> base.Agent:
    """Creates an actor-critic agent with default hyperparameters."""

    def network(inputs: jnp.ndarray) -> Tuple[Logits, Value]:
        return CNNTimeSeries(num_channels=obs_spec.shape[-1],
                             output_size=action_spec.num_values)(inputs)

    return ActorCritic(
        obs_spec=obs_spec,
        action_spec=action_spec,
        network=network,
        optimizer=optax.adam(3e-3),
        rng=hk.PRNGSequence(seed),
        sequence_length=32,
        discount=0.99,
        td_lambda=0.9,
    )
