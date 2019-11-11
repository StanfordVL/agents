# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r'''Train and Eval SAC.

To run:

```bash
tensorboard --logdir $HOME/tmp/sac_v1/gym/HalfCheetah-v2/ --port 2223 &

python tf_agents/agents/sac/examples/v1/train_eval.py \
  --root_dir=$HOME/tmp/sac_v1/gym/HalfCheetah-v2/ \
  --alsologtostderr
```
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
#from tf_agents.environments import suite_gibson
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import batched_py_metric
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.networks.utils import mlp_layers
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.utils import episode_utils
from tf_agents.trajectories.time_step import TimeStep
from tensorflow.python.framework.tensor_spec import TensorSpec, BoundedTensorSpec
import numpy as np
from IPython import embed
import collections


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None,
                          'Path to the gin config files.')
flags.DEFINE_multi_string('gin_param', None,
                          'Gin binding to pass through.')

flags.DEFINE_integer('num_iterations', 1000000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_integer('initial_collect_steps', 1000,
                     'Number of steps to collect at the beginning of training using random policy')
flags.DEFINE_integer('collect_steps_per_iteration', 1,
                     'Number of steps to collect and be added to the replay buffer after every training iteration')
flags.DEFINE_integer('num_parallel_environments', 1,
                     'Number of environments to run in parallel')
flags.DEFINE_integer('num_parallel_environments_eval', 1,
                     'Number of environments to run in parallel for eval')
flags.DEFINE_integer('replay_buffer_capacity', 1000000,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer('train_steps_per_iteration', 1,
                     'Number of training steps in every training iteration')
flags.DEFINE_integer('batch_size', 256,
                     'Batch size for each training step. '
                     'For each training iteration, we first collect collect_steps_per_iteration steps to the '
                     'replay buffer. Then we sample batch_size steps from the replay buffer and train the model'
                     'for train_steps_per_iteration times.')
flags.DEFINE_float('gamma', 0.99,
                   'Discount_factor for the environment')
flags.DEFINE_float('actor_learning_rate', 3e-4,
                   'Actor learning rate')
flags.DEFINE_float('critic_learning_rate', 3e-4,
                   'Critic learning rate')
flags.DEFINE_float('alpha_learning_rate', 3e-4,
                   'Alpha learning rate')

flags.DEFINE_integer('num_eval_episodes', 10,
                     'The number of episodes to run eval on.')
flags.DEFINE_integer('eval_interval', 10000,
                     'Run eval every eval_interval train steps')
flags.DEFINE_boolean('eval_only', False,
                     'Whether to run evaluation only on trained checkpoints')
flags.DEFINE_boolean('eval_deterministic', False,
                     'Whether to run evaluation using a deterministic policy')
flags.DEFINE_integer('gpu_c', 0,
                     'GPU id for compute, e.g. Tensorflow.')

# Added for Gibson
flags.DEFINE_string('config_file', '../test/test.yaml',
                    'Config file for the experiment.')
flags.DEFINE_list('model_ids', None,
                  'A comma-separated list of model ids to overwrite config_file.'
                  'len(model_ids) == num_parallel_environments')
flags.DEFINE_list('model_ids_eval', None,
                  'A comma-separated list of model ids to overwrite config_file for eval.'
                  'len(model_ids) == num_parallel_environments_eval')
flags.DEFINE_float('collision_reward_weight', 0.0,
                   'collision reward weight')
flags.DEFINE_string('env_mode', 'headless',
                    'Mode for the simulator (gui or headless)')
flags.DEFINE_string('env_type', 'gibson',
                    'Type for the Gibson environment (gibson or ig)')
flags.DEFINE_float('action_timestep', 1.0 / 10.0,
                   'Action timestep for the simulator')
flags.DEFINE_float('physics_timestep', 1.0 / 40.0,
                   'Physics timestep for the simulator')
flags.DEFINE_integer('gpu_g', 0,
                     'GPU id for graphics, e.g. Gibson.')
flags.DEFINE_boolean('random_position', False,
                     'Whether to randomize initial and target position')

FLAGS = flags.FLAGS


def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1):
    del init_action_stddev
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True)


class InferenceEngine(object):
    def __init__(
        self,
        root_dir,
        conv_layer_params=None,
        encoder_fc_layers=[256],
        actor_fc_layers=[256, 256],
        critic_obs_fc_layers=None,
        critic_action_fc_layers=None,
        critic_joint_fc_layers=[256, 256],
        # Params for target update
        target_update_tau=0.005,
        target_update_period=1,
        # Params for train
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=0.99,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        # Params for eval
        eval_deterministic=False,
        # Params for summaries and logging
        debug_summaries=False,
        summarize_grads_and_vars=False
    ):
        '''A simple train and eval for SAC.'''

        root_dir = os.path.expanduser(root_dir)
        train_dir = os.path.join(root_dir, 'train')

        time_step_spec = TimeStep(
            TensorSpec(shape=(), dtype=tf.int32, name='step_type'),
            TensorSpec(shape=(), dtype=tf.float32, name='reward'),
            BoundedTensorSpec(shape=(), dtype=tf.float32, name='discount',
                              minimum=np.array(0., dtype=np.float32), maximum = np.array(1., dtype=np.float32)),
            collections.OrderedDict({
                'sensor': BoundedTensorSpec(shape=(26,), dtype=tf.float32, name=None,
                                            minimum=np.array(-3.4028235e+38, dtype=np.float32),
                                            maximum=np.array(3.4028235e+38, dtype=np.float32)),
                'depth': BoundedTensorSpec(shape=(60, 80, 1), dtype=tf.float32, name=None,
                                           minimum=np.array(-3.4028235e+38, dtype=np.float32),
                                           maximum=np.array(3.4028235e+38, dtype=np.float32)),
            })
        )
        observation_spec = time_step_spec.observation
        action_spec = BoundedTensorSpec(shape=(2,), dtype=tf.float32, name=None,
                                        minimum=np.array(-1.0, dtype=np.float32),
                                        maximum=np.array(1.0, dtype=np.float32))
        print('observation_spec', observation_spec)
        print('action_spec', action_spec)

        glorot_uniform_initializer = tf.compat.v1.keras.initializers.glorot_uniform()
        preprocessing_layers = {
            'depth': tf.keras.Sequential(mlp_layers(
                conv_layer_params=conv_layer_params,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            )),
            'sensor': tf.keras.Sequential(mlp_layers(
                conv_layer_params=None,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            )),
        }
        preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=actor_fc_layers,
            continuous_projection_net=normal_projection_net,
            kernel_initializer=glorot_uniform_initializer,
        )

        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers,
            kernel_initializer=glorot_uniform_initializer,
        )

        global_step = tf.compat.v1.train.get_or_create_global_step()
        tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=td_errors_loss_fn,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        if eval_deterministic:
            eval_py_policy = py_tf_policy.PyTFPolicy(greedy_policy.GreedyPolicy(tf_agent.policy))
        else:
            eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy)

        def _filter_invalid_transition(trajectories, unused_arg1):
            return ~trajectories.is_boundary()[0]
        batch_size = 1
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=1,
            max_length=1)
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=1,
            sample_batch_size=1,
            num_steps=2).apply(tf.data.experimental.unbatch()).filter(
            _filter_invalid_transition).batch(batch_size).prefetch(1)
        dataset_iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        trajectories, unused_info = dataset_iterator.get_next()
        train_op = tf_agent.train(trajectories)

        train_checkpointer = common.Checkpointer(
            ckpt_dir=train_dir,
            agent=tf_agent,
            global_step=global_step)

        with sess.as_default():
            # Initialize graph.
            train_checkpointer.initialize_or_restore(sess)

        self.sess = sess
        self.eval_py_policy = eval_py_policy

        # activate the session
        obs = {'depth': np.ones((1, 60, 80, 1)), 'sensor': np.ones((1, 26))}
        self.inference(obs)

    def inference(self, obs):
        import time
        start = time.time()
        with self.sess.as_default():
            time_step = TimeStep(
                np.ones(1),
                np.ones(1),
                np.ones(1),
                obs,
            )
            policy_state = ()
            action_step = self.eval_py_policy.action(time_step, policy_state)
            action = action_step.action[0]
            print('time', time.time() - start)
            return action


def main(_):
    tf.compat.v1.enable_resource_variables()
    logging.set_verbosity(logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    encoder_fc_layers = [256]
    actor_fc_layers = [256]
    critic_obs_fc_layers = [256]
    critic_action_fc_layers = [256]
    critic_joint_fc_layers = [256]

    print('conv_layer_params', conv_layer_params)
    print('encoder_fc_layers', encoder_fc_layers)
    print('actor_fc_layers', actor_fc_layers)
    print('critic_obs_fc_layers', critic_obs_fc_layers)
    print('critic_action_fc_layers', critic_action_fc_layers)
    print('critic_joint_fc_layers', critic_joint_fc_layers)

    engine = InferenceEngine(
        root_dir=FLAGS.root_dir,
        conv_layer_params=conv_layer_params,
        encoder_fc_layers=encoder_fc_layers,
        actor_fc_layers=actor_fc_layers,
        critic_obs_fc_layers=critic_obs_fc_layers,
        critic_action_fc_layers=critic_action_fc_layers,
        critic_joint_fc_layers=critic_joint_fc_layers,
        actor_learning_rate=0.0,
        critic_learning_rate=0.0,
        alpha_learning_rate=0.0,
        gamma=1.0,
    )

    obs = {'depth': np.ones((1, 60, 80, 1)), 'sensor': np.zeros((1, 26))}
    obs['depth'] *= 3.0
    for i in range(10):
        obs['sensor'][0, i * 2] = i * 0.2
    obs['sensor'][0, 20] = 5.0
    obs['sensor'][0, 21] = 0.0
    obs['sensor'][0, 22] = 0.5
    print(obs['sensor'])
    for _ in range(100):
        action = engine.inference(obs)
        print(action)

if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
