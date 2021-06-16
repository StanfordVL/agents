# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
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

r"""Train and Eval PPO.

To run:

```bash
tensorboard --logdir $HOME/tmp/ppo_v1/gym/HalfCheetah-v2/ --port 2223 &

python tf_agents/agents/ppo/examples/v1/train_eval.py \
  --root_dir=$HOME/tmp/ppo_v1/gym/HalfCheetah-v2/ \
  --alsologtostderr
```
"""

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

from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_gibson
from tf_agents.environments import suite_behavior_mp
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import py_metrics
from tf_agents.metrics import batched_py_metric
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.networks.utils import mlp_layers
from tf_agents.policies import py_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.utils import episode_utils


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('replay_buffer_capacity', 501,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer('num_parallel_environments', 1,
                     'Number of environments to run in parallel')
flags.DEFINE_integer('num_parallel_environments_eval', 1,
                     'Number of environments to run in parallel for eval')
flags.DEFINE_integer('num_environment_steps', 10000000,
                     'Number of environment steps to run before finishing.')
flags.DEFINE_integer('num_epochs', 25,
                     'Number of epochs for computing policy updates.')
flags.DEFINE_integer(
    'collect_episodes_per_iteration', 1,
    'The number of episodes to take in the environment before '
    'each update. This is for each parallel environments.')
flags.DEFINE_boolean('use_rnns', False,
                     'If true, use RNN for policy and value function.')
flags.DEFINE_float('learning_rate', 1e-4,
                   'learning rate')
flags.DEFINE_integer('num_eval_episodes', 10,
                     'The number of episodes to run eval on.')
flags.DEFINE_integer('eval_interval', 10000,
                     'Run eval every eval_interval train steps')
flags.DEFINE_boolean('eval_only', False,
                     'Whether to run evaluation only on trained checkpoints')
flags.DEFINE_boolean('eval_deterministic', False,
                     'Whether to run evaluation using a deterministic policy')
flags.DEFINE_boolean('motion_planning', False,
                     'Whether to use motion_planning')
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
flags.DEFINE_string('action_filter', 'mobile_manipulation',
                    'action filter')
flags.DEFINE_string('env_mode', 'headless',
                    'Mode for the simulator (gui or headless)')
flags.DEFINE_string('env_type', 'gibson',
                    'Type for the Gibson environment (gibson or ig)')
flags.DEFINE_float('action_timestep', 1.0 / 300.0,
                   'Action timestep for the simulator')
flags.DEFINE_float('physics_timestep', 1.0 / 300.0,
                   'Physics timestep for the simulator')
flags.DEFINE_integer('gpu_g', 0,
                     'GPU id for graphics, e.g. Gibson.')
flags.DEFINE_boolean('random_position', False,
                     'Whether to randomize initial and target position')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
        root_dir,
        gpu='1',
        env_load_fn=None,
        model_ids=None,
        eval_env_mode='headless',
        conv_1d_layer_params=None,
        conv_2d_layer_params=None,
        encoder_fc_layers=[256],
        actor_fc_layers=[256, 256],
        value_fc_layers=[256, 256],
        use_rnns=False,
        # Params for collect
        num_environment_steps=10000000,
        collect_episodes_per_iteration=30,
        num_parallel_environments=30,
        replay_buffer_capacity=1001,  # Per-environment
        # Params for train
        num_epochs=25,
        learning_rate=1e-4,
        # Params for eval
        num_eval_episodes=30,
        eval_interval=500,
        eval_only=False,
        eval_deterministic=False,
        num_parallel_environments_eval=1,
        model_ids_eval=None,
        # Params for summaries and logging
        train_checkpoint_interval=500,
        policy_checkpoint_interval=500,
        rb_checkpoint_interval=500,
        log_interval=10,
        summary_interval=50,
        summaries_flush_secs=1,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        eval_metrics_callback=None):
    """A simple train and eval for PPO."""
    if root_dir is None:
        raise AttributeError('train_eval requires a root_dir.')


    gpus = tf.config.experimental.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        batched_py_metric.BatchedPyMetric(
            py_metrics.AverageReturnMetric,
            metric_args={'buffer_size': num_eval_episodes},
            batch_size=num_parallel_environments_eval),
        batched_py_metric.BatchedPyMetric(
            py_metrics.AverageEpisodeLengthMetric,
            metric_args={'buffer_size': num_eval_episodes},
            batch_size=num_parallel_environments_eval),
    ]
    eval_summary_writer_flush_op = eval_summary_writer.flush()
    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):
        if model_ids is None:
            model_ids = [None] * num_parallel_environments
        else:
            assert len(model_ids) == num_parallel_environments,\
                'model ids provided, but length not equal to num_parallel_environments'

        if model_ids_eval is None:
            model_ids_eval = [None] * num_parallel_environments_eval
        else:
            assert len(model_ids_eval) == num_parallel_environments_eval,\
                'model ids eval provided, but length not equal to num_parallel_environments_eval'

        tf_py_env = [lambda model_id=model_ids[i]: env_load_fn(model_id, 'headless', gpu)
                     for i in range(num_parallel_environments)]
        tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(tf_py_env))

        if eval_env_mode == 'gui':
            assert num_parallel_environments_eval == 1, 'only one GUI env is allowed'
        eval_py_env = [lambda model_id=model_ids_eval[i]: env_load_fn(model_id, eval_env_mode, gpu)
                       for i in range(num_parallel_environments_eval)]
        eval_py_env = parallel_py_environment.ParallelPyEnvironment(eval_py_env)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        time_step_spec = tf_env.time_step_spec()
        observation_spec = tf_env.observation_spec()
        action_spec = tf_env.action_spec()
        print('observation_spec', observation_spec)
        print('action_spec', action_spec)

        glorot_uniform_initializer = tf.compat.v1.keras.initializers.glorot_uniform()
        preprocessing_layers = {}
        if 'rgb' in observation_spec:
            preprocessing_layers['rgb'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=conv_2d_layer_params,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'depth' in observation_spec:
            preprocessing_layers['depth'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=conv_2d_layer_params,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'scan' in observation_spec:
            preprocessing_layers['scan'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=conv_1d_layer_params,
                conv_2d_layer_params=None,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'task_obs' in observation_spec:
            preprocessing_layers['task_obs'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=None,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))
        if 'proprioception' in observation_spec:
            preprocessing_layers['proprioception'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=None,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if len(preprocessing_layers) <= 1:
            preprocessing_combiner = None
        else:
            preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

        if use_rnns:
            actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                observation_spec,
                action_spec,
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                input_fc_layer_params=actor_fc_layers,
                output_fc_layer_params=None)
            value_net = value_rnn_network.ValueRnnNetwork(
                observation_spec,
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                input_fc_layer_params=value_fc_layers,
                output_fc_layer_params=None)
        else:
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                observation_spec,
                action_spec,
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                fc_layer_params=actor_fc_layers,
                kernel_initializer=glorot_uniform_initializer
            )
            value_net = value_network.ValueNetwork(
                observation_spec,
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                fc_layer_params=value_fc_layers,
                kernel_initializer=glorot_uniform_initializer
            )

        tf_agent = ppo_agent.PPOAgent(
            time_step_spec,
            action_spec,
            optimizer,
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=num_epochs,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity)

        if eval_deterministic:
            eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy)
        else:
            eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.collect_policy)

        environment_steps_metric = tf_metrics.EnvironmentSteps()
        environment_steps_count = environment_steps_metric.result()
        step_metrics = [
            tf_metrics.NumberOfEpisodes(),
            environment_steps_metric,
        ]
        train_metrics = step_metrics + [
            tf_metrics.AverageReturnMetric(
                buffer_size=100,
                batch_size=num_parallel_environments),
            tf_metrics.AverageEpisodeLengthMetric(
                buffer_size=100,
                batch_size=num_parallel_environments),
        ]

        # Add to replay buffer and other agent specific observers.
        replay_buffer_observer = [replay_buffer.add_batch]

        collect_policy = tf_agent.collect_policy

        collect_op = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            collect_policy,
            observers=replay_buffer_observer + train_metrics,
            num_episodes=collect_episodes_per_iteration * num_parallel_environments).run()

        trajectories = replay_buffer.gather_all()

        train_op, _ = tf_agent.train(experience=trajectories)

        with tf.control_dependencies([train_op]):
            clear_replay_op = replay_buffer.clear()

        with tf.control_dependencies([clear_replay_op]):
            train_op = tf.identity(train_op)

        train_checkpointer = common.Checkpointer(
            ckpt_dir=train_dir,
            agent=tf_agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'policy'),
            policy=tf_agent.policy,
            global_step=global_step)
        rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
            max_to_keep=1,
            replay_buffer=replay_buffer)

        summary_ops = []
        for train_metric in train_metrics:
            summary_ops.append(train_metric.tf_summaries(
                train_step=global_step, step_metrics=step_metrics))

        with eval_summary_writer.as_default(), tf.compat.v2.summary.record_if(True):
            for eval_metric in eval_metrics:
                eval_metric.tf_summaries(
                    train_step=global_step, step_metrics=step_metrics)

        init_agent_op = tf_agent.initialize()

        with sess.as_default():
            # Initialize graph.
            train_checkpointer.initialize_or_restore(sess)
            rb_checkpointer.initialize_or_restore(sess)

            if eval_only:
                metric_utils.compute_summaries(
                    eval_metrics,
                    eval_py_env,
                    eval_py_policy,
                    num_episodes=num_eval_episodes,
                    global_step=0,
                    callback=eval_metrics_callback,
                    tf_summaries=False,
                    log=True,
                )
                #episodes = eval_py_env.get_stored_episodes()
                #episodes = [episode for sublist in episodes for episode in sublist][:num_eval_episodes]
                #metrics = episode_utils.get_metrics(episodes)
                #for key in sorted(metrics.keys()):
                #    print(key, ':', metrics[key])

                #save_path = os.path.join(eval_dir, 'episodes_eval.pkl')
                #episode_utils.save(episodes, save_path)
                #print('EVAL DONE')
                return

            common.initialize_uninitialized_variables(sess)
            sess.run(init_agent_op)
            sess.run(train_summary_writer.init())
            sess.run(eval_summary_writer.init())

            collect_time = 0
            train_time = 0
            timed_at_step = sess.run(global_step)
            steps_per_second_ph = tf.compat.v1.placeholder(
                tf.float32, shape=(), name='steps_per_sec_ph')
            steps_per_second_summary = tf.compat.v2.summary.scalar(
                name='global_steps_per_sec', data=steps_per_second_ph,
                step=global_step)

            global_step_val = sess.run(global_step)
            while sess.run(environment_steps_count) < num_environment_steps:
                global_step_val = sess.run(global_step)
                if global_step_val % eval_interval == 0:
                    metric_utils.compute_summaries(
                        eval_metrics,
                        eval_py_env,
                        eval_py_policy,
                        num_episodes=num_eval_episodes,
                        global_step=global_step_val,
                        callback=eval_metrics_callback,
                        log=True,
                    )
                    # with eval_summary_writer.as_default(), tf.compat.v2.summary.record_if(True):
                        #with tf.name_scope('Metrics/'):
                        #    episodes = eval_py_env.get_stored_episodes()
                        #    episodes = [episode for sublist in episodes for episode in sublist][:num_eval_episodes]
                        #    metrics = episode_utils.get_metrics(episodes)
                        #    for key in sorted(metrics.keys()):
                        #        print(key, ':', metrics[key])
                        #        metric_op = tf.compat.v2.summary.scalar(name=key,
                        #                                                data=metrics[key],
                        #                                                step=global_step_val)
                        #        sess.run(metric_op)
                    sess.run(eval_summary_writer_flush_op)

                start_time = time.time()
                sess.run(collect_op)
                collect_time += time.time() - start_time
                start_time = time.time()
                total_loss, _ = sess.run([train_op, summary_ops])
                train_time += time.time() - start_time

                global_step_val = sess.run(global_step)
                if global_step_val % log_interval == 0:
                    logging.info('step = %d, loss = %f', global_step_val, total_loss)
                    steps_per_sec = (
                            (global_step_val - timed_at_step) / (collect_time + train_time))
                    logging.info('%.3f steps/sec', steps_per_sec)
                    sess.run(
                        steps_per_second_summary,
                        feed_dict={steps_per_second_ph: steps_per_sec})
                    logging.info('%s', 'collect_time = {}, train_time = {}'.format(
                        collect_time, train_time))
                    timed_at_step = global_step_val
                    collect_time = 0
                    train_time = 0

                if global_step_val % train_checkpoint_interval == 0:
                    train_checkpointer.save(global_step=global_step_val)

                if global_step_val % policy_checkpoint_interval == 0:
                    policy_checkpointer.save(global_step=global_step_val)

                if global_step_val % rb_checkpoint_interval == 0:
                    rb_checkpointer.save(global_step=global_step_val)

            # One final eval before exiting.
            metric_utils.compute_summaries(
                eval_metrics,
                eval_py_env,
                eval_py_policy,
                num_episodes=num_eval_episodes,
                global_step=global_step_val,
                callback=eval_metrics_callback,
                log=True,
            )
            sess.run(eval_summary_writer_flush_op)

        sess.close()


def main(_):
    tf.compat.v1.enable_resource_variables()
    if tf.executing_eagerly():
        # self.skipTest('b/123777119')  # Secondary bug: ('b/123775375')
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_c)

    # conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    conv_1d_layer_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    conv_2d_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)]
    encoder_fc_layers = [256]
    actor_fc_layers = [256]
    value_fc_layers = [256]

    for k, v in FLAGS.flag_values_dict().items():
        print(k, v)
    print('conv_1d_layer_params', conv_1d_layer_params)
    print('conv_2d_layer_params', conv_2d_layer_params)
    print('encoder_fc_layers', encoder_fc_layers)
    print('actor_fc_layers', actor_fc_layers)
    print('value_fc_layers', value_fc_layers)

    logging.set_verbosity(logging.INFO)
    train_eval(
        FLAGS.root_dir,
        gpu=FLAGS.gpu_g,
        env_load_fn=lambda model_id, mode, device_idx: suite_behavior_mp.load(
            config_file=FLAGS.config_file,
            model_id=model_id,
            env_mode=mode,
            device_idx=device_idx,
            action_filter=FLAGS.action_filter,
            use_motion_planning=FLAGS.motion_planning
        ),
        model_ids=FLAGS.model_ids,
        eval_env_mode=FLAGS.env_mode,
        conv_1d_layer_params=conv_1d_layer_params,
        conv_2d_layer_params=conv_2d_layer_params,
        encoder_fc_layers=encoder_fc_layers,
        actor_fc_layers=actor_fc_layers,
        value_fc_layers=value_fc_layers,
        use_rnns=FLAGS.use_rnns,
        num_environment_steps=FLAGS.num_environment_steps,
        collect_episodes_per_iteration=FLAGS.collect_episodes_per_iteration,
        num_parallel_environments=FLAGS.num_parallel_environments,
        replay_buffer_capacity=FLAGS.replay_buffer_capacity,
        num_epochs=FLAGS.num_epochs,
        learning_rate=FLAGS.learning_rate,
        num_eval_episodes=FLAGS.num_eval_episodes,
        eval_interval=FLAGS.eval_interval,
        eval_only=FLAGS.eval_only,
        eval_deterministic=FLAGS.eval_deterministic,
        num_parallel_environments_eval=FLAGS.num_parallel_environments_eval,
        model_ids_eval=FLAGS.model_ids_eval,
    )


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('config_file')
    app.run(main)
