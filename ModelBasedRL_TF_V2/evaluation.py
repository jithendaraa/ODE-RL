import argparse
import collections
import functools
import os
import pathlib
import sys
import warnings
import pickle
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*TensorFloat-32 matmul/conv*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
from tqdm import tqdm
#import wandb

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import datasets


class Dreamer(tools.Module):

  def __init__(self, config, logger, dataset, task, n_classes):
    # print("Dreamer: start init")
    self._config = config
    self._logger = logger
    self._float = prec.global_policy().compute_dtype
    self._should_log = tools.Every(config.log_every)
    self._should_log_video = tools.Every(config.log_video_every)
    self._should_train = tools.Every(config.train_every)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)
    self._should_expl = tools.Until(int(
        config.expl_until / config.action_repeat))
    self._metrics = collections.defaultdict(tf.metrics.Mean)
    with tf.device('cpu:0'):
      self._step = tf.Variable(0, dtype=tf.int64)
    self._dataset = iter(dataset)
    self._task = task
    self._wm = models.WorldModel(self._step, config)
    if self._task != 0:
      self._ac = models.ConvGRUClassifier(config, self._wm, task, n_classes)
      if self._task == 1 or self._task == 2:
        self._metric = tools.mAP(n_classes, True)
      else:
        self._metric = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
    self._train(next(self._dataset))

  def break_batch(self, data):
      B = data['obs'].shape[0]
      T = data['obs'].shape[1]
      t = self._config.batch_length
      n = T // t
      batch = {}
      batch['obs'] = tf.reshape(data['obs'], [n * B, t] + list(data['obs'].shape[-3:]))
      # n = [t] * (T // t) + [T % t] if T % t else [t] * (T // t)
      # batches = tf.stack([{'obs': sub_obs, 'label': data['label']} for sub_obs in tf.split(data['obs'], num_or_size_splits=n, axis=1)])
      labels = data['label']
      return batch, labels

  def __call__(self):
    self._train(next(self._dataset))
    step = self._step.numpy().item()

    if self._should_log(step):
      for name, mean in self._metrics.items():
        self._logger.scalar(name, float(mean.result()))
        mean.reset_states()

      if self._should_log_video(step):
        for _ in range(self._config.train_iter):
          data = next(self._dataset)
          if self._task != 0:
            data, labels = self.break_batch(data)
            preds = self._ac.action_pred(data)
            self._metric.update_state(labels, preds)
        if self._task != 0:
          self._logger.scalar('train_classifier_metric', self._metric.result())
          self._metric.reset_states()
        openl, recon, openl_val, recon_val = self._wm.video_pred(data)
        self._logger.plot('train_psnr_comparision', tools.cal_psnr_graph_openl(openl_val, recon_val, 'train_psnr_comparision'))
        self._logger.video('train_openl', openl)
        self._logger.video('train_recon', recon)

      self._logger.write(fps=True)

    self._step.assign_add(1)
    self._logger.step = self._step.numpy().item()

  @tf.function
  def _train(self, data):
    #  B, 300, H, W, c
    tqdm.write('Tracing train function.')
    metrics = {}
    if self._task != 0:
        data, labels = self.break_batch(data)
    _, context, mets = self._wm.train(data)
    posteriors=context['feat']
    metrics.update(mets)
    if self._task != 0:
      mets = self._ac.train(posteriors[:, -1], labels)
      metrics.update(mets)

    for name, value in metrics.items():
      self._metrics[name].update_state(value)


# def count_steps(folder):
#   return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))

def make_dataset(episodes, config, batch_length):
  generator = lambda: tools.sample_episodes(episodes, batch_length)
  example = next(iter(generator()))
  types = {k: v.dtype for k, v in example.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in example.items()}
  # print (types, shapes)
  dataset = tf.data.Dataset.from_generator(generator, types, shapes)
  dataset = dataset.batch(config.batch_size, drop_remainder=True)
  dataset = dataset.prefetch(10)
  return dataset

def main(config):
  print(config.id)
  folder_name = f"{config.task}/{config.id}/seed{config.seed}"
  logdir = pathlib.Path(config.logdir).expanduser() / folder_name

  tqdm.write(f'Eval Logdir: {logdir}')

  config.act = getattr(tf.nn, config.act)

  message = 'No GPU found. To actually train on CPU remove this assert.'
  assert tf.config.experimental.list_physical_devices('GPU'), message
  if config.gpu_growth:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    prec.set_policy(prec.Policy('mixed_float16'))

  logger = tools.Logger(logdir, 0)

  dataset_name, task = config.task.split('_')

  if dataset_name == 'CATER':
    raise NotImplementedError(dataset_name)
    train_eps, eval_eps, batch_length, n_classes = datasets.get_cater_data(task, config)
  elif dataset_name == 'MovingMNIST':
    train_eps, eval_eps, batch_length, n_classes = datasets.get_mm_gen_data(task, config)
    size = 2000 // config.batch_size
  train_dataset = make_dataset(train_eps, config, batch_length)
  if hasattr(config, 'eval_batch_length'):
    batch_length = config.eval_batch_length
  eval_dataset = iter(make_dataset(eval_eps, config, batch_length))
  agent = Dreamer(config, logger, train_dataset, int(task), n_classes)
  assert (logdir / 'variables.pkl').exists(), "Saved model not found."
  agent.load(logdir / 'variables.pkl')
  agent._should_pretrain._once = False

  if int(task) != 0:
    if int(task) == 1 or int(task) == 2:
      eval_metric = tools.mAP(n_classes, True)
    else:
      eval_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=5)

  psnr_vals = []
  for _ in tqdm(range(size)):
    data = next(eval_dataset)
    eval_openl, eval_recon, eval_openl_val, eval_recon_val = agent._wm.video_pred(data, nenvs=config.batch_size)
    x, y = eval_openl_val
    psnr = tf.image.psnr(x, y, 1.0)
    psnr_vals.append(psnr)


  psnrtf = tf.stack(psnr_vals, 1)
  psnrtf = tf.reshape(psnrtf, [-1, 200])
  psnrtf_mean = tf.math.reduce_mean(psnrtf, 0)
  psnrtf_std = tf.math.reduce_std(psnrtf, 0)
  n_frames = psnrtf_mean.shape[0]
  plt.plot(np.arange(n_frames), psnrtf_mean.numpy(), label='Openl')
  plt.fill_between(np.arange(n_frames), psnrtf_mean.numpy()+psnrtf_std.numpy(), psnrtf_mean.numpy()-psnrtf_std.numpy(), alpha=0.7)
  plt.xlabel('Frames')
  plt.ylabel('PSNR')
  plt.xlim((0, n_frames))
  plt.title(f'{config.id}: ({psnrtf_mean.numpy()[-1]:.2f}+{psnrtf_std.numpy()[-1]:.2f})')
  plt.grid()
  plt.tight_layout()
  plt.legend()
  plt.savefig(f'out/eval_{config.id}_psnr_full.png')

  # with open('psnrvals.pkl', 'wb') as fp:
  #   pickle.dump(psnr_vals, fp)

  # should_eval = tools.Every(config.log_video_every)
  # should_save = tools.Every(config.log_video_every)
  # logger.step = agent._step.numpy().item()
  # if int(task) != 0:
  #   if int(task) == 1 or int(task) == 2:
  #     eval_metric = tools.mAP(n_classes, True)
  #   else:
  #     eval_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=5)

  # initial=agent._step.numpy().item()
  # end = config.steps + 5
  # pbar = tqdm(total=end, initial=initial)
  # while agent._step.numpy().item() < end:
  #   logger.write()
  #   if should_eval(agent._step.numpy().item()):
  #     tqdm.write('Start evaluation.')
  #     for _ in range(config.eval_iter):
  #       data = next(eval_dataset)
  #       if int(task) != 0:
  #         data, labels = agent.break_batch(data)
  #         preds = agent._ac.action_pred(data)
  #         eval_metric.update_state(labels, preds)
  #     if int(task) != 0:
  #       logger.scalar('eval_metric', eval_metric.result())
  #       eval_metric.reset_states()
  #     eval_openl, eval_recon, eval_openl_val, eval_recon_val = agent._wm.video_pred(data)
  #     logger.plot('eval_psnr_comparision', tools.cal_psnr_graph_openl(eval_openl_val, eval_recon_val, 'eval_psnr_comparision'))
  #     logger.video('eval_openl', eval_openl)
  #     logger.video('eval_recon', eval_recon)
  #   agent()
  #   if should_save(agent._step.numpy().item()):
  #     agent.save(logdir / 'variables.pkl')
  #   pbar.update(agent._step.numpy().item() - initial)
  #   initial=agent._step.numpy().item()
  # pbar.close()


if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument('--logdir', required=True)
  # parser.add_argument('--task', required=True)
  # parser.add_argument('--id', required=True)
  # parser.add_argument('--configs', required=True)
  # args, remaining = parser.parse_known_args()
  # config_path = os.path.join(args.logdir, args.task, args.id, 'seed0', 'configs.yaml')
  # print(config_path)
  # assert os.path.exists(config_path), "Config file for the run does not exist."
  # configs = yaml.safe_load((pathlib.Path(config_path)).read_text())
  # defaults = configs

  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', required=True)
  args, remaining = parser.parse_known_args()
  configs = yaml.safe_load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
  parser = argparse.ArgumentParser()
  for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
  main(parser.parse_args(remaining))
