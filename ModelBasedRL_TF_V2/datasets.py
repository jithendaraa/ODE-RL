import os
import tools
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.video import moving_sequence
import numpy as np
from tqdm.auto import tqdm

def get_cater_data(task, config):
  assert task in ['0', '1', '2', '3'], "Invalid Task"
  n_classes = {
    '0': 0,
    '1': 14,
    '2': 301,
    '3': 36,
  }[task]
  task_dir = {
    '0': ['all_actions', 'localize'],
    '1': ['max2action', 'actions_present'],
    '2': ['max2action', 'actions_order_uniq'],
    '3': ['all_actions', 'localize'],
  }[task]

  if task == '0':
    assert config.video_data in ['all_actions', 'max2action'], "Invalid Dataset for Video Reconstruction"
    task_dir = {
      'all_actions': ['all_actions', 'localize'],
      'max2action': ['max2action', 'actions_present'],
    }[config.video_data]

  data_dir = os.path.join('data', 'CATER')
  dataset_dir = os.path.join(data_dir, task_dir[0])
  vid_dir = os.path.join(dataset_dir, 'videos/')
  label_dir = os.path.join(dataset_dir, 'lists', task_dir[1])
  print(f"Dataset Directory: {dataset_dir}")
  print(f"Video Directory: {vid_dir}")
  print(f"Label Directory: {label_dir}")

  train_fpath = os.path.join(label_dir, 'train.txt')
  train_eps = {}

  train_eps['labels'] = tools.load_labels(train_fpath, n_classes)

  if config.encoder_model in ['resnet18', 'resnet34', 'resnet50']:
    config.size = (224, 224)
    config.save_eps_dict = False
    config.dyn_spatial = 7

  if config.lazy_load:
    train_eps['episodes'] = tools.load_episodes(vid_dir, list(train_eps['labels'].keys()), 30, size=config.size)
  else:
    saved_train_eps_path = os.path.join(dataset_dir, f'train_eps_{task_dir[0]}.pkl')
    if os.path.exists(saved_train_eps_path) and config.size[0] == 64:
      print(f"Found saved train episodes at {saved_train_eps_path}. Loading..")
      with open(saved_train_eps_path, 'rb') as writer:
        train_eps['episodes'] = pickle.load(writer)
    else:
      train_eps['episodes'] = tools.load_episodes(vid_dir, list(train_eps['labels'].keys()), size=config.size)
      if config.save_eps_dict:
        print(f"Saving train episodes dict at {saved_train_eps_path}.")
        with open(saved_train_eps_path, 'wb') as writer:
          pickle.dump(train_eps['episodes'], writer)

  eval_fpath = os.path.join(label_dir, 'val.txt')
  eval_eps = {}

  eval_eps['labels'] = tools.load_labels(eval_fpath, n_classes)

  if config.lazy_load:
    eval_eps['episodes'] = tools.load_episodes(vid_dir, list(eval_eps['labels'].keys()), 30, size=config.size)
  else:
    saved_eval_eps_path = os.path.join(dataset_dir, f'eval_eps_{task_dir[0]}.pkl')
    if os.path.exists(saved_eval_eps_path) and config.size[0] == 64:
      print(f"Found saved eval episodes at {saved_eval_eps_path}. Loading..")
      with open(saved_eval_eps_path, 'rb') as writer:
        eval_eps['episodes'] = pickle.load(writer)
    else:
      eval_eps['episodes'] = tools.load_episodes(vid_dir, list(eval_eps['labels'].keys()), size=config.size)
      if config.save_eps_dict:
        print(f"Saving eval episodes dict at {saved_eval_eps_path}.")
        with open(saved_eval_eps_path, 'wb') as writer:
          pickle.dump(eval_eps['episodes'], writer)

  if int(task) == 0:
    batch_length = config.batch_length
  else:
    assert 300 % config.batch_length == 0, "Batch length should be a factor of 300"
    batch_length = 300

  return train_eps, eval_eps, batch_length

# DOwnload mnist dataaset 
def get_mm_data(task, config):
  assert task == '0', 'Only video prediction task available.'
  dataset_dir = os.path.join('data', 'MovingMNIST')
  vid_path = os.path.join(dataset_dir, 'mnist_test_seq.npy')
  print(f"Dataset Directory: {dataset_dir}")
  print(f"Video Path: {vid_path}")
  x = np.load(vid_path)
  x = x.transpose((1, 0, 2, 3))[..., None]
  size = x.shape[0]
  indices = np.arange(size)
  np.random.shuffle(indices)
  train_eps = {}
  train_frac = 0.8
  train_eps['episodes'] = {str(indices[i]): x[i] for i in range(int(train_frac * size))}
  train_eps['labels'] = {str(indices[i]): np.array([indices[i]]) for i in range(int(train_frac * size))}
  eval_eps = {}
  eval_eps['episodes'] = {str(indices[i]): x[i] for i in range(int(train_frac * size), size)}
  eval_eps['labels'] = {str(indices[i]): np.array([indices[i]]) for i in range(int(train_frac * size), size)}
  return train_eps, eval_eps, config.batch_length, 0

def get_mm_gen_data(task, config):
  assert task == '0', 'Only video prediction task available.'
  dataset_dir = os.path.join('data', 'MovingMNIST')
  train_eps_path = os.path.join(dataset_dir, 'train_eps_mm.pkl')
  eval_eps_path = os.path.join(dataset_dir, 'eval_eps_mm.pkl')

  print(f"Dataset Directory: {dataset_dir}")
  if not os.path.exists(train_eps_path):
    print("Train Dataset not found. Generating...")
    generate_mm_sequences(dataset_dir, split='train', size=8000)
  if not os.path.exists(eval_eps_path):
    print("Eval Dataset not found. Generating...")
    generate_mm_sequences(dataset_dir, split='test', size=2000)

  print(f"Train Episode Path: {train_eps_path}")
  print(f"Eval Episode Path: {eval_eps_path}")

  with open(train_eps_path, 'rb') as writer:
      train_eps = pickle.load(writer)

  with open(eval_eps_path, 'rb') as writer:
      eval_eps = pickle.load(writer)
  return train_eps, eval_eps, config.batch_length, 0

def make_mm_dataset(split='train', sequence_length=200, output_size=(64, 64), velocity=0.1):
  def map_fn(sequence_length=sequence_length, output_size=output_size, velocity=velocity):
    def create_seq(image, label):
      sequence = moving_sequence.image_as_moving_sequence(image, sequence_length=sequence_length, output_size=output_size, velocity=velocity)
      return sequence.image_sequence
    return create_seq

  mnist_ds = tfds.load("mnist", split=split, as_supervised=True, shuffle_files=True)
  mnist_ds = mnist_ds.repeat().shuffle(1024)

  moving_mnist_ds = mnist_ds.map(map_fn(sequence_length=sequence_length, output_size=output_size, velocity=velocity))
  moving_mnist_ds = moving_mnist_ds.batch(2).map(lambda x: dict(image_sequence=tf.reduce_max(x, axis=0)))
  return moving_mnist_ds

def generate_mm_sequences(save_dir, split='train', size=8000, sequence_length=200, output_size=(64, 64), velocity=0.1):
  moving_mnist_ds = make_mm_dataset(split=split, sequence_length=sequence_length, output_size=output_size, velocity=velocity)
  # size = min(len(moving_mnist_ds), size)
  # if len(moving_mnist_ds) < size:
  #   print(f"Insufficient files for creating dataset. Needed {size}, found {len(moving_mnist_ds)}")
  indices = np.arange(size)
  eps_dict = {}
  eps_dict['episodes'] = {str(indices[i]): next(iter(moving_mnist_ds))['image_sequence'] for i in tqdm(range(size))}
  eps_dict['labels'] = {str(indices[i]): np.array([indices[i]]) for i in range(size)}

  fname = split if split == 'train' else 'eval'
  with open(os.path.join(save_dir, f'{fname}_eps_mm.pkl'), 'wb') as writer:
      pickle.dump(eps_dict, writer)