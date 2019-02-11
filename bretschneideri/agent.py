from time import time

import simplejson as json
from collections import namedtuple

import torch

from bretschneideri.utils import as_obj
from bretschneideri.directory_manager\
  import DirectoryManager, EpochSummary, TrainingIndex
from bretschneideri.task import Task

EpochSizes = namedtuple('EpochSizes',
  ['epoch_size', 'batch_size', 'batch_multiplier'])

def merge(list, reduction):
  tensor = torch.tensor(list, dtype = torch.float)
  if reduction == 'sum':
    return tensor.sum().item()
  else:
    return tensor.mean().item()

def merge_summary(xs, reduction):
  keys = xs[0].keys()
  return {
    key: merge([x[key] for x in xs], reduction.get(key))
    for key in keys
  }

class Agent():
  def __init__(self,
               Task,
               config: str,
               workdir: str,
               action = 'train',
               n_epoch = 200,
               resuming = False,
               overwriting = False):
    self.io = DirectoryManager(workdir, resuming, overwriting)

    if torch.cuda.is_available():
      self.device = torch.device('cuda')
    else:
      self.device = torch.device('cpu')

    with open(config, 'r') as fp:
      self.config_dict = json.load(fp)

    self.task         = Task(self.config_dict)
    self.model        = self.task.model(self.config_dict).to(self.device)
    self.optim        = self.task.optim(self.config_dict, self.model)
    self.sample_train = self.task.sample(self.config_dict, True)
    self.sample_test  = self.task.sample(self.config_dict, False)

    self.states_dict  = { 'model': self.model, 'optim': self.optim }

    if action == 'load' or resuming:
      self.index = self.io.load_index()
      self.io.load_torch_states(self.index.total_epoch - 1, self.states_dict)
    else:
      self.index = TrainingIndex(total_epoch = 0, progress = [])

    self.action  = action
    self.n_epoch = n_epoch

    training_sizes = self.config_dict.get('training_sizes')
    if training_sizes is None:
      training_sizes = {
        'epoch_size': 1024,
        'batch_size': 128,
        'batch_multiplier': 1
      }

    testing_sizes  = self.config_dict.get('testing_sizes')
    if testing_sizes is None:
      testing_sizes = training_sizes
    else:
      testing_sizes['batch_multiplier'] = 1

    self.training_sizes = EpochSizes(**training_sizes)
    self.testing_sizes  = EpochSizes(**testing_sizes)

  def update_index(self, train, test):
    self.index.progress.append(EpochSummary(time(), train, test))
    index = TrainingIndex(self.index.total_epoch + 1, self.index.progress)
    self.io.save_index(index)
    self.index = index

  def train_epoch(self, model, sizes):
    for i in range(1, 1 + sizes.epoch_size * sizes.batch_multiplier):
      self.task.summary_cache = {}

      batch, labels = self.sample_train(sizes.batch_size, self.device)
      loss = self.task.train_batch(model, batch, labels)
      loss.backward()
      if i % sizes.batch_multiplier == 0:
        self.optim.step()
      
      self.train_summaries.append(self.task.summary_cache)

  def test_epoch(self, model, sizes):
    for i in range(1, sizes.epoch_size + 1):
      self.task.summary_cache = {}

      batch, labels = self.sample_test(sizes.batch_size, self.device)
      self.task.test_batch(model, batch, labels)

      self.test_summaries.append(self.task.summary_cache)

  def train(self):
    self.optim.zero_grad()
    
    for i in range(self.index.total_epoch, self.n_epoch):
      self.train_summaries = []
      self.test_summaries  = []

      model = self.model.train()
      self.train_epoch(model, self.training_sizes)

      with torch.no_grad():
        self.test_epoch(model, self.testing_sizes)

      self.io.save_torch_states(self.index.total_epoch, self.states_dict)

      train = merge_summary(self.train_summaries, self.task.summary_reduction)
      test  = merge_summary(self.test_summaries,  self.task.summary_reduction)
      self.update_index(train, test)
