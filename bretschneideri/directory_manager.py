import os
from shutil import rmtree

import simplejson as json
from collections import namedtuple

import torch

EpochSummary  = namedtuple('EpochSummary', ['timestamp', 'train', 'test'])
TrainingIndex = namedtuple('TrainingIndex', ['total_epoch', 'progress'])

class DirectoryManager:
  def __init__(self, workdir, resuming, overwriting):
    self.root = os.path.join(workdir, 'states')

    if resuming:
      if not os.path.isfile(self.index_path):
        raise Exception('Index file not found: %s' % self.index_path)
    elif os.path.isdir(self.root):
      if overwriting:
        rmtree(self.root)
        os.mkdir(self.root)
      else:
        raise Exception('Directory exists: %s' % self.root)
    else:
      os.mkdir(self.root)

  @property
  def index_path(self):
    return os.path.join(self.root, 'index.json')

  def load_index(self):
    with open(self.index_path, 'r') as fp:
      info = json.load(fp)
      info['progress'] = [EpochSummary(**kv) for kv in info['progress']]
      return TrainingIndex(**info)

  def save_index(self, index):
    with open(self.index_path, 'w') as fp:
      json.dump(index, fp, ignore_nan = True)

  def resource_path_for(self, id, key, padding = 4):
    id = str(id).zfill(padding)
    return os.path.join(self.root, '%s.%s' % (id, key))

  def load_torch_states(self, id, dict):
    for key in dict:
      path = self.resource_path_for(id, key)
      dict[key].load_state_dict(torch.load(path))

  def save_torch_states(self, id, dict):
    for key in dict:
      path = self.resource_path_for(id, key)
      torch.save(dict[key].state_dict(), path)
