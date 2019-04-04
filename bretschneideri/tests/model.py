from bretschneideri import Task
from bretschneideri.utils import as_obj

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TheTask(Task):
  def __init__(self, config_dict):
    super(TheTask, self).__init__()

  def model(self, config_dict):
    shapes = as_obj(config_dict['shapes'])
    model  = nn.Linear(shapes.in_features, shapes.out_features, shapes.bias)
    return model

  def optim(self, config_dict, model):
    hypers = as_obj(config_dict['hypers'])
    return optim.SGD(model.parameters(), lr = hypers.lr)

  def sample(self, config_dict, training):
    shapes = as_obj(config_dict['shapes'])
    answer = nn.Linear(shapes.in_features, shapes.out_features, shapes.bias)
    error_scale = 1.0 if training else 0.0

    def fn(batch_size, device):
      x = torch.rand([batch_size, shapes.in_features], device = device)
      e = torch.randn([batch_size, shapes.out_features], device = device)
      y = answer(x) + e * error_scale
      return (x, y)

    return fn

  def train_batch(self, model, x, k):
    k_est = model(x)
    loss  = F.mse_loss(k_est, k)
    self.summary(loss = loss, k_est = k_est, k = k)
    return loss

  def test_batch(self, model, x, k):
    k_est = model(x)
    loss  = F.mse_loss(k_est, k)
    self.summary_field('field', loss = loss)

json_config = {
  'shapes': {
    'in_features': 10,
    'out_features': 5,
    'bias': True
  },
  'hypers': {
    'lr': 1e-3
  },
  'training_sizes' : {
    'epoch_size': 1,
    'batch_size': 8,
    'batch_multiplier': 2
  },
  'testing_sizes' : {
    'epoch_size': 1,
    'batch_size': 1
  }
}
