from unittest import TestCase
import tempfile
import simplejson as json

from bretschneideri import Task, launch
from bretschneideri.utils import as_obj

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

class TaskDescriptor(nn.Module):
  def __init__(self,
               normalizers,
               input_size,
               merge_inputs = 'cat'):
    self.normalizers = normalizers
    self.input_size  = input_size

    if merge_inputs != 'cat' and merge_inputs != 'sum':
      raise ValueError('Expected "cat" | "sum". Found "%s" instead.' % merge_inputs)
    self.merge_inputs = merge_inputs

class Picker(nn.Module):
  def __init__(self, descriptor: TaskDescriptor):
    super(Picker, self).__init__()
    self.__dict__.update(descriptor.__dict__)

  def prepare_inputs(self, inputs):
    sum = None
    for arg, normalize in zip(inputs, self.normalizers):
      if normalize is None:
        continue

      arg = normalize(arg)
      if sum is None:
        sum = arg
      elif self.merge_inputs == 'cat':
        sum = torch.cat([sum, arg], dim = -1)
      else:
        sum = sum + arg
    return sum

  def forward(self, inputs):
    input = self.prepare_inputs(inputs)
    return self.pick(input).squeeze(-1)

class LSTMPicker(Picker):
  def __init__(self,
               descriptor: TaskDescriptor,
               hidden_size,
               num_layers   = 1,
               train_hidden = False,
               train_cell   = False):
    super(LSTMPicker, self).__init__(descriptor)

    self.lstm   = nn.LSTM(input_size  = self.input_size,
                          hidden_size = hidden_size,
                          num_layers  = num_layers,
                          batch_first = True)
    self.decode = nn.Linear(hidden_size, 1)
    zeros       = torch.zeros(num_layers, 1, hidden_size)
    self.h_0    = nn.Parameter(zeros, requires_grad = train_hidden)
    self.c_0    = nn.Parameter(zeros, requires_grad = train_cell)

  def pick(self, inputs):
    batch_size = inputs.shape[0]
    h_0        = self.h_0.repeat(1, batch_size, 1)
    c_0        = self.c_0.repeat(1, batch_size, 1)
    outputs, _ = self.lstm(inputs, (h_0, c_0))
    return torch.sigmoid(self.decode(outputs))

def identity():
  return lambda x: x

def linear(weight, bias = 0):
  return lambda x: weight * x + bias

def normalize(p = 2, dim = 1, eps = 1e-12):
  return lambda x: F.normalize(x, p, dim, eps)

def wls(x, y, w, l2_reg = 0.0):
  w    = torch.diag_embed(w)
  xt   = x.transpose(dim0 = -2, dim1 = -1)
  xtw  = torch.bmm(xt, w)
  xtwx = torch.bmm(xtw, x)
  xtwy = torch.bmm(xtw, y)
  return torch.bmm(torch.inverse(xtwx + l2_reg), xtwy)

def ols(x, y, l2_reg = 0.0):
  xt   = x.transpose(dim0 = -2, dim1 = -1)
  xtx = torch.bmm(xt, x)
  xty = torch.bmm(xt, y)
  return torch.bmm(torch.inverse(xtx + l2_reg), xty)

def weight_loss(w):
  return torch.mean(torch.norm(w, p = 1, dim = -1))

def clip(weights, threshold):
  return weights * (weights >= threshold).type(torch.float)

def shuffle(weights):
  n_rows = weights.shape[0]
  n_cols = weights.shape[1]
  for i in range(n_rows):
    indices = torch.randperm(n_cols)
    weights[i] = (weights[i])[indices - 1]
  return weights

class MatchOLS(Task):
  def __init__(self, config_dict):
    super(MatchOLS, self).__init__()

    self.k_dist = dist.Uniform(5.0, 105.0)
    self.x_dist = dist.Uniform(3.0, 10.0)
    self.e_dist = dist.Normal(0.0, 0.25)

    self.shapes = as_obj(config_dict['shapes'])
    self.hypers = as_obj(config_dict['hypers'])
    self.l2_reg = self.hypers.l2_multiplier * self.shapes.x_dim
  
  def model(self, config_dict):
    x_dim   = self.shapes.x_dim
    y_mean  = self.k_dist.mean * self.x_dist.mean * x_dim
    scale   = (self.x_dist.mean / y_mean).item()

    normalizers = [identity(), linear(scale)]
    input_size  = x_dim + 1
    descriptor  = TaskDescriptor(normalizers, input_size)

    model = LSTMPicker(descriptor, **config_dict['net_params'])
    model.l2_reg    = self.l2_reg
    model.lp_coef   = 1.0
    model.lw_coef   = 1.0
    model.threshold = 0.1
    return model

  def optim(self, config_dict, model):
    return optim.Adam(model.parameters(), lr = self.hypers.lr)

  def sample(self, config_dict, training):
    seq_len = self.shapes.seq_len
    x_dim   = self.shapes.x_dim
    l2_reg  = self.l2_reg if training else 0
    estim   = lambda x, y: ols(x, y, l2_reg = l2_reg)

    def fn(batch_size, device):
      k = self.k_dist.sample([batch_size, x_dim, 1]).to(device)
      x = self.x_dist.sample([batch_size, seq_len, x_dim]).to(device)
      e = self.e_dist.sample([batch_size, seq_len, 1]).to(device)
      y = torch.bmm(x, k) + e

      batch  = (x, y)
      labels = estim(x, y)
      return (batch, labels)

    return fn

  def train_batch(self, model, batch, labels):
    x, y  = batch
    w     = model(batch)
    k_est = wls(x, y, w, l2_reg = model.l2_reg)

    lp    = F.mse_loss(k_est, labels)
    lw    = weight_loss(w)
    loss  = model.lp_coef * lp + model.lw_coef * lw
    self.summary(lp = lp, lw = lw, loss = loss)

    return loss

  def test_batch(self, model, batch, labels):
    x, y  = batch
    wo    = model(batch)
    wc    = clip(wo, threshold = model.threshold)
    lp    = F.mse_loss(wls(x, y, wc), labels)
    self.summary(wo = weight_loss(wo), wc = weight_loss(wc), lp = lp)

    w0    = shuffle(wc)
    l0    = F.mse_loss(wls(x, y, w0), labels)
    self.summary_field('comparsion', shuffled = l0 / lp)

json_config = {
  'shapes': {
    'seq_len': 128,
    'x_dim': 2
  },
  'hypers': {
    'l2_multiplier': 50,
    'lr': 1e-5
  },
  'net_params': {
    'hidden_size': 128,
    'num_layers': 2
  },
  'training_sizes' : {
    'epoch_size': 4,
    'batch_size': 8,
    'batch_multiplier': 2
  },
  'testing_sizes' : {
    'epoch_size': 4,
    'batch_size': 4
  }
}

class TestDummy(TestCase):
  def test_runnable(self):
    json_path = tempfile.mktemp()
    workdir   = tempfile.mkdtemp()
    with open(json_path, 'w') as fp:
      json.dump(json_config, fp)
    launch(MatchOLS, {
      'config': json_path,
      'workdir': workdir,
      'n_epoch': 2
    })
    self.assertTrue(True)
