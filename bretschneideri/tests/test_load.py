from unittest import TestCase
import tempfile
import simplejson as json

import torch

from bretschneideri import Task, launch
from bretschneideri.tests.model import json_config, TheTask

class TestDummy(TestCase):
  def test_overwriting(self):
    json_path = tempfile.mktemp()
    workdir   = tempfile.mkdtemp()
    with open(json_path, 'w') as fp:
      json.dump(json_config, fp)
    agent = launch(TheTask, {
      'config': json_path,
      'workdir': workdir,
      'n_epoch': 2
    })

    x, _ = agent.sample_test(1, agent.device)
    y    = (agent.model.eval())(x)

    # It should silently remove the directory.
    agent = launch(TheTask, {
      'config': json_path,
      'workdir': workdir,
      'n_epoch': 2,
      'action': 'load',
      'resuming': True
    })
    y_   = (agent.model.eval())(x)

    self.assertTrue(torch.all(torch.eq(y, y_)).item())
