from unittest import TestCase
import tempfile
import simplejson as json

from bretschneideri import Task, launch
from bretschneideri.tests.model import json_config, MatchOLS

class TestDummy(TestCase):
  def test_overwriting(self):
    json_path = tempfile.mktemp()
    workdir   = tempfile.mkdtemp()
    with open(json_path, 'w') as fp:
      json.dump(json_config, fp)
    launch(MatchOLS, {
      'config': json_path,
      'workdir': workdir,
      'n_epoch': 2
    })
    # It should silently remove the directory.
    launch(MatchOLS, {
      'config': json_path,
      'workdir': workdir,
      'n_epoch': 2,
      'overwriting': True
    })
    self.assertTrue(True)
