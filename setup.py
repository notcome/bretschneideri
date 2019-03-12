import os
from setuptools import setup

setup(
  name     = 'Bretschneideri',
  version  = '0.1.4',
  packages = ['bretschneideri'],
  url      = 'https://github.com/notcome/bretschneideri',

  author       = 'Liu, Minsheng',
  author_email = 'lambda@liu.ms',
  license      = 'MIT',

  long_description = open('README.md').read(),

  test_suite    = 'nose.collector',
  tests_require = ['nose'],

  install_requires = ['simplejson']
)
