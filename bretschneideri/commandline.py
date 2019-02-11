import argparse

cmd_parser = argparse.ArgumentParser()

cmd_parser.add_argument('action', choices = { 'train', 'load' },
  help = 'Action to perform.')
cmd_parser.add_argument('config', type = str,
  help = 'Path to the JSON configuration file.')
cmd_parser.add_argument('workdir', type = str,
  help = 'Path to the working directory.')

cmd_parser.add_argument('--n_epoch', type = int, default = 200,
  help = 'Number of epochs to train. Default = 200.')
cmd_parser.add_argument('--resuming', action = 'store_true',
  help = 'Resume training from previous progress in the workdir.')
cmd_parser.add_argument('--overwriting', action = 'store_true',
  help = 'Overwriting previous workdir if it exists.')
