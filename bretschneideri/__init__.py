from bretschneideri.task import Task
from bretschneideri.agent import Agent
from bretschneideri.commandline import cmd_parser

agent = None

def launch(Task, config: dict = None):
  if config is None:
    config = cmd_parser.parse_args()
  agent = Agent(Task, **config)

  if agent.action == 'train':
    agent.train()
