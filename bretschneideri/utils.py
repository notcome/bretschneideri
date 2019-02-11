import types

def as_obj(dict):
  if type(dict) is list:
    return dict

  obj = types.SimpleNamespace()
  obj.__dict__.update(dict)
  return obj
