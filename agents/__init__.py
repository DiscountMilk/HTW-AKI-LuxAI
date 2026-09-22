# Auto-discovery aller Submodule/-pakete unter 'agents'
from importlib import import_module
from pkgutil import walk_packages

for m in walk_packages(__path__, __name__ + "."):
    import_module(m.name)
