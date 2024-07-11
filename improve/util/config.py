
import inspect
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field

def default(data):
    return field(default_factory=lambda: data)

cs = ConfigStore.instance()


def store(cls):
    """
    @store will call
    cs.store(node=<class type>, name=<filename with no extension>, group=<dirname>)
    """

    def wrapper(cls):
        tree = inspect.getfile(cls).split(".")[0].split("/")
        name = cls.name # tree[-1] 
        base = tree.index("cn")
        group = '/'.join(tree[base + 1:-1])
        print(name, group)
        cs.store(name=name, node=cls, group=group)
        return cls

    return wrapper(cls)

