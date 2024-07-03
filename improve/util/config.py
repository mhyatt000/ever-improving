
from dataclasses import dataclass, field

def default(data):
    return field(default_factory=lambda: data)
