from fire import Fire

from .Augmentation   import augmentation
from .Distribution   import distribution
from .Transformation import transformation

def cli():
    Fire({"augmentation": augmentation,
          "distribution": distribution,
          "transformation": transformation
         }
        )