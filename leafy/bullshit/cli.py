from fire import Fire

from .Augmentation import augmentation
from .Distribution import distribution
from .Transformation import transformation
from ..train import train
from ..predict import predict


def cli():
    Fire(
        {
            "augmentation": augmentation,
            "distribution": distribution,
            "transformation": transformation,
            "train": train,
            "predict": predict,
        }
    )
