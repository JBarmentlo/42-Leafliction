from fire import Fire

from .data.fix_jpg_extensions import fix_jpg_extensions
from .data.Distribution import distribution
from .data.Augmentinator import augmentation
from .data.Transformation import transformation
from .train import train
from .predict import predict, evaluate_folder


def cli_func():
    Fire(
        {
            "fix": fix_jpg_extensions,
            "distribution": distribution,
            "augmentation": augmentation,
            "transformation": transformation,
            "train": train,
            "predict": predict,
            "evaluate_folder": evaluate_folder,
        }
    )


if __name__ == "__main__":
    cli_func()
