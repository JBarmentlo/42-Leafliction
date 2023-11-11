from fire import Fire

from .data.fix_jpg_extensions import fix_jpg_extensions
from .data.Distribution import distribution
from .data.Augmentinator import augmentation
def cli():
    Fire({
        "fix" : fix_jpg_extensions,
        "d"   : distribution,
        "a"   : augmentation,
    })

if __name__ == "__main__":
    cli()