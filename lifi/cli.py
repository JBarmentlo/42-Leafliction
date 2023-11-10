from fire import Fire

from .data.fix_jpg_extensions import fix_jpg_extensions
from .data.Distribution import distribution

def cli():
    Fire({
        "fix" : fix_jpg_extensions,
        "d": distribution,
    })

if __name__ == "__main__":
    cli()