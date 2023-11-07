from fire import Fire

from leafy.bullshit.Augmentation   import augmentation
from leafy.bullshit.Distribution   import distribution
from leafy.bullshit.Transformation import transformation
from leafy.train                   import train
from leafy.predict                 import predict

def cli():
    Fire({"augmentation"  : augmentation,
          "distribution"  : distribution,
          "transformation": transformation,
          "train"         : train,
          "predict"       : predict,
         }
        )

if __name__ == "__main__":
    cli()