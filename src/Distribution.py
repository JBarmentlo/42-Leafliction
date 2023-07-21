from pathlib import Path

import matplotlib.pyplot as plt
from fire import Fire
from random import shuffle
import matplotlib.colors as mcolors
import numpy as np

from .loader import ImageLoader

color_list = list(mcolors.cnames.values())
shuffle(color_list)    

def plot_class_distribution(data_folder: Path):
    data_folder = Path(data_folder)
    loader      = ImageLoader(data_folder)
    
    class_distribution = loader.get_class_distribution()
    names              = list(class_distribution.keys())
    values             = list(class_distribution.values())

    fig = plt.figure(figsize = (18, 8))
    ax  = fig.add_subplot(111)
    ax.set_title(f"Class Distribution: {data_folder}")    
    ax.bar(range(len(class_distribution)), values, tick_label=names, color=list(color_list)[:len(class_distribution)])
    plt.show()

def better_plot_class_distribution(data_folder: Path):
    data_folder = Path(data_folder)
    loader      = ImageLoader(data_folder)
        
    class_distribution = loader.get_better_class_distribution()
    fig = plt.figure(figsize = (18, 8 * len(class_distribution)))
    axes = fig.subplots(len(class_distribution), 2)

    if not isinstance(axes[0], np.ndarray):
        axes = np.array([axes])

    color_offset = 0
    for ax, fruit in zip(axes, class_distribution.keys()):
        fruit_distribution = class_distribution[fruit]
        names              = list(fruit_distribution.keys())
        values             = list(fruit_distribution.values())
        pieax = ax[0]
        pieax.set_title(f"Class Distribution: {fruit}")    
        pieax.pie(values, labels=names, colors=list(color_list)[color_offset:color_offset+len(names)], autopct='%1.1f%%')
        
        barax = ax[1]
        barax.set_title(f"Class Distribution: {fruit}")    
        barax.bar(range(len(names)), values, tick_label=names, color=list(color_list)[color_offset:color_offset+len(names)])
        
        color_offset += len(names)

    plt.show()

plt.show()
if __name__ == "__main__":
    # Fire(plot_class_distribution)
    Fire(better_plot_class_distribution)
    
