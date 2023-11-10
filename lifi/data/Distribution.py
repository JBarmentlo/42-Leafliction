from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from fire import Fire
from random import shuffle
import matplotlib.colors as mcolors
import numpy as np

from .Loader import ImageDataset

color_list = list(mcolors.cnames.values())
shuffle(color_list)    

def better_plot_class_distribution(data_folder: str):
    data   = Path(data_folder)
    loader = ImageDataset(data)
        
    class_distribution = loader.get_better_class_distribution()
    return class_distribution

def show_class_distribution(class_distribution):
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

def distribution(data_folder):
    class_distribution = better_plot_class_distribution(data_folder)
    show_class_distribution(class_distribution)
    
