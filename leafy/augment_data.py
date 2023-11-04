from .loader import ImageLoader

def augment_data(loader: ImageLoader):
    class_distribution = loader.get_better_class_distribution()
    print(f"{class_distribution = }")