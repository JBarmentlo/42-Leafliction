from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch import Tensor


def image_grid(imgs, rows, cols) -> Image.Image:
    assert len(imgs) == rows * cols

    images = []
    if isinstance(imgs[0], Tensor):
        for i, im in enumerate(imgs):
            images.append(to_pil_image(im))
    elif isinstance(imgs[0], Image.Image):
        images = imgs

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
