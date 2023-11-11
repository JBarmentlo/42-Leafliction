from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch import Tensor

def image_grid(imgs, rows, cols) -> Image.Image:
    assert len(imgs) == rows*cols

    images = []
    if isinstance(imgs[0], Tensor):
        for i, im in enumerate(imgs):
            images.append(to_pil_image(im))
    elif isinstance(imgs[0], Image.Image):
        images = imgs
        
    
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(images):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def double_im_with_text(im1, im2, pred) -> Image.Image:
    w, h = im1.size
    margin = 10
    grid = Image.new('RGB', size=(2*w + margin, 1*h + 200))
    grid_w, grid_h = grid.size
    
    grid.paste(im1, box=(0, 0))
    grid.paste(im2, box=(w + margin, 0))
    
    draw = ImageDraw.Draw(grid)
    text = f"Predicted class: {pred}"
    fontsize = 20
    font = ImageFont.truetype("arial.ttf", fontsize)
    draw.text((10, h + 20), text, font=font)
    return grid
        