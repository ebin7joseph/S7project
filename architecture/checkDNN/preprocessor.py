from PIL import Image
import numpy as np
import os
import glob
import PIL
import re

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def normalizer(img):
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_nrm = img_ybr.convert('RGB')

    return img_nrm

def resizer(img, w,h):
    """
    Resize PIL image

    Resizes image to be square with sidelength size. Pads with black if needed.
    """
    # Resize
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = h
        n_x_new = int(h * n_x / n_y + 0.5)
    else:
        n_x_new = w
        n_y_new = int(w * n_y / n_x + 0.5)

    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

    # Pad the borders to create a square image
    img_pad = Image.new('RGB', (w, h), (128, 128, 128))
    ulc = ((w - n_x_new) // 2, (h - n_y_new) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad


width = 300
height = 700

input_path = 'noise'
output_path = 'prepped-noise'
image_paths = sorted(glob.glob(os.path.join(input_path, '*.jpg')), key=natural_key)
os.makedirs(output_path, exist_ok=True)
for index,path in enumerate(image_paths):
    img = Image.open(path)
    img_normalized = normalizer(img)
    img_resized = resizer(img_normalized,width,height)
    basename = os.path.basename(path)
    path_out = os.path.join(output_path, basename)
    img_resized.save(path_out)
