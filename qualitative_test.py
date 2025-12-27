import argparse
import os
from PIL import Image, ImageDraw

import torch
from torchvision import transforms

import models
from utils import make_coord
from test_noscale import batched_predict as batched_predict_noscale
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', default='qualitative_results', help='Directory to save results')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--no_scale', type=bool, default=False)
    parser.add_argument('--patch_pos', type=str, default='center', help='Patch position "center" or "x,y" (top-left)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. Load Image (Use Original Resolution)
    # The requirement is "Use original resolution HR as input".
    img_hr_orig = Image.open(args.input).convert('RGB')
    img_lr = img_hr_orig.resize((216, 144), Image.BICUBIC)
    
    # 2. Define Patch
    W, H = img_lr.size
    patch_size = 48
    
    if args.patch_pos == 'center':
        left = (W - patch_size) // 2
        top = (H - patch_size) // 2
    else:
        left, top = map(int, args.patch_pos.split(','))
        
    # Ensure patch is within bounds
    left = max(0, min(W - patch_size, left))
    top = max(0, min(H - patch_size, top))
    
    print(f"Selecting patch at (left={left}, top={top}) with size {patch_size}x{patch_size}")

    # 3. Save Image 1: 48x48 with Red Box
    img_box = img_lr.copy()
    draw = ImageDraw.Draw(img_box)
    # PIL rectangle: [x0, y0, x1, y1] where x1, y1 are inclusive in typical usage for outline
    # We want to outline the pixels from (left, top) to (left+patch_size-1, top+patch_size-1)
    draw.rectangle([left, top, left + patch_size - 1, top + patch_size - 1], outline='red', width=4)
    img_box.save(os.path.join(args.output_dir, '1_input_box.png'))

    # 4. Save Image 2: The 12x12 Patch
    patch = img_lr.crop((left, top, left + patch_size, top + patch_size))
    patch.save(os.path.join(args.output_dir, '2_patch.png'))

    # 5. Save Image 3: Bicubic Upscale 30x
    upscale_factor = 30
    target_size = patch_size * upscale_factor # 360
    patch_bicubic = patch.resize((target_size, target_size), Image.BICUBIC)
    patch_bicubic.save(os.path.join(args.output_dir, '3_bicubic.png'))

    # 6. Save Image 4: Model Upscale 30x
    print(f"Loading model from {args.model}...")
    ckpt = torch.load(args.model)
    model = models.make(ckpt['model'], load_sd=True).cuda()
    
    # Prepare input tensor (Full 48x48 image)
    img_tensor = transforms.ToTensor()(img_lr).cuda()
    inp = (img_tensor - 0.5) / 0.5
    inp = inp.unsqueeze(0) # Batch size 1
    
    # Prepare coordinates for the patch
    # Map pixel indices to continuous coordinates [-1, 1]
    # Pixel centers: -1 + (i + 0.5) * (2/N)
    # Pixel boundaries: -1 + i * (2/N)
    
    r_w = 2.0 / W
    x_min = -1 + left * r_w
    x_max = -1 + (left + patch_size) * r_w
    
    r_h = 2.0 / H
    y_min = -1 + top * r_h
    y_max = -1 + (top + patch_size) * r_h
    
    # Create query coordinates
    coord = make_coord((target_size, target_size), ranges=((y_min, y_max), (x_min, x_max))).cuda()
    coord = coord.unsqueeze(0) # (1, 360*360, 2)
    
    # Cell size
    pixel_size_y = (y_max - y_min) / target_size
    pixel_size_x = (x_max - x_min) / target_size
    
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= pixel_size_y
    cell[:, :, 1] *= pixel_size_x
    
    print("Running inference...")
    # Batched predict
    if args.no_scale:
        pred = batched_predict_noscale(model, inp, coord, cell, bsize=30000)[0]
    else:
        pred = batched_predict(model, inp, coord, cell, bsize=30000, scale=upscale_factor, scale2=upscale_factor)[0]
    
    # Post-process
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(target_size, target_size, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(os.path.join(args.output_dir, '4_liif.png'))
    
    print(f"Results saved to {args.output_dir}")
