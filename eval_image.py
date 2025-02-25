#! /usr/bin/python3

from common_header import *
from PIL import Image
import argparse
from torchvision import transforms
import torch as tch
from allocate_cuda_device import allocate_cuda
import errno

mean = [.485, .456, .406]
std  = [.229, .224, .225]
#mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
#std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

parser = argparse.ArgumentParser(description='Evaluate trained models.')
parser.add_argument('-M', '--models', nargs='+', default=[])
parser.add_argument('-i', '--images', nargs='+', default=[])
parser.add_argument('-v', '--verbose', action='store_true')
parser.set_defaults(verbose=False)
args = parser.parse_args()

models = args.models
images = args.images
verbose = args.verbose

device = allocate_cuda()

device = allocate_cuda()
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# create a color pallette, selecting a color for each class
palette = tch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = tch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

og_dir = os.getcwd()
for model in models:
    net_id = model.split('/')[-1][:-4]
    if verbose:
        print(f'Started evaluation of {net_id}.')

    save_at = os.path.join(og_dir, f'{net_id}_images')

    try:
        os.makedirs(save_at)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    net = tch.load(model, map_location=tch.device('cpu'))
    n = net.n_branches
    net.to(device)
    net.eval()
    for img in images:
        if verbose:
            print(f'\tImage: {img}')
        input_image = Image.open(os.path.join(og_dir,img))
        input_image = input_image.convert("RGB")

        input_tensor = preprocess(input_image).to(device)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        with tch.no_grad():
            output_tensor = net(input_batch).squeeze().cpu()

        n = output_tensor.shape[0]
        output_predictions = output_tensor.argmax(dim=1)
        img_name = img.split('/')[-1].split('.')[0]
        for i in range(n):
            dest_save = os.path.join(save_at, f'{img_name}_b{i+1}.png')
            r = Image.fromarray(output_predictions[i].byte().numpy()).resize(input_image.size)
            r.putpalette(colors)
            r.save(dest_save)

    if verbose:
        print(f'Finished {net_id} evalutation. Resulting images can be found @ {save_at}.')
