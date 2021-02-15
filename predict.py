import argparse

import model_class as mc
import data_process as dp

from PIL import Image
import torch
from torchvision import transforms, datasets, models
import json

import matplotlib.pyplot as plt
import seaborn as sb

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('checkpoint', type=str)
parser.add_argument('--top_k', dest = 'top_k', type=int, default=5)

args = parser.parse_args()

# read json labels
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# load the model
model = mc.load_checkpoint(args.checkpoint)

# define inputs
img_dir = args.input
cat = img_dir.split('/')[-2]
dp.imshow(dp.process_image(img_dir))
plt.title(cat_to_name[cat])
plt.axis('off')

# predict
probs, cats = mc.predict(img_dir, model, topk=args.top_k)
print('prob: {}'.format(probs))
print('labels: {}'.format(cats))

classes = [cat_to_name[cat] for cat in cats]

img = Image.open(img_dir).convert('RGB')
img_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
img = img_transform(img)

plt.figure(figsize = [4, 8])
plt.subplot(2, 1, 1)
plt.imshow(img)
plt.title(cat_to_name[cat])
plt.axis('off')

plt.subplot(2, 1, 2)
sb.barplot(x = probs, y = classes, color = 'pink')
plt.xlabel('')
plt.ylabel('')
#plt.show()
