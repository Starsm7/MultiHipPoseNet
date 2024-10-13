import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from nets.modules.config import get_parser
from datasets.inference import json_to_numpy, generate_heatmaps
from datasets.utils import cvtColor, preprocess_input


class MDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(MDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]
        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
        jpg   = cvtColor(jpg)
        # Read Tags
        mask = json_to_numpy(os.path.join(os.path.join(self.dataset_path, "VOC2007/json"), name + ".json"))
        png = Image.fromarray(np.array(png))
        jpg, png, nw, nh, scale = self.get_random_data(jpg, png, self.input_shape, random=self.train)
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float32)), [2, 0, 1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        offset = (get_parser().input_h- nh) // 2
        # print(scale)
        heatmaps = generate_heatmaps(mask, scale, offset)
        return jpg, png, seg_labels,heatmaps
    def get_random_data(self, image, label, input_shape, random=False):
        h, w    = input_shape
        iw, ih = image.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        if not random:
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))
            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label,nw,nh,scale

def dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    heatmaps=[]
    for img, png, labels,heat in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
        heatmaps.append(heat)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    heatmaps = torch.from_numpy(np.array(heatmaps)).type(torch.FloatTensor)
    return images, pngs, seg_labels,heatmaps
