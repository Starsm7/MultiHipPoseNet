import json
import numpy as np
import cv2
import torch
from nets.modules.config import get_parser


def json_to_numpy(dataset_path):
    with open(dataset_path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        points = json_data['shapes']
    # print(points)
    landmarks = []
    for point in points:
        if point['label'] == '基线点': 
            for p in point['points']:
                landmarks.append(p)
    for point in points:
        if point['label'] == '骨缘点':
            for p in point['points']:
                landmarks.append(p)
    for point in points:
        if point['label'] == '髂骨下缘点':
            for p in point['points']:
                landmarks.append(p)
    for point in points:
        if point['label'] == '盂唇中心点':
            for p in point['points']:
                landmarks.append(p)
    for point in points:
        if point['label'] == '骨性髋臼顶凸面最高点':
            for p in point['points']:
                landmarks.append(p)
    for point in points:
        if point['label'] == '回声失落点':
            for p in point['points']:
                landmarks.append(p)

    assert len(landmarks) == 6
    landmarks = np.array(landmarks)
    landmarks = landmarks.reshape(-1, 2)
    points=[]
    for x in landmarks:
        points.append(tuple(x))
    return points

def generate_heatmaps(joints,scare,offset):
    target = np.zeros((get_parser().kpt_n,get_parser().input_h,get_parser().input_w),dtype=np.float32)
    for joint_id in range(get_parser().kpt_n):
        mu_x = joints[joint_id][0]*scare+offset
        mu_y = joints[joint_id][1]*scare+offset
        x = np.arange(0, get_parser().input_w, 1, np.float32)
        y = np.arange(0, get_parser().input_h, 1, np.float32)
        y = y[:, np.newaxis]
        target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * get_parser().gauss ** 2))
    return target*255
def get_max_preds(heatmaps):
    assert isinstance(heatmaps, np.ndarray), 'heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 3, 'heatmaps should be 3-ndim'
    num_joints = heatmaps.shape[0]
    width = heatmaps.shape[2]
    heatmaps_reshaped = heatmaps.reshape((num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 1)
    maxvals = np.amax(heatmaps_reshaped, 1)
    maxvals = maxvals.reshape((num_joints, 1))
    idx = idx.reshape((num_joints, 1))
    preds = np.tile(idx, (1, 2)).astype(np.float32)
    preds[:, 0] = (preds[:, 0]) % width
    preds[:, 1] = np.floor((preds[:, 1]) / width)
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask
    return preds, maxvals

def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx  = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1]+ hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = np.matrix([[dx],[dy]])
        hessian = np.matrix([[dxx,dxy],[dxy,dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord

def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2
    num_points=hm.shape[0]
    height = hm.shape[1]
    width = hm.shape[2]
    for i in range(num_points):
        origin_max = np.max(hm[i])
        dr = np.zeros((height + 2 * border, width + 2 * border))
        dr[border: -border, border: -border] = hm[i].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        hm[i] = dr[border: -border, border: -border].copy()
        hm[i] *= origin_max / np.max(hm[i])
    return hm

def heatmap_to_point(hm,scare,offset):
    coords, maxvals = get_max_preds(hm)
    hm = gaussian_blur(hm, 11)
    hm = np.maximum(hm, 1e-10)
    hm = np.log(hm)
    for p in range(coords.shape[0]):
        coords[p] = taylor(hm[p], coords[p])
    preds = coords.copy()
    return torch.tensor((preds-offset)/scare, dtype=torch.float32)

def resize_image(image, size):
    ih, iw, _ = image.shape
    h, w = size
    scale = min(w/iw, h/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.full((h, w, 3), 128, dtype=np.uint8)
    new_image[(h-nh)//2:(h-nh)//2+nh, (w-nw)//2:(w-nw)//2+nw, :] = resized_image
    return new_image, nw, nh,scale

if __name__ == '__main__':
    img = cv2.imread('../VOCdevkit/VOC2007/JPEGImages/1.jpg')
    new, w, h, scare = resize_image(img, (384,512))
    offset = (384 - h) // 2
    landmarks = json_to_numpy('../VOCdevkit/VOC2007/json/1.json')
    print('Key point coordinates', landmarks, '-------------', sep='\n')
    heatmaps = generate_heatmaps(landmarks,scare,offset)
    print(heatmaps.shape)
    landmarks = heatmap_to_point(heatmaps,scare,offset)
    print(landmarks)

