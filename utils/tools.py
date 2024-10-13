import torch
from torchvision import transforms
from datasets.inference import resize_image
from nets.modules.config import get_parser
import cv2
import matplotlib
import os
import sys
import math
file_path=os.getcwd()
sys.path.append(file_path)
import numpy as np
from sklearn.linear_model import LinearRegression
matplotlib.use('Agg')
lin_reg = LinearRegression()

class ab:
    def __init__(self, data, num):
        self.data = data.groupby('Point')
        self.num = num
        self.result_dict = {}
        for group_name, group_data in self.data:
            x_values = group_data['X'].tolist()
            y_values = group_data['Y'].tolist()
            self.result_dict[group_name] = list(zip(x_values, y_values))
    def angle(self):
        angle_alpha = []
        angle_beta = []
        for i in range(len(self.num)):
            label = f'img{i+1}'
            x1 = list(self.result_dict[label][0])
            x2 = list(self.result_dict[label][1])
            p1 = np.array([x1[0], x1[1]])
            p2 = np.array([x2[0], x2[1]])
            x1 = list(self.result_dict[label][3])
            p4 = np.array([x1[0], x1[1]])
            x1 = list(self.result_dict[label][2])
            x2 = list(self.result_dict[label][4])
            p5 = np.array([x2[0], x2[1]])
            p3 = np.array([x1[0], x1[1]])
            x3 = list(self.result_dict[label][5])
            p6 = np.array([x3[0],x3[1]])
            X = np.array([p1[0], p2[0]]).reshape(-1, 1)
            Y = np.array([p1[1], p2[1]]).reshape(-1, 1)
            lin_reg.fit(X, Y)
            jixian_k = lin_reg.coef_[0][0]
            jixian_b = lin_reg.intercept_[0]
            # beta
            X = np.array([p4[0], p6[0]]).reshape(-1, 1)
            Y = np.array([p4[1], p6[1]]).reshape(-1, 1)
            lin_reg.fit(X, Y)
            beta_k = lin_reg.coef_[0][0]
            beta_b = lin_reg.intercept_[0]
            # alpha
            X = np.array([p5[0], p3[0]]).reshape(-1, 1)
            Y = np.array([p5[1], p3[1]]).reshape(-1, 1)
            lin_reg.fit(X, Y)
            alpha_k = lin_reg.coef_[0][0]
            alpha_b = lin_reg.intercept_[0]

            alpha = calculate_angle(alpha_k,alpha_b,jixian_k,jixian_b)
            beta = calculate_angle(beta_k,beta_b,jixian_k,jixian_b)
            angle_alpha.append(alpha)
            angle_beta.append(beta)
        return angle_alpha, angle_beta

def calculate_angle(k1, b1, k2, b2, default_angle=0.0):
    try:
        # Check that the denominator is zero
        if 1 + k1 * k2 == 0:
            raise ValueError("Slope calculation has a divide-by-zero error, unable to calculate the angle of pinch, returns to the default value of 0.0!")
        slope_difference = (k2 - k1) / (1 + k1 * k2)
        # Calculate the angle (radians)
        angle_rad = math.atan(abs(slope_difference))
        # Convert radians to angles
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    except ValueError as e:
        print(e)
        return default_angle

def cls_accuracy(X, Z):
    # 1 is positive no disease, 0 is negative patient
    i=0
    count = 0
    false_positive = 0
    false_negative = 0
    true_positive=0
    true_negative=0
    if len(X) != len(Z):
        raise ValueError("Input lists must have the same length.")
    fpw, fnw=[],[]
    for x, name in zip(X, Z):
        i += 1
        if name.endswith('+'):
            name=name[:-1]
        true_label = 0 if int(name) > 1000 else 1
        predicted_label = 0 if x < 60 else 1
        if predicted_label == true_label:
            count += 1
            if predicted_label == 1 and true_label == 1:
                true_positive+=1
            if predicted_label == 0 and true_label == 0:
                true_negative +=1
        else:
            if predicted_label == 1 and true_label == 0:
                false_positive += 1
                fpw.append(name)
            elif predicted_label == 0 and true_label == 1:
                false_negative += 1
                fnw.append(name)

    total_count = len(X)
    accuracy = count / total_count * 100
    false_positive_acc = false_positive / total_count * 100
    false_negative_acc = false_negative / total_count * 100
    presion = true_negative/(true_negative+false_negative+0.00001)* 100
    rescall = true_negative/(true_negative+false_positive+0.00001)* 100
    return accuracy,presion,rescall,false_positive_acc, false_negative_acc,fpw,fnw
def calculate_accuracy(X, Y):
    accuracy=[]
    if len(X) != len(Y):
        raise ValueError("Input lists must have the same length.")
    for i in np.arange(3,11,1):
        correct_count = sum(1 for x, y in zip(X, Y) if abs(x - y) <= i)
        total_count = len(X)
        accuracy.append(correct_count / total_count * 100)
    return accuracy
def read_images_from_folder_test(data_path,folder_path):
    images = []
    offsets=[]
    scare=[]
    for filename in folder_path:
        image = cv2.imread(data_path + filename + '.jpg')
        image, nw, nh, s = resize_image(image, (get_parser().input_h, get_parser().input_w))
        o = (get_parser().input_h - nh) // 2
        image = transforms.ToTensor()(image)
        image = torch.unsqueeze(image, dim=0)  # The DataLoader function is used for training, and the first dimension is added directly.
        image = image.to(get_parser().device)
        images.append(image)
        offsets.append(o)
        scare.append(s)
    return images,scare,offsets

def read_images_from_folder_visualization(data_path,folder_path):
    images = []
    for filename in folder_path:
        image = cv2.imread(data_path+'/VOC2007/JPEGImages/' + filename + '.jpg')
        images.append(image)
    return images

def show_point_on_picture(data_path, folder_path, pre, gro, test): 
    img = read_images_from_folder_visualization(data_path+'/VOCdevkit', folder_path)
    # color=((0, 0, 255),(255,240,245),(82,139,139),(0, 0, 128),(139,105,255))
    for i in range(test):
        label = f'img{i + 1}'
        for j, point in enumerate(pre[label]):
            point = tuple([int(point[0]), int(point[1])])
            image = cv2.drawMarker(img[i], position=point, color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        for point in gro[label]:
            point = tuple([int(point[0]), int(point[1])])
            image = cv2.drawMarker(img[i], position=point, color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        output_folder = data_path+'/miou_out/results/predict_pictures/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_image_path = os.path.join(output_folder, label+'_predict.jpg')
        cv2.imwrite(output_image_path, image)
        print(f"'{label}_predict.jpg' saved to {output_image_path}")