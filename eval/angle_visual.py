import os
import cv2
import pandas as pd


# Visualization of coordinate prediction results


file_path=os.getcwd()
name = pd.read_csv(file_path+f'/miou_out/results/angles/ab_1.csv')
df1 = pd.read_csv(file_path+f'/miou_out/results/predict_coordinate/point_predict_1.csv')
df2 = pd.read_csv(file_path+f'/miou_out/results/predict_coordinate/grouth_1.csv')
X,Y=[[],[],[],[],[]],[[],[],[],[],[]]
X_sum,Y_sum=[[],[],[],[],[]],[[],[],[],[],[]]
name = name['file_name'].values
X2 = df2['X'].values
X1 = df1['X'].values
Y2 = df2['Y'].values
Y1 = df1['Y'].values
dict1,dict2={},{}
i=0
for n in name:
    points=[]
    for j in range(5):
        points.append([X1[i],Y1[i]])
        i+=1
    dict1[str(n)]=points
i=0
for n in name:
    points=[]
    for j in range(5):
        points.append([X2[i],Y2[i]])
        i+=1
    dict2[str(n)]=points
with open(os.path.join(file_path, f"VOCdevkit/VOC2007/ImageSets/Segmentation/test_fold0.txt"), "r") as f:
    test_lines = f.readlines()
for name in test_lines:
    img_path=file_path+f"/VOCdevkit/VOC2007/JPEGImages/{name[:-1]}.jpg"
    img=cv2.imread(img_path)

    for j, point in enumerate(dict1[name[:-1]]):
        point = tuple([int(point[0]), int(point[1])])
        image = cv2.drawMarker(img, point, color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
    for j, point in enumerate(dict2[name[:-1]]):
        point = tuple([int(point[0]), int(point[1])])
        image = cv2.drawMarker(img, point, color=(0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
    output_folder =file_path+ '/miou_out/results/predict_pictures/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_image_path = os.path.join(output_folder, name[:-1] + '_predict.jpg')
    cv2.imwrite(output_image_path, image)
    print(f"'{name[:-1]}_predict.jpg' saved to {output_image_path}")
