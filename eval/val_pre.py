import csv
from datasets.inference import heatmap_to_point, json_to_numpy
import matplotlib
import pandas as pd
from sklearn.linear_model import LinearRegression
matplotlib.use('Agg')
lin_reg = LinearRegression()
from utils.tools import *


def predict_val(file_path,data_path,model,n):
    folder_path = []
    # Test Path
    with open(os.path.join(data_path, f"VOC2007/ImageSets/Segmentation/val_fold{n}.txt"), "r") as f:
        test_lines = f.readlines()
    dict = {}
    i = 1
    NAME=[]
    for name in test_lines :
        folder_path.append(name[:-1])
        point = json_to_numpy(data_path+'/VOC2007/json/' + name[:-1] + '.json')
        img=f'img{i}'
        dict[img] = point
        NAME.append(name[:-1])
        i = i + 1
    val_test =i-1
    output_file = file_path+f'/results/val/grouth_{n+1}.csv'
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Point', 'X', 'Y'])
        for m, tensor in dict.items():
            for j, (x, y) in enumerate(tensor):
                csv_writer.writerow([m, x, y])
    images,scare,offsets = read_images_from_folder_test(data_path+'/VOC2007/JPEGImages/',folder_path)
    keypoints=[]
    ################################################################################################################
    with torch.no_grad():
        for i in range(val_test):
            _,results = model(images[i])
            results = results[0].cpu().detach().numpy()
            results = heatmap_to_point(results,scare[i],offsets[i])
            results = results.reshape(1,get_parser().kpt_n,2).numpy()
            keypoints.append(results)
    output_file = file_path+f'/results/val/point_predict_{n+1}.csv'
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Point', 'X', 'Y'])
        for m, tensor in enumerate(keypoints):
            points = tensor[0]
            for j, (x, y) in enumerate(points):
                csv_writer.writerow([f'img{m + 1}', x, y])
    # Read data data_pre
    data_pre = pd.read_csv(file_path+f'/results/val/point_predict_{n+1}.csv', dtype={'Point': str})
    data_gro = pd.read_csv(file_path+f'/results/val/grouth_{n+1}.csv')
    ################################################################################################################
    a=ab(data_pre,NAME)
    a1,b1=a.angle()
    b=ab(data_gro,NAME)
    a2,b2=b.angle()
    result_accuracy_a = calculate_accuracy(a1,a2)
    result_accuracy_b = calculate_accuracy(b1, b2)
    cls_acc, presion, rescall, false_positive_acc, false_negative_acc, _, _ = cls_accuracy(a1, NAME)
    print(f"Accuracy_{n + 1} within 5 degrees of the Alpha angle: {result_accuracy_a[2]:.3f}%")
    print(f"Accuracy_{n + 1} within 5 degrees of the Beta angle: {result_accuracy_b[2]:.3f}%")
    print('FP, negative predicts positive, patient predicts not sick; FN, positive predicts negative, not sick predicts patient')
    print(
        f'Accuracy_{n + 1} for disease diagnosis results, Presion, Rescall, FP_{n + 1}, and FN_{n + 1} for negative cases, respectively:{cls_acc:.3f}%,{presion:.3f}%,{rescall:.3f}%,{false_positive_acc:.3f}%,{false_negative_acc:.3f}%')
    return result_accuracy_a[2], result_accuracy_b[2]
