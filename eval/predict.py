import csv
from eval.get_miou import get_mious
from datasets.inference import heatmap_to_point,json_to_numpy
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from nets.modules.config import get_parser
lin_reg = LinearRegression()
from utils.tools import *


def predict(file_path,ksplit,nosiy,gious,weight,miou_mode=2):
    pre_a = []
    pre_b = []
    MAE_A, MAE_B = [], []
    D=[]
    Cls_acc,Presion,Rescall, FP, FN =[],[],[],[],[]
    IoUs=[]
    Dices=[]
    HDS=[]
    for n in range(ksplit):
        model = torch.load(file_path + '/logs/' + weight[n], weights_only=False)
        model.eval()
        folder_path = []
        with open(os.path.join(file_path, f"VOCdevkit/VOC2007/ImageSets/Segmentation/test2.txt"),"r") as f:
            test_lines = f.readlines()
        dict = {}
        i = 1
        NAME = []
        for name in test_lines:
            folder_path.append(name[:-1])
            point = json_to_numpy(file_path+'/VOCdevkit/VOC2007/json/'+ name[:-1] + '.json')
            img = f'img{i}'
            dict[img] = point
            NAME.append(name[:-1])
            i = i + 1
        test = i - 1
        output_file = file_path+f'/miou_out/results/predict_coordinate/grouth_{n+1}.csv'
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Point', 'X', 'Y'])
            for m, tensor in dict.items():
                for j, (x, y) in enumerate(tensor):
                    csv_writer.writerow([m, x, y])
        if nosiy[0]==True:
            images, scare, offset = read_images_from_folder_test(file_path + f'/data/nosiy_images/nosiy_{nosiy[1]}/', folder_path)
        else:
            images,scare,offset = read_images_from_folder_test(file_path+'/VOCdevkit/VOC2007/JPEGImages/',folder_path)
        keypoints=[]
        ################################################################################################################
        with torch.no_grad():
            for i in range(test):
                _,results = model(images[i])
                results = results[0].cpu().detach().numpy()
                results = heatmap_to_point(results,scare[i],offset[i])
                results = results.reshape(1,get_parser().kpt_n,2).numpy()
                keypoints.append(results)
        output_file =file_path+f'/miou_out/results/predict_coordinate/point_predict_{n+1}.csv'
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Point', 'X', 'Y'])
            for m, tensor in enumerate(keypoints):
                points = tensor[0]
                for j, (x, y) in enumerate(points):
                    csv_writer.writerow([f'img{m + 1}', x, y])
        print(f'Data has been saved to {output_file}')
        # Read data data_pre
        data_pre = pd.read_csv(file_path+f'/miou_out/results/predict_coordinate/point_predict_{n+1}.csv', dtype={'Point': str})
        data_gro = pd.read_csv(file_path+f'/miou_out/results/predict_coordinate/grouth_{n+1}.csv')
        X1 = data_pre['X'].values
        Y1 = data_pre['Y'].values
        X2 = data_gro['X'].values
        Y2 = data_gro['Y'].values
        print("mean_absolute_error_X:", mean_absolute_error(X1, X2))
        print("mean_absolute_error_Y:", mean_absolute_error(Y1, Y2))
        D.append(pow(mean_absolute_error(X1, X2)*mean_absolute_error(X1, X2)+mean_absolute_error(Y1, Y2)*mean_absolute_error(Y1, Y2),0.5))
        # Calculate the angle, visualize it and save the file.
        ################################################################################################################
        a=ab(data_pre,NAME)
        a1,b1=a.angle()
        b=ab(data_gro,NAME)
        a2,b2=b.angle()
        if get_parser().show_point_on_picture==True:
            show_point_on_picture(file_path,folder_path,a.result_dict,b.result_dict,test)
        result_df = pd.DataFrame({'file_name':folder_path,'a1': a1, 'b1': b1})
        result_df.to_csv(file_path+f'/miou_out/results/angles/ab_{n + 1}.csv',index=False)
        result_df = pd.DataFrame({'file_name':folder_path,'a2': a2, 'b2': b2})
        result_df.to_csv(file_path+f'/miou_out/results/angles/ab_gro_{n + 1}.csv',index=False)
        mae_a = np.mean(abs(np.array(a2) - np.array(a1)))
        mae_b = np.mean(abs(np.array(b2) - np.array(b1)))
        print("mean_squared_error_alpha:", mae_a)
        print("mean_squared_error_beta:", mae_b)
        MAE_A.append(mae_a)
        MAE_B.append(mae_b)
        # Preservation of single prediction accuracy
        result_accuracy_a = calculate_accuracy(a1,a2)
        result_accuracy_b = calculate_accuracy(b1, b2)
        print(result_accuracy_a)
        print(result_accuracy_b)
        pre_a.append(result_accuracy_a)
        pre_b.append(result_accuracy_b)
        cls_acc,presion,rescall, false_positive_acc, false_negative_acc,fpw,fnw = cls_accuracy(a1, NAME)
        Cls_acc.append(cls_acc)
        Presion.append(presion)
        Rescall.append(rescall)
        FP.append(false_positive_acc)
        FN.append(false_negative_acc)
        print(f"Accuracy_{n + 1} within 5 degrees of the Alpha angle: {result_accuracy_a[2]:.3f}%")
        print(f"Accuracy_{n + 1} within 5 degrees of Beta angle: {result_accuracy_b[2]:.3f}%")
        print('FP, negative predicts positive, patient predicts not sick; FN, positive predicts negative, not sick predicts patient')
        print(f'Accuracy_{n + 1} for disease diagnosis results, Presion, Rescall, FP_{n + 1}, and FN_{n + 1} for negative cases, respectively: {cls_acc:.3f}%,{presion:.3f}%,{rescall:.3f}%,{false_positive_acc:.3f}%,{false_negative_acc:.3f}%')
        print(f'The error picture for FP_{n + 1} is: {fpw}')
        print(f'The error picture for FN_{n + 1} is: {fnw}')
        if gious==True:
            if miou_mode!=1:
                iou, dice, hd = get_mious(n=n, miou_mode=miou_mode, nosiy=nosiy, file_path=file_path,
                                          model_path=file_path + f'/logs/{weight[n]}')
                IoUs.append(iou)
                Dices.append(dice)
                HDS.append(hd)
            else:
                get_mious(n=n, miou_mode=miou_mode, nosiy=nosiy, file_path=file_path,
                          model_path=file_path + f'/logs/{weight[n]}')
    x = list(np.arange(3,11,1))
    alpha = []
    beta = []
    for i in range(8):
        p = []
        q = []
        for j in range(ksplit):
            p.append(pre_a[j][i])
            q.append(pre_b[j][i])
        alpha.append(np.mean(p))
        beta.append(np.mean(q))
    # current_font = plt.rcParams['font.family']
    # plt.rcParams['font.sans-serif'] = ['sans-serif']
    # print("Current font family:", current_font)
    plt.figure()
    plt.plot(x, alpha, marker='p', linestyle='-', color='#C82423', label='alpha')
    plt.plot(x, beta, marker='p', linestyle='-', color='#14517C', label='beta')
    plt.xlabel('Absolute error of angle(°)')
    plt.ylabel('Successful percent')
    plt.ylim(70, 105)
    plt.xticks(x)
    plt.yticks(range(70, 105, 10))
    plt.grid(linestyle='-.')
    plt.legend(frameon=False,loc='lower right',fontsize=11)
    plt.savefig(file_path+f'/miou_out/results/angles/预测角度误差累积分布.png',dpi=800)
    print('################################################################################################################')
    print('################################################################################################################')
    print('################################################################################################################')
    print('################################################################################################################')
    print('################################################################################################################')
    print(f"Final accuracy of alpha: {np.mean(alpha[2]):.3f}%")
    print(f"Final accuracy of beta: {np.mean(beta[2]):.3f}%")
    print(f"The final D,mean_squared_error_alpha,mean_squared_error_beta are: {np.mean(D):.3f},{np.mean(MAE_A):.3f},{np.mean(MAE_B):.3f}")
    print(f'The final Accuracy, Presion, Rescall, FP, and FN of negative cases for the diagnostic results of the disease condition were respectively:{np.mean(Cls_acc):.3f}%,{np.mean(Presion):.3f}%,{np.mean(Rescall):.3f}%,{np.mean(FP):.3f}%,{np.mean(FN):.3f}%')
    if gious == True:
        if miou_mode != 1:
            print(f'IoUs:{np.mean(IoUs):.3f}')
            print(f'Dices:{np.mean(np.array(Dices),axis=0)}')
            print(f'HDs:{np.mean(np.array(HDS), axis=0)}')
            data = {
                'Femoral Head Dices': np.array(Dices).T[1],
                'Large Rotor Dices': np.array(Dices).T[2],
                'Glenoid Labrum Dices': np.array(Dices).T[3],
                'Cartilage Acetabular Roof Dices': np.array(Dices).T[4],
                'Synovial Fold Dices': np.array(Dices).T[5],
                'Perichondrium Dices': np.array(Dices).T[6],
                'Y Type Cartilage Dices': np.array(Dices).T[7],
                'Femoral Head HDS': np.array(HDS).T[1],
                'Large Rotor HDS': np.array(HDS).T[2],
                'Glenoid Labrum HDS': np.array(HDS).T[3],
                'Cartilage Acetabular Roof HDS': np.array(HDS).T[4],
                'Synovial Fold HDS': np.array(HDS).T[5],
                'Perichondrium HDS': np.array(HDS).T[6],
                'Y Type Cartilage HDS': np.array(HDS).T[7]
            }
            df = pd.DataFrame(data)
            df.to_csv(file_path+f'/miou_out/results/mask_output.csv', index=False)

