import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
current_font = plt.rcParams['font.family']
# plt.rcParams['font.sans-serif'] = ['sans-serif']
print("Current font family:", current_font)


# Calculate a set of assessment indicators for the model


# Calculate the average absolute error of alpha and beta from the true values
file_path=os.getcwd()
d=0.2646 
A_mse,B_mse=[],[]
A,B=[],[]
point_precisions = []
for i in range(5):
    df1 = pd.read_csv(file_path+f'/miou_out/results/angles/ab_{i+1}.csv')
    df2 = pd.read_csv(file_path+f'/miou_out/results/angles/ab_gro_{i+1}.csv')

    a_column1 = df1['a1'].values
    b_column1 = df1['b1'].values
    a_column2 = df2['a2'].values
    b_column2 = df2['b2'].values
    A.append(a_column1)
    B.append(b_column1)
    print(a_column1)
    A_mse.append(np.mean(abs(np.array(a_column2)-np.array(a_column1))))
    B_mse.append(np.mean(abs(np.array(b_column2)-np.array(b_column1))))
A=np.mean(np.array(A),axis=0).reshape(26,1)
B=np.mean(np.array(B),axis=0).reshape(26,1)
df = pd.DataFrame(np.concatenate((A,B),axis=1),columns=['Alpha_angle','Beta_angle'])
df.to_csv(file_path+f'/miou_out/results/angles/ab.csv',index=True)
print(df)
print('Mean Absolute Difference in Alpha:',np.mean(np.array(A_mse)))
print('Mean Absolute Difference in Beta:',np.mean(np.array(B_mse)))
data=np.array([[np.mean(np.array(A_mse))],[np.mean(np.array(B_mse))]])
df = pd.DataFrame(data,index=['Mean Absolute Difference in Alpha','Mean Absolute Difference in Beta'],columns=['Value'])
df.to_csv(file_path+f'/miou_out/results/angles/ab_mse.csv',index=True)
print(df)




# Calculation of key point assessment indicators
XX,YY,DD,EE,PP=[],[],[],[],[]
for i in range(5):
    df1 = pd.read_csv(file_path+f'/miou_out/results/predict_coordinate/point_predict_{i+1}.csv')
    df2 = pd.read_csv(file_path+f'/miou_out/results/predict_coordinate/grouth_{i+1}.csv')
    X,Y=[[],[],[],[],[],[]],[[],[],[],[],[],[]]
    X_sum,Y_sum=[[],[],[],[],[],[]],[[],[],[],[],[],[]]
    name = df1['Point'].values
    X2 = df2['X'].values
    X1 = df1['X'].values
    m=1
    for i,j in zip(range(6),range(6)):
        X1_dict={}
        for n,x1 in zip(name,X1):
            if i%6==0:
                    X1_dict[n]=x1
            i += 1
        X2_dict = {}
        for n,x2 in zip(name,X2):
            if (j)%6==0:
                X2_dict[n] = x2
            j+=1
        X_sum[m-1].append(d*abs(np.array(list(X1_dict.values()))-np.array(list(X2_dict.values()))))
        X[m-1].append(d*np.mean(abs(np.array(list(X1_dict.values()))-np.array(list(X2_dict.values())))))
        print(f'The average absolute difference in the key point {m}X:',d*np.mean(abs(np.array(list(X1_dict.values()))-np.array(list(X2_dict.values())))))
        m+=1

    Y2 = df2['Y'].values
    Y1 = df1['Y'].values
    m=1
    for i,j in zip(range(6),range(6)):
        y1_dict={}
        for n,y1 in zip(name,Y1):
            if i%6==0:
                    y1_dict[n]=y1
            i += 1
        y2_dict = {}
        for n,y2 in zip(name,Y2):
            if (j)%6==0:
                y2_dict[n] = y2
            j+=1
        Y_sum[m-1].append(d*abs(np.array(list(y1_dict.values()))-np.array(list(y2_dict.values()))))
        Y[m-1].append(d*np.mean(abs(np.array(list(y1_dict.values()))-np.array(list(y2_dict.values())))))
        print(f'The average absolute difference in the key point {m}Y:',d*np.mean(abs(np.array(list(y1_dict.values()))-np.array(list(y2_dict.values())))))
        m+=1
    D=[]
    for i in range(6):
        for x,y in zip(X_sum[i],Y_sum[i]):
            D.append(np.mean(np.array(pow(x*x+y*y,0.5))))
            print(f'The average absolute difference of the key points {i+1}:',np.mean(np.array(pow(x*x+y*y,0.5))))
    # Calculate distance percentile
    E,P=[],[]
    for i in range(6):
        distance_errors = np.sqrt(np.square(X_sum[i]) + np.square(Y_sum[i]))
        distance_percentiles = np.percentile(distance_errors, [50, 75, 95])

        # calculation accuracy
        precision_05mm_1mm_15mm = [np.mean(np.less_equal(distance_errors, 0.5))*100,np.mean(np.less_equal(distance_errors, 1.0))*100,np.mean(np.less_equal(distance_errors, 1.5))*100]
        E.append(distance_percentiles)
        P.append(precision_05mm_1mm_15mm)

        # Print or use results as needed
        print(f"距离百分位（50th, 75th, 95th）{i+1}:", distance_percentiles)
        print(f"精度（0.5mm, 1mm, 1.5mm）{i+1}:", precision_05mm_1mm_15mm)

    XX.append(X)
    YY.append(Y)
    DD.append(D)
    EE.append(E)
    PP.append(P)
    precision_per_point=[]
    for i in range(6):
        # Calculate accuracy for each threshold (0.5 mm, 1 mm, 1.5 mm)
        distance_errors = np.sqrt(np.square(X_sum[i]) + np.square(Y_sum[i]))
        precision_per_threshold = []
        for threshold in np.linspace(0,4.0,1000):
            precision_per_threshold.append(np.mean(np.less_equal(distance_errors, threshold))*100)
        precision_per_point.append(precision_per_threshold)
    # Stores accuracy data for each point
    point_precisions.append(precision_per_point)
landmark_names=['BP','BRP','LLP','COP','POAP','ELP']
avg_point_precisions = np.transpose(np.mean(point_precisions, axis=0))
# Plotting SDR curves for all points in one graph
thresholds = np.linspace(0,4.0,1000)
plt.figure()
plt.rc('lines', linewidth=1.5)  # 所有线条的默认宽度
plt.rc('axes', linewidth=1.5)   # 坐标轴边框线宽
# 设置全局字体为 Cascadia Code 且加粗
# plt.rc('font', family='Cascadia Code', weight='bold')  # 全局字体为 Cascadia Code 且加粗
# plt.rc('axes', titleweight='bold', labelweight='bold')  # 坐标轴标题和标签加粗
plt.title("Successful detection rate of MultiHipPoseNet")
plt.xlabel("Distance (mm)")
plt.ylabel("SDR (%)")
for i in range(6):
    sdr_values = [precision[i] for precision in avg_point_precisions]
    if i!=0:
        plt.plot(thresholds, sdr_values, label=f"{landmark_names[i]}", linewidth = 1.5)
plt.legend(frameon=False,loc='lower right')
plt.grid(linestyle='-.', linewidth=1.5)
# plt.show()
plt.savefig(file_path+f'/miou_out/results/predict_coordinate/SDR_points.png',dpi=800)
X,Y,D,E,P=np.mean(np.array(XX),axis=0),np.mean(np.array(YY),axis=0),np.mean(np.array(DD),axis=0),np.mean(np.array(EE),axis=0),np.mean(np.array(PP),axis=0)
data=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
i=0
for j in range(6):
    data[0+i].append(X[j][0])
    data[0+i].append(E[j][0])
    data[0+i].append(E[j][1])
    data[0+i].append(E[j][2])
    data[0+i].append(P[j][0])
    data[0+i].append(P[j][1])
    data[0+i].append(P[j][2])
    data[1 + i].append(Y[j][0])
    data[1+ i].append(E[j][0])
    data[1 + i].append(E[j][1])
    data[1 + i].append(E[j][2])
    data[1 + i].append(P[j][0])
    data[1 + i].append(P[j][1])
    data[1 + i].append(P[j][2])
    data[2 + i].append(D[j])
    data[2 + i].append(E[j][0])
    data[2 + i].append(E[j][1])
    data[2 + i].append(E[j][2])
    data[2 + i].append(P[j][0])
    data[2 + i].append(P[j][1])
    data[2 + i].append(P[j][2])
    i+=3
data=np.array(data)
# Save results
df = pd.DataFrame(data,
                 index=[["Keypoint#1", "Keypoint#1", "Keypoint#1", "Keypoint#2", "Keypoint#2", "Keypoint#2", "Keypoint#3", "Keypoint#3", "Keypoint#3","Keypoint#4", "Keypoint#4", "Keypoint#4","Keypoint#5", "Keypoint#5", "Keypoint#5","Keypoint#6", "Keypoint#6", "Keypoint#6"], ["△X", "△Y", "△D", "△X", "△Y", "△D","△X", "△Y", "△D","△X", "△Y", "△D","△X", "△Y", "△D","△X", "△Y", "△D"]],
                columns=[["MAE", "Percentile of error distance (mm)", "Percentile of error distance (mm)", "Percentile of error distance (mm)",'Precision (%)','Precision (%)','Precision (%)'], ["", "50%", "75%", "95%",'0.5mm','1mm','1.5mm']])
df.to_csv(file_path+f'/miou_out/results/predict_coordinate/result_points.csv',index=True)
print(df)



