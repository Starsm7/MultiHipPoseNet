import csv
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
# Query all fonts in the current system
# from matplotlib.font_manager import FontManager
# mpl_fonts = set(f.name for f in FontManager().ttflist)
# print('all font list get from matplotlib.font_manager:')
# for f in sorted(mpl_fonts):
#     print('\t' + f)
import matplotlib
from medpy.metric import hd, hd95
# plt.rcParams['font.sans-serif'] = ['AR PL UMing CN']
# Setting the negative sign to display the correct shape
matplotlib.rcParams['axes.unicode_minus'] = False


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)
    #--------------------------------------------#
    #   Calculation of dice coefficients
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# Let the label width W and length H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    # a is the label transformed into a one-dimensional array, shape (H×W,); b is the prediction result transformed into a one-dimensional array, shape (H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    # np.bincount counts the number of times each of the n**2 numbers from 0 to n**2-1 occurs, return value shape(n, n)
    # In the return, write diagonally for the correctly categorized pixel point
    # --------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def per_class_Dice(pred, gt, smooth=1e-5):
    N = gt.shape[0]
    Dices = np.zeros(N)
    for i in range(N):
        pred_flat = pred[i].reshape(1, -1)
        gt_flat = gt[i].reshape(1, -1)
        intersection = (pred_flat * gt_flat).sum(1)
        unionset = pred_flat.sum(1) + gt_flat.sum(1)
        Dices[i]= (2 * intersection + smooth) / (unionset + smooth)
    return Dices

def per_class_HD(pred, gt):
    N = gt.shape[0]
    HD = np.zeros(N)
    HD95 = np.zeros(N)
    for i in range(N):
        if np.max(pred[i])==1:
            HD[i] = hd(pred[i], gt[i])
            HD95[i] = hd95(pred[i], gt[i])
        else:
            HD[i],HD95[i]=1000,1000
    return HD,HD95
def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 
def to_one_hot(mask, num_classes=8):
    one_hot_mask = np.zeros((num_classes, *mask.shape), dtype=np.uint8)
    for i in range(num_classes):
        one_hot_mask[i][mask == i] = 1
    return one_hot_mask
def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  
    print('Num classes', num_classes)  
    #-----------------------------------------#
    # Create a matrix of all zeros that is an obfuscation matrix
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    DICE = []
    HD = []
    #------------------------------------------------#
    # Get a list of validation set label paths for direct reading
    # Get a list of validation set image segmentation result paths for direct reading
    #------------------------------------------------#
    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]  

    #------------------------------------------------#
    # Read each (image-tag) pair
    #------------------------------------------------#
    for ind in range(len(gt_imgs)):
        #------------------------------------------------#
        # Read an image segmentation result into a numpy array
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        # Read a sheet of corresponding labels, transform into numpy array
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))  

        # If the image segmentation result is not the same size as the label, this image is not computed
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue
        pred_hot = to_one_hot(pred, num_classes)
        label_hot = to_one_hot(label, num_classes)

        dice = per_class_Dice(pred_hot, label_hot)
        hd, hd95 = per_class_HD(pred_hot, label_hot)
        DICE.append(dice)
        ################################################################################
        HD.append(hd95)
        #------------------------------------------------#
        # Calculate a 21×21 hist matrix for a single image and accumulate it
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  
        # Output the mIoU value averaged over all categories in the currently computed images every 10 computations
        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )
    #------------------------------------------------#
    # Calculate category-by-category mIoU values for all validation set images
    #------------------------------------------------#
    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)

    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)))


    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))  
    return np.array(hist, np), IoUs, PA_Recall, Precision,np.mean(np.array(DICE), axis=0),np.mean(np.array(HD), axis=0)

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = False):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
            