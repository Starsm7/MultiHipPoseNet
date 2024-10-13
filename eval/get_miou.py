import os
import shutil
from PIL import Image
from tqdm import tqdm
from eval.Hipnet import Hipnet
from utils.utils_metrics import compute_mIoU, show_results

#---------------------------------------------------------------------------#
# miou_mode is used to specify what the file calculates at runtime
# miou_mode is 0 for the entire miou calculation process, including getting the prediction results and calculating the miou.
# miou_mode is 1 for just getting the prediction results.
# miou_mode is 2 for just calculating miou.
#---------------------------------------------------------------------------#

def get_mious(n,miou_mode=0, nosiy=None,file_path=None,model_path='logs/min_loss_epoch_weights0.pth'):

    num_classes     = 8
    name_classes    = ["_background_","股骨头", "大转子", "盂唇", "软骨性髋臼顶", "滑膜皱襞", "近端软骨膜","Y型软骨"]
    VOCdevkit_path  = file_path+'/VOCdevkit'
    image_ids       = open(os.path.join(VOCdevkit_path, f"VOC2007/ImageSets/Segmentation/test1.txt"),'r').read().splitlines()
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   =  file_path+'/miou_out'
    pred_miou_dir        = os.path.join(miou_out_path, 'miou_detection-results')
    pred_dir = os.path.join(miou_out_path, f'detection-results_{n}')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_miou_dir):
            os.makedirs(pred_miou_dir)
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        print("Load model.")
        hipnet = Hipnet(model_path=model_path)
        print("Load model done.")
        print("Get predict result.")
        for image_id in tqdm(image_ids):
            if nosiy[0]==True:
                image_path = os.path.join(file_path + f'/data/nosiy_images/nosiy_{nosiy[1]}/', image_id + ".jpg")
            else:
                image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image1       = hipnet.get_miou_png(image)
            image1.save(os.path.join(pred_miou_dir, image_id + ".png"))
            image2 = hipnet.detect_image(image, True,name_classes)
            image2.save(os.path.join(pred_dir, image_id + "_pre.png"))
        print("Get predict result done.")
    if miou_mode == 2:
        if not os.path.exists(pred_miou_dir):
            os.makedirs(pred_miou_dir)
        print("Load model.")
        hipnet = Hipnet(model_path=model_path)
        print("Load model done.")
        for image_id in tqdm(image_ids):
            if nosiy[0] == True:
                image_path = os.path.join(file_path + f'/data/nosiy_images/nosiy_{nosiy[1]}/', image_id + ".jpg")
            else:
                image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = hipnet.get_miou_png(image)
            image.save(os.path.join(pred_miou_dir, image_id + ".png"))
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision, dice,hd = compute_mIoU(gt_dir, pred_miou_dir, image_ids, num_classes,name_classes)  # Execute the function that calculates mIoU
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
        print('#######################################################################################################')
        print('dice:', dice)
        print('hd:', hd)
        shutil.rmtree(pred_miou_dir)
        return IoUs, dice,hd
    if miou_mode == 0:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision,dice,hd = compute_mIoU(gt_dir, pred_miou_dir, image_ids, num_classes, name_classes)  # Execute the function that calculates mIoU
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
        print('#######################################################################################################')
        print('dice:',dice)
        print('hd:', hd)