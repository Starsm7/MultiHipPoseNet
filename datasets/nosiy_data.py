import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import warnings
from utils import seed_everything
warnings.filterwarnings('ignore')
import cv2
import random


# Generate test set images for different levels of noise


def pepper_and_salt(img,percentage):
    num=int(percentage*img.shape[0]*img.shape[1])
    random.randint(0, img.shape[0])
    img2=img.copy()
    for i in range(num):
        X=random.randint(0,img2.shape[0]-1)
        Y=random.randint(0,img2.shape[1]-1)
        if random.randint(0,1) ==0: 
            img2[X,Y] = (255,255,255)
        else:
            img2[X,Y] =(0,0,0)
    return img2

def calculate_psnr_ssim(clear_img_path, hazy_img_path, clear_img_names, hazy_img_names):
    """
    This function is used to calculate the PSNR and SSIM values of the image before and after defogging.
    :param clear_img_path: clear image folder path
    :param hazy_img_path: folder path of the image to be defogged
    :param clear_img_names: list of clear image file names
    :param hazy_img_names: list of image filenames to be defogged
    :return: None
    """
    SSIM_list = []
    PSNR_list = []
    for i in range(len(clear_img_names)):
        clear_img = cv2.imread(os.path.join(clear_img_path, clear_img_names[i]))
        hazy_img = cv2.imread(os.path.join(hazy_img_path, hazy_img_names[i]))
        if clear_img.shape[0] != hazy_img.shape[0] or clear_img.shape[1] != hazy_img.shape[1]:
            pil_img = Image.fromarray(hazy_img)
            pil_img = pil_img.resize((clear_img.shape[1], clear_img.shape[0]))  # Keep width and height consistent with clear_img
            hazy_img = np.array(pil_img)
        # Calculating PSNR
        PSNR = peak_signal_noise_ratio(clear_img, hazy_img)
        print(i + 1, 'PSNR: ', PSNR)
        PSNR_list.append(PSNR)
        # Calculating SSIM
        SSIM = structural_similarity(clear_img, hazy_img, channel_axis=2,multichannel=True)
        print(i + 1, 'SSIM: ', SSIM)
        SSIM_list.append(SSIM)
    print("average SSIM", sum(SSIM_list) / len(SSIM_list))
    print("average PSNR", sum(PSNR_list) / len(PSNR_list))
# test function
# calculate_psnr_ssim(clear_img_path, hazy_img_path, clear_img_names, hazy_img_names)

if __name__ == "__main__":
    seed_everything(967)
    with open(os.path.join(f'../VOCdevkit/VOC2007/ImageSets/Segmentation/test_fold0.txt'),"r") as f:
        lines = f.readlines()
    print(lines)
    test_names=[]
    for m in lines:
        m=m.rstrip('\n')
        image = cv2.imread(f'../data/JPEGImages/{m}.jpg')
        # Save image locally
        output_folder_img = f'../data/nosiy_images/test'
        if not os.path.exists(output_folder_img):
            os.makedirs(output_folder_img)
        output_image_path = os.path.join(output_folder_img, f'{m}.jpg')
        cv2.imwrite(output_image_path, image)
        print(f"'{m}.jpg' 已保存到{output_image_path}")
        test_names.append(m+'.jpg')
    # 0,1,63.043
    # 0.02,0.645,20.542
    # 0.01,0.799,23.531
    # 0.005,0.893,26.526
    # 0.001,0.977,33.503
    # 0.0001,0.998,43.533
    percent=[0.02,0.01,0.005,0.001,0.0001]
    for p in percent:
        for m in lines:
            m=m.rstrip('\n')
            image = cv2.imread(f'../data/nosiy_images/test/{m}.jpg')
            # Save image locally
            output_folder_img = f'../data/nosiy_images/nosiy_{p}/'
            if not os.path.exists(output_folder_img):
                os.makedirs(output_folder_img)
            reimg = pepper_and_salt(image,p)
            output_image_path = os.path.join(output_folder_img, f'{m}.jpg')
            cv2.imwrite(output_image_path, reimg)
            print(f"'{m}.jpg' 已保存到{output_image_path}")
        clear_img_path=f'../data/nosiy_images/test'
        hazy_img_path=f'../data/nosiy_images/nosiy_{p}/'
        clear_img_names=test_names
        hazy_img_names=test_names
        print(hazy_img_names)
        #calculate_psnr_ssim(clear_img_path, hazy_img_path, clear_img_names, hazy_img_names)