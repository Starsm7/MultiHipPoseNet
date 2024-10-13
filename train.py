import os
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nets.modules.config import get_parser
from datasets.dataloader import dataset_collate, MDataset
from nets.MultiHipPoseNet_plus import MultiHipPoseNet
from nets.modules.MultiHipPoseNet_training import weights_init
from datasets.utils import show_config, worker_init_fn, seed_everything
import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.callbacks import EvalCallback, LossHistory
from nets.modules.model_fit import fit_one_epoch


# Train the master file, click to train the model


def train(kplits,epochs,path,VOCdevkit_path='',save_dir=''):
    """
    kplits: the number of cross-validation
    """
    # training parameter, you can learn more in the model profile, get_parser().
    args = get_parser()
    input_shape = [args.input_h,args.input_w]
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    eval_flag           = True
    eval_period         = 15
    # Loss function occupancy for background and each key anatomical structure
    dice_loss       = True
    focal_loss      = True
    cls_weights     = np.array([1,5,5,5,5,5,5,5], np.float32)
    # the set of Cuda
    local_rank      = 0
    rank            = 0
    Cuda = True if args.device.lower() == 'cuda' else False
    # Model setup
    model=[MultiHipPoseNet(hip_classes=args.hip_classes,kpt_n=args.kpt_n,n_expert=args.n_expert, n_task=args.n_task, use_gate=args.use_gate),MultiHipPoseNet(hip_classes=args.hip_classes,kpt_n=args.kpt_n,n_expert=args.n_expert, n_task=args.n_task, use_gate=args.use_gate),
           MultiHipPoseNet(hip_classes=args.hip_classes,kpt_n=args.kpt_n,n_expert=args.n_expert, n_task=args.n_task, use_gate=args.use_gate),MultiHipPoseNet(hip_classes=args.hip_classes,kpt_n=args.kpt_n,n_expert=args.n_expert, n_task=args.n_task, use_gate=args.use_gate),
           MultiHipPoseNet(hip_classes=args.hip_classes,kpt_n=args.kpt_n,n_expert=args.n_expert, n_task=args.n_task, use_gate=args.use_gate)]
    # Start! 5-cross-validation cycle
    for n in range(kplits):
        weights_init(model[n])
        model[n].to(args.device)
        if local_rank == 0:
            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
            loss_history    = LossHistory(log_dir)
        else:
            loss_history    = None
        with open(os.path.join(VOCdevkit_path, f"VOC2007/ImageSets/Segmentation/train_fold{n}.txt"),"r") as f:
            train_lines = f.readlines()
        with open(os.path.join(VOCdevkit_path, f"VOC2007/ImageSets/Segmentation/val_fold{n}.txt"),"r") as f:
            val_lines = f.readlines()
        num_train   = len(train_lines)
        num_val     = len(val_lines)
        if local_rank == 0:
            show_config(
                num_classes=(args.hip_classes, args.kpt_n), input_shape=input_shape, num_train_val=(num_train, num_val),
                n_expert=args.n_expert, n_task=args.n_task, use_gate=args.use_gate,
                learning_rate=args.learning_rate, batch_size=args.batch_size,
                max_epoach_no_improve=args.max_epoach_no_improve,
                show_point_on_picture=args.show_point_on_picture, save_dir=save_dir)
            eval_callback = EvalCallback(model[n], input_shape, args.hip_classes, val_lines, VOCdevkit_path, log_dir, Cuda,miou_out_path=path+"/temp_miou_out",
                                     eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None
        if True:
            nbs             = 16
            lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit     = min(max(args.batch_size / nbs * args.learning_rate, lr_limit_min), lr_limit_max)
            # Using adam
            optimizer = {
                'adam'  : optim.Adam(model[n].parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
                'sgd'   : optim.SGD(model[n].parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
            }[optimizer_type]
            # Using ReduceLROnPlateau to update the learning rate
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)
            epoch_step      = num_train // args.batch_size
            epoch_step_val  = num_val // args.batch_size
            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("The dataset is too small to continue training, please expand the dataset.")
            # the reading of dataset 
            train_dataset   = MDataset(train_lines, input_shape, args.hip_classes, False, VOCdevkit_path)
            val_dataset     = MDataset(val_lines, input_shape, args.hip_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset,shuffle=True, batch_size = args.batch_size, num_workers = 0, pin_memory=True,
                                         drop_last = True, collate_fn = dataset_collate,worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
            gen_val         = DataLoader(val_dataset , shuffle=True,batch_size =args.batch_size, num_workers = 0, pin_memory=True,
                                        drop_last = True, collate_fn = dataset_collate,worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
            no_improve_epochs=0
            epoch_list=[]
            total_loss_list_train=[]
            total_loss_list_val=[]
            A=[]
            B=[]
            max_precision = 0
            # cycle
            for epoch in range(epochs):
                totallosslisttrain, totallosslistval, a, b, fcore, no,max_precision = fit_one_epoch(n, model[n],path,
                                                                                                  loss_history,
                                                                                                  eval_callback,
                                                                                                  optimizer, epoch,
                                                                                                  epoch_step,
                                                                                                  epoch_step_val, gen,
                                                                                                  gen_val,
                                                                                                  epochs, Cuda,
                                                                                                  dice_loss, focal_loss,
                                                                                                  cls_weights,
                                                                                                  args.hip_classes, save_dir,
                                                                                                  VOCdevkit_path,
                                                                                                  max_precision,
                                                                                                  local_rank)
                print(f'{epoch+1}:The composite indicator is {a * 0.3 + b * 0.3 + fcore * 100 * 0.4},max{max_precision}')
                scheduler.step(a * 0.3 + b * 0.3 + fcore * 100 * 0.4)
                epoch_list.append(epoch + 1)
                total_loss_list_train.append(totallosslisttrain)
                total_loss_list_val.append(totallosslistval)
                A.append(a)
                B.append(b)
                if no == True :
                    no_improve_epochs += 1
                    if no_improve_epochs >= args.max_epoach_no_improve:
                        print("Overfitting, run stop !!!!")
                        break
                else:
                    no_improve_epochs=0
            fig, axs = plt.subplots(2, 2)
            axs[0,0].plot(epoch_list, total_loss_list_train, label='Train Total Loss', marker='o')
            axs[0,0].set_title('Train Total Loss')
            axs[0,0].set_xlabel('Epoch')
            axs[0,0].set_ylabel('Loss')
            axs[0,1].plot(epoch_list, A, label='alpha', marker='o')
            axs[0,1].set_title('Alpha')
            axs[0,1].set_xlabel('Epoch')
            axs[0,1].set_ylabel('Successful percent(5°)')
            axs[1,0].plot(epoch_list, total_loss_list_val, label='Val Total Loss', marker='o')
            axs[1,0].set_title('Val Total Loss')
            axs[1,0].set_xlabel('Epoch')
            axs[1,0].set_ylabel('Loss')
            axs[1,1].plot(epoch_list, B, label='beta', marker='o')
            axs[1,1].set_title('Beta')
            axs[1,1].set_xlabel('Epoch')
            axs[1,1].set_ylabel('Successful percent(5°)')
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            plt.savefig(save_dir+f'/results_keypoints{n}.png')
            if local_rank == 0:
                loss_history.writer.close()
if __name__ == "__main__":
    seed = 967
    file_path=os.getcwd()
    seed_everything(seed)
    train(5, 15, file_path, file_path+'/VOCdevkit', file_path+'/logs')
