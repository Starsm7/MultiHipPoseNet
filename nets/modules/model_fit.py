import os
import numpy as np
import torch
from eval.val_pre import predict_val
from nets.modules.MultiHipPoseNet_training import CE_Loss, Dice_loss, Focal_Loss,JointsOHKMMSELoss
from tqdm import tqdm
from datasets.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(n,model,path, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, save_dir, VOCdevkit_path,max_precision,local_rank=0):
    total_loss      = 0
    total_f_score   = 0
    val_loss        = 0
    val_f_score     = 0
    loss_p_train    = 0
    loss_p_val      = 0

    Loss1_train=[]
    Loss2_train=[]
    A1_train=[]
    A2_train = []
    Loss1_val = []
    Loss2_val = []
    A1_val = []
    A2_val = []
    loss_F2 = JointsOHKMMSELoss(True)
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',mininterval=0.3)
    model.train()
    for iteration, batch in enumerate(gen):

        if iteration >= epoch_step: 
            break

        imgs, pngs, labels, heatmaps= batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                heatmaps = heatmaps.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        # Forward propagation
        outputs,out2 = model(imgs)
        # Loss calculation
        if focal_loss:
            loss1 = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
        else:
            loss1 = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

        if dice_loss:
            main_dice = Dice_loss(outputs, labels)
            loss1      = loss1 + main_dice
        loss_points=loss_F2(out2, heatmaps,torch.Tensor([.25,.25,.25,.25,.25,.25]).view(1,6).to('cuda'))

        # the achieve of the coefficient of variation
        a1,a2=0,0
        Loss1_train.append(loss1)
        if len(Loss1_train)==1:
            a1=0.5
        elif len(Loss1_train)==2 or len(Loss1_train)==3:
            A1_train.append(loss1 / torch.mean(torch.tensor(Loss1_train[:-1])))
            a1=0.5
        else:
            A1_train.append(loss1 / torch.mean(torch.tensor(Loss1_train[:-1])))
            xa1_train = torch.std(torch.tensor(A1_train))/torch.mean(torch.tensor(A1_train))
        Loss2_train.append(loss_points)
        if len(Loss2_train)==1:
            a2=0.5
        elif len(Loss2_train)==2 or len(Loss2_train)==3:
            A2_train.append(loss_points / torch.mean(torch.tensor(Loss2_train[:-1])))
            a2=0.5
        else:
            A2_train.append(loss_points / torch.mean(torch.tensor(Loss2_train[:-1])))
            xa2_train = torch.std(torch.tensor(A2_train)) / torch.mean(torch.tensor(A2_train))
        if a1==0.5 and a2==0.5:
            pass
        else:
            a1=xa1_train/(xa1_train+xa2_train)
            a2=xa2_train/(xa1_train+xa2_train)
        loss=loss1*a1+loss_points*a2

        loss_p_train+=loss_points.item()
        with torch.no_grad():
            _f_score = f_score(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'task1_weight': np.array(a1),
                                'task2_weight': np.array(a2),
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',mininterval=0.3)

    model.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels,heatmaps= batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                heatmaps = heatmaps.cuda(local_rank)
                weights = weights.cuda(local_rank)
            # Forward propagation
            outputs,out2 = model(imgs)
            # Loss calculation
            if focal_loss:
                loss1 = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss1 = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss1  = loss1 + main_dice
            loss_points=loss_F2(out2, heatmaps, torch.Tensor([.25, .25, .25, .25, .25,.25]).view(1, 6).to('cuda'))


            # the achieve of the coefficient of variation
            a1, a2 = 0, 0
            Loss1_val.append(loss1)
            if len(Loss1_val) == 1:
                a1 = 0.5
            elif len(Loss1_val) == 2 or len(Loss1_val) == 3:
                A1_val.append(loss1 / torch.mean(torch.tensor(Loss1_val[:-1])))
                a1 = 0.5
            else:
                A1_val.append(loss1 / torch.mean(torch.tensor(Loss1_val[:-1])))
                xa1_val = torch.std(torch.tensor(A1_val)) / torch.mean(torch.tensor(A1_val))
            Loss2_val.append(loss_points)
            if len(Loss2_val) == 1:
                a2 = 0.5
            elif len(Loss2_val) == 2 or len(Loss2_val) == 3:
                A2_val.append(loss_points / torch.mean(torch.tensor(Loss2_val[:-1])))
                a2 = 0.5
            else:
                A2_val.append(loss_points / torch.mean(torch.tensor(Loss2_val[:-1])))
                xa2_val = torch.std(torch.tensor(A2_val)) / torch.mean(torch.tensor(A2_val))
            if a1 == 0.5 and a2 == 0.5:
                pass
            else:
                a1 = xa1_val / (xa1_val + xa2_val)
                a2 = xa2_val / (xa1_val + xa2_val)
            loss = loss1 * a1 + loss_points * a2

            loss_p_val+=loss_points.item()
            _f_score    = f_score(outputs, labels)
            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                'task1_weight': np.array(a1),
                                'task2_weight': np.array(a2),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    a, b = predict_val(path+'/miou_out',VOCdevkit_path, model, n)
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        # -----------------------------------------------#
        # Save weights
        # -----------------------------------------------#
        no_improve_epochs = False

        # Determining whether a multitasking model is boosted or not, thus serving as a basis for regulating the learning rate, max_precision
        if epoch + 1 > 0:
            if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
                print(f'Save min loss model to min_loss_epoch_weights{n}.pth')
                torch.save(model, os.path.join(save_dir, f"min_loss_epoch_weights{n}.pth"))
            print(f'.....................{max_precision}....................')
            if a * 0.3 + b * 0.3 + val_f_score * 0.4 / epoch_step_val * 100 >= max_precision:
                max_precision = a * 0.3 + b * 0.3 + val_f_score * 0.4 / epoch_step_val * 100
                no_improve_epochs = False
            else:
                no_improve_epochs = True
            print(f'。。。。。。。。。。。。。。。{max_precision}。。。。。。。。。。。。。。。。。。。。。')

    return loss_p_train,loss_p_val,a,b,+ val_f_score/epoch_step_val,no_improve_epochs,max_precision