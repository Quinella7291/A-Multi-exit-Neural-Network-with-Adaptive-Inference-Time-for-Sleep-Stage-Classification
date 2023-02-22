import argparse
import collections
import sys 
sys.path.append("..") 
# Leave One Subject Out

from predata.data_loaders import LoadDataset_from_numpy,data_generator_np
from util import writer_func
from util.train_test import test_3ch, test_4ch, train, test, train_3ch, train_4ch, inference_4ch, compute_params
# from util.func import standard_scaler, normalizer, min_max_scaler, max_abs_scaler

import model

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
from torch.utils.data import ConcatDataset

from torch.utils.tensorboard import SummaryWriter
import os
from util.utils import *
import collections



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.manual_seed(0)
print("Using {} device".format(device))


# fix random seeds for reproducibility
SEED = 24
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)




class Config():
    def __init__(self,  
                # channels
                fold_id,
                subject,
                SS,
                subjects_num,
                 i=0
                ):

        self.scaler = None
        self.batch_size=256
        self.afr_reduced_cnn_size=3
        self.d_model = 96
        self.subjects_num=subjects_num
        #transformer
        self.ch=SS+1
        self.nhead=4
        self.num_layers=1
        self.inplanes=self.ch -1
        self.EEG_channels=2
        self.subject=subject
        self.SS=SS
        self.fold_id = fold_id
        self.lr = 0.0003
        self.epochs = 150

        self.training_stage = 0 # 0:pretrain 1:train -1: inference
        self.inference_speed = 0
        self.inference = False
        self.pretrained_model_path = '../main/ckpt/MMASleep-8-plus/sleepedf-78/fold0_subject42/1025_1449_model/MMASleepNet_epoch_68_acc_83.06637130442192.pth'
        self.trained_model_path ='../main/ckpt/MMASleep-8-plus/sleepedf-78/distill_model/MMASleepNet_epoch_142_acc_80.21640964471331.pth'

        # Model
        # self.model=model.MMASleepNet(self).to(device)
        self.model=model.DynamicSleepNet(self).to(device)
        # self.models = [model.MAttnSleep().to(device) for i in range(len(self.channels))]
        # self.model = model.EEGNet(C=8, T=30*128)
        
        # self.optimizer = torch.optim.SGD
        # self.optimizer = torch.optim.Adam([{"params":model.parameters()} for model in self.models], lr=self.lr, weight_decay=0.01, amsgrad=True)
        
        self.loss_fn=nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 1.80, 1.0, 1.25, 1.20]).to(device=device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.01, amsgrad=True)
        # self.scheduler = None
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.93, last_epoch=-1)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10], gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 10, last_epoch=-1)

        # self.Dataset = SleepAdult(data_path=self.data_path, downsample=128)
        # self.comment = f'_subject{subject}' # comment in the save dir name
        if self.training_stage == 0:
            self.output_dir = f'ckpt/MMASleep-8-plus/sleepedf-{self.subjects_num}/fold{self.fold_id}_subject{self.subject}/{self.model.init_time}_model/'
        elif self.training_stage == 1 :
            self.output_dir = f'ckpt/MMASleep-8-plus/sleepedf-{self.subjects_num}/distill_model/'

        elif self.training_stage == -1:
            self.output_dir = f'ckpt/MMASleep-8-plus/sleepedf-{self.subjects_num}/inference_results/'



# def main(fold_id,chs,subjects_num,writer,train_loader,test_loader,counts,i):
def main(fold_id,chs,subjects_num):
    print(f"------------------------------------FOLD_{fold_id}------------------------------------")
    # cfg = Config(fold_id=fold_id,subject=subjects[fold_id],SS=chs-1,subjects_num=subjects_num, i=i)
    cfg = Config(fold_id=fold_id,subject=subjects[fold_id],SS=chs-1,subjects_num=subjects_num)



    net = cfg.model #空的model
    # 将第一次训练模型数据导入蒸馏模型，有几个学生分类器代码改过了，初始化时不将这些预训练数据（实际没有训练到）导入模型
    if cfg.training_stage == 1:
        model_weight = torch.load(cfg.pretrained_model_path,map_location='cpu')
        new_state_dict = collections.OrderedDict()
        for k, v in model_weight.items():
            if 'AFR_STUDENT_EOG' not in k and 'linear_student_EEG1' not in k and 'linear_student_EOG1' not in k:
                new_state_dict[k] = v
        net.load_state_dict(new_state_dict,strict=False)

        # 冻结除分类器以外的模型
        for name, p in net.named_parameters():
            if "student" not in name and 'STUDENT' not in name:
                p.requires_grad = False

    if cfg.training_stage == -1:
        model_weight = torch.load(cfg.trained_model_path, map_location='cpu')
        net.load_state_dict(model_weight,strict=True)


    loss_fn = cfg.loss_fn
    optimizer=cfg.optimizer

    train_loader, test_loader, counts = data_generator_np(folds_data[fold_id][0],
                                                                   folds_data[fold_id][1], cfg.batch_size,SS=cfg.SS)

    writer = SummaryWriter(log_dir=cfg.output_dir)
    writer_func.save_config(writer, cfg)


    # if cfg.training_stage == 1:
    #     writer_b1 = SummaryWriter(log_dir=cfg.output_dir)
    #     writer_b2 = SummaryWriter(log_dir=cfg.output_dir)


    start_time = time.time()
    best_acc = 0
    best1_acc = 0
    best2_acc = 0
    for t in range(cfg.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        if chs==4:
            if cfg.training_stage == 0:
                train_loss, train_acc = train_4ch(train_loader, net, loss_fn, optimizer, cfg.training_stage, cfg.inference)
                test_loss,test_acc = test_4ch(test_loader, net, loss_fn, cfg.training_stage, cfg.inference)

            elif cfg.training_stage == 1:
                train_KLloss, train1_acc, train2_acc, trainfin_acc = train_4ch(train_loader, net, loss_fn, optimizer, cfg.training_stage, cfg.inference)
                test_KLloss,test1_acc, test2_acc, testfin_acc = test_4ch(test_loader, net, loss_fn, cfg.training_stage, cfg.inference)
            # 推理函数
            elif cfg.training_stage == -1 and cfg.inference:

                # 推理查看每个出口正确率
                # total_acc, test1_acc, test2_acc, testfin_acc = inference_4ch(test_loader, net, loss_fn, cfg.training_stage, cfg.inference)
                inference_4ch(test_loader, net, loss_fn, cfg.training_stage, cfg.inference)
                # 推理查看f1 k
                # w, n1, n2, n3,rem,kappa = inference_4ch(test_loader, net, loss_fn, cfg.training_stage, cfg.inference)
                # mf1 = (w+n1+n2+n3+rem)/5

                # 推理查看每个出口mf1 k
                # eeg_f1score,eeg_kappa,eog_f1score,eog_kappa,emg_f1score,emg_kappa = inference_4ch(test_loader, net, loss_fn, cfg.training_stage, cfg.inference)

                # 推理查看每个出口分类样本占总样本百分比
                # eeg_p,eog_p,emg_p = inference_4ch(test_loader, net, loss_fn, cfg.training_stage, cfg.inference)


                #计算不同推理速度的参数
                # flops,params = compute_params(test_loader, net)

                # writer.add_scalars('info',{'total_acc':total_acc,'test1_acc': test1_acc, 'test2_acc': test2_acc, 'test3_acc': testfin_acc}, i)
                # writer.add_scalar('flops',flops,i)
                # writer.add_scalar('params',params,i)

                #查看f1和kappa图
                # writer.add_scalars('info',{'w':w,'n1': n1, 'n2': n2, 'n3': n3, 'rem':rem, 'mf1':mf1,'kappa':kappa}, i)

                #查看每个出口f1和kappa图
                # writer.add_scalars('info',{'eeg_f1score':eeg_f1score,'eeg_kappa': eeg_kappa, 'eog_f1score': eog_f1score, 'eog_kappa': eog_kappa, 'emg_f1score':emg_f1score, 'mf1':emg_kappa}, i)

                writer.add_scalars('info',{'eeg_p':eeg_p,'eog_p': eog_p, 'emg_p': emg_p}, i)
                break #推理一次就够

        if cfg.scheduler is not None and cfg.training_stage != -1:
            cfg.scheduler.step()

        # 保存第一次训练的模型
        if cfg.training_stage == 0 and test_acc > best_acc:
            best_acc = test_acc
            net.save(cfg.output_dir,acc=best_acc, best=True, epoch=t)
        elif cfg.training_stage == 0 and test_acc > 80.0:
            net.save(cfg.output_dir, acc=test_acc, best=False, epoch=t)

        # 这里是保存蒸馏模型的逻辑
        if cfg.training_stage == 1:

            if test1_acc > 79.9:
                net.save_distill(cfg.output_dir, acc=test1_acc, best=False, epoch=t)

            # if test2_acc > best2_acc:
            #     best2_acc = test2_acc
            #     net.save_distill(cfg.output_dir, acc=best2_acc, best=True, epoch=t, stage=2)

        # tensorboard画图
        if cfg.training_stage == 0:
            writer_func.save_scalar(writer, {'Loss/train':train_loss,'Accuracy/train':train_acc,'Loss/test':test_loss, 'Accuracy/test':test_acc, 'lr':cfg.optimizer.state_dict()['param_groups'][0]['lr']}, t)

        elif cfg.training_stage == 1:
            writer.add_scalars('Accuracy/train',{'train1_acc':train1_acc,'train2_acc':train2_acc,'train3_acc':trainfin_acc},t)
            writer.add_scalars('Accuracy/test',{'test1_acc':test1_acc,'test2_acc':test2_acc,'test3_acc':testfin_acc},t)
            writer.add_scalars('Loss/train',{'train_loss':train_KLloss,'test_loss':test_KLloss},t)




    end_time = time.time()

    if cfg.training_stage == 0:
        print(f'best_acc {best_acc}, time {(end_time-start_time)/60}min')

    if cfg.training_stage == 0:
        writer_func.save_text(writer, {'best_acc':str(best_acc)})
        writer_func.save_text(writer, {'time':str((end_time-start_time)/60)})
    # if cfg.training_stage == 1:
    #     writer_b1.close()
    #     writer_b2.close()

    writer.close()

    print("Done!")
    


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    # Kfold，把数据分成k份进行交叉验证
    args.add_argument('-f', '--fold_id', default='9',type=str,
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', default='../predata/data/data_npy/sleepedf78-npz/',type=str,
                      help='Directory containing numpy files')
    args.add_argument('-ch', '--channel_number',default='4', type=str,
                      help='the number of channels')


    # CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)
    chs = int(args2.channel_number)
    # config = ConfigParser.from_args(args, fold_id, options)
    if "isruc" in args2.np_data_dir:
        # folds_data,subjects = load_folds_data_shhs(args2.np_data_dir, 10)
        # subjects_num=10
        pass
    elif  "78" in  args2.np_data_dir:
        folds_data,subjects = load_folds_data(args2.np_data_dir, 10)
        subjects_num=78
    elif  "20" in  args2.np_data_dir:
        folds_data,subjects = load_folds_data(args2.np_data_dir, 20)
        subjects_num=20


    # train_loader, test_loader, counts = data_generator_np(folds_data[fold_id][0],
    #                                                                folds_data[fold_id][1], 256,SS=chs-1)
    # writer = SummaryWriter(log_dir=f'ckpt/MMASleep-8-plus/sleepedf-78/inference_results/')

    # for i in range(100):
    #     main(fold_id,chs,subjects_num,writer,train_loader,test_loader,counts, i)
    main(fold_id,chs,subjects_num)

    # writer.close()