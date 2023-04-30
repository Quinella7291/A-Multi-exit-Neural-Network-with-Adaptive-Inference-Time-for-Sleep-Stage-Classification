## Attention based MultiModal Sleep Staging Network

from re import T
# from turtle import forward
import torch
from torch._C import TensorType
import torch.nn as nn
from model.TransformerEncoderLayer import TransformerEncoderLayer
# from model.fairseq.modules.tuckerhead_attention import TransformerEncoderLayer
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
from torch.nn import init
from thop import profile
from thop import clever_format

from model.BasicModel import BasicModel

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Temporal_feature_EEG(nn.Module):
    def __init__(self,channels):
        super(Temporal_feature_EEG, self).__init__()
        drate = 0.5
        # self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)
        self.features1 = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=64, stride=8, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            CBAMLayer(64),

            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(), 
    
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(), 

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            CBAMLayer(128),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
            
        )
        

        self.features2 = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=512, stride=64, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            CBAMLayer(64),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(), 
    
            nn.Conv1d(128, 128, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(), 

            nn.Conv1d(128, 128, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            CBAMLayer(128),

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
    

    def forward(self, y):
        if len(y.shape) == 2:
            y = y.unsqueeze(dim = 1)
        y1 = self.features1(y)
        y2 = self.features2(y)
        # print(y1.shape)
        y_concat = torch.cat((y1, y2), dim=2)
        y_concat = self.dropout(y_concat)
        # y_concat = self.AFR(y_concat)
        return y_concat

class Temporal_feature_multimodel(nn.Module):
    def __init__(self,channels):
        super(Temporal_feature_multimodel, self).__init__()
        drate = 0.5
        self.features1 = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=64, stride=8, bias=False, padding=24),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            CBAMLayer(32),

            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(), 
    
            nn.Conv1d(64, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(), 

            nn.Conv1d(64, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            CBAMLayer(64),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
            
        )
        
        self.features2 = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=512, stride=64, bias=False, padding=24),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            CBAMLayer(32),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(32, 64, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(), 
    
            nn.Conv1d(64, 64, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(), 

            nn.Conv1d(64, 64, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            CBAMLayer(64),

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
    
    def forward(self,x):
        
        if len(x.shape) == 2:
            x = x.unsqueeze(dim = 1)
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)

        return x_concat

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=8, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            # nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv1d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class Spatial_EEG_Feature_Fusion(nn.Module):

    def __init__(self, inplanes=2, planes=1, stride=1, reduction=1):
        super(Spatial_EEG_Feature_Fusion, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=(stride, stride))
        # self.bn1 = nn.BatchNorm2d(planes)  # [b,c,h,w],作用于c（channel）特征上，即平面（h，w）上
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(planes, planes, 1)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(inplanes, reduction)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=(stride, stride))
        self.bn1 = nn.BatchNorm2d(planes)  # [b,c,h,w],作用于c（channel）特征上，即平面（h，w）上
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()



    def forward(self, x):
        residual = x
        out = self.se(x)
        out += residual
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _  = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        y=y.view(b, c, 1 ,1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=1):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,stride=(stride,stride))
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        # print("x.shape:",x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print("out shape:",out.shape)
        # [256, 30, 128, 24]
        out = self.se(out)
        # print("out.shape:",out.shape)

        if self.downsample is not None:
            residual = self.downsample(x)

        # print("x_downsample.shape:",residual.shape)
        out += residual
        out = self.relu(out)

        return out

class FastClassifier(nn.Module):
    def __init__(self, cfg):
        super(FastClassifier, self).__init__()
        """
        多出口的出口分类器
        """
        self.d_model = cfg.d_model
        self.afr_reduced_cnn_size = cfg.afr_reduced_cnn_size
        self.inplanes = cfg.inplanes
        self.nhead=cfg.nhead
        self.num_layers=cfg.num_layers

        # 1.SELayer
        self.AFR = self._make_layer(SEBasicBlock, self.afr_reduced_cnn_size, blocks=1, stride=1)

        # 2.Transformer
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.nhead, batch_first=True, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 3.Classifier
        self.linear1 = nn.Sequential(nn.Linear(self.d_model*self.afr_reduced_cnn_size*16,32),nn.ReLU(True))
        self.linear2 = nn.Sequential(nn.Linear(32,5),nn.Softmax(dim=1))

    def forward(self, x, which_layer = 'eeg'):
        """
        分类器
        :param x: 应该是不同维度的矩阵，0:[eeg],1:[eeg, eog], 2:[eeg, eog, emg]
        :return:
        """
        self.batch_size = x.shape[0]
        if which_layer != 'eeg': # 第0层就一层eeg，用不用se都一样，这里跳过
            x_afr = self.AFR(x)
        else:
            x_afr = x

        x_afr = x_afr.view(self.batch_size, -1, 96)

        encoded_features = self.transformer_encoder(x_afr.view(self.batch_size, -1, 96))
        encoded_features = x_afr * encoded_features
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)

        x_final = self.linear1(encoded_features)
        x_final = self.linear2(x_final)

        return x_final

    def _make_layer(self, block, planes, blocks, stride=4):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class ClassifiyLayer(nn.Module):
    def __init__(self, cfg):
        super(ClassifiyLayer, self).__init__()
        """
        该层应该实现一个功能：
        1.卷积+cbma
        # 2.SELayer
        # 3.TransformerEncoderLayer
        """

        self.d_model = cfg.d_model
        self.EEG_channels=cfg.EEG_channels


        self.EEG_feature = Temporal_feature_EEG(channels=self.EEG_channels) #eeg信号卷积（fpz-cz，pz-oz）
        self.EEG_feature_fusion = Spatial_EEG_Feature_Fusion()

        self.EOG_feature = Temporal_feature_multimodel(channels=1) #eog信号卷积

        self.EMG_feature = Temporal_feature_multimodel(channels=1) #emg信号卷积


    def forward(self, x, last_layer_x = None, which_layer = 'eeg'):
        self.batch_size = x.shape[0]

        if which_layer == 'eeg':
            x_EEG=self.EEG_feature(x).view(self.batch_size,2,64,24)
            x_EEG = self.EEG_feature_fusion(x_EEG)
            x_cat = x_EEG

        elif which_layer == 'eog':
            x.squeeze().unsqueeze(1)  #x是eog信号 [batch_size, channel, embedding]
            x_EOG = self.EOG_feature(x).view(self.batch_size, 1, 64, 24)
            x_cat = torch.cat((last_layer_x, x_EOG), dim=1)  # [256,2,64,24] 2->1eeg+1eog; last_layer_x = residual eeg

        elif which_layer == 'emg':
            x.squeeze().unsqueeze(1)  # [batch_size, channel, embedding]
            x_EMG = self.EMG_feature(x).view(self.batch_size, 1, 64, 24) #x是emg信号 [batch_size, channel, embedding]
            x_cat = torch.cat((last_layer_x, x_EMG), dim=1)  # [256,3,64,24] (2->)1eeg+1eog+1emg ; last_layer_x = residual eeg+eog

        return x_cat



class DynamicSleepNetGraph(nn.Module):
    def __init__(self, cfg):
        super(DynamicSleepNetGraph, self).__init__()
        # 参数
        self.afr_reduced_cnn_size_emg = cfg.afr_reduced_cnn_size
        self.afr_reduced_cnn_size_eog = cfg.afr_reduced_cnn_size - 1
        self.afr_reduced_cnn_size_eeg = cfg.afr_reduced_cnn_size - 2
        self.d_model = cfg.d_model
        self.inplanes = cfg.inplanes
        self.nhead=cfg.nhead
        self.num_layers=cfg.num_layers
        self.EEG_channels=cfg.EEG_channels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_class = torch.tensor(5).to(self.device)

        self.EEG_feature = Temporal_feature_EEG(channels=self.EEG_channels)
        self.EOG_feature = Temporal_feature_multimodel(channels=1)
        self.EMG_feature = Temporal_feature_multimodel(channels=1)
        #试一试
        self.EEG_feature_fusion = Spatial_EEG_Feature_Fusion()


        """
        多出口的出口分类器
        """
        # 1.SELayer
        self.AFR_STUDENT_EOG = self._make_layer(SEBasicBlock, self.afr_reduced_cnn_size_eog, blocks=1, stride=1)
        self.AFR_TEACHER = self._make_layer(SEBasicBlock, self.afr_reduced_cnn_size_emg, blocks=1, stride=1)
        # self.AFR_STUDENT_EEG = self._make_layer(SEBasicBlock, self.afr_reduced_cnn_size, blocks=1, stride=1)

        # todo(重要改动)学生分类器减少3个头
        # 2.Transformer
        ## teacher_layer
        encoder_teacher_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.nhead, batch_first=True, activation="relu")
        self.teacher_layer = nn.TransformerEncoder(encoder_teacher_layer, num_layers=self.num_layers)
        ## student_eeg_layer
        encoder_student_eeg_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.nhead-3, batch_first=True, activation="relu")
        self.student_eeg_layer = nn.TransformerEncoder(encoder_student_eeg_layer, num_layers=self.num_layers)
        ## studuent_eog_layer
        encoder_student_eog_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.nhead-3, batch_first=True, activation="relu")
        self.student_eog_layer = nn.TransformerEncoder(encoder_student_eog_layer, num_layers=self.num_layers)

        # 3.Classifier
        # todo(重要改动)softmax注释掉了
        # teacher_emg
        self.linear_teacher1 = nn.Sequential(nn.Linear(self.d_model*self.afr_reduced_cnn_size_emg*16,32),nn.ReLU(True))
        # self.linear_teacher2 = nn.Sequential(nn.Linear(32,5),nn.Softmax(dim=1))
        self.linear_teacher2 = nn.Sequential(nn.Linear(32,5))

        # student_eeg
        self.linear_student_EEG1 = nn.Sequential(nn.Linear(self.d_model*self.afr_reduced_cnn_size_eeg*16,32),nn.ReLU(True))
        # self.linear_student_EEG2 = nn.Sequential(nn.Linear(32,5),nn.Softmax(dim=1))
        self.linear_student_EEG2 = nn.Sequential(nn.Linear(32,5))

        # student_eog
        self.linear_student_EOG1 = nn.Sequential(nn.Linear(self.d_model*self.afr_reduced_cnn_size_eog*16,32),nn.ReLU(True))

        # self.linear_student_EOG2 = nn.Sequential(nn.Linear(32,5),nn.Softmax(dim=1))
        self.linear_student_EOG2 = nn.Sequential(nn.Linear(32,5))

        """
        1.定义 ClassifiyLayer 层
        2.定义 FastClassifier 分类器
        
        """
        # self.cfg = cfg
        # step 1：定义 ClassifiyLayer 层
        # self.layers = ClassifiyLayer(cfg)

        # step 2：定义 FastClassifier 层 分类器
        # self.layer_classifier = FastClassifier(cfg=cfg)
        # self.layer_classifiers = nn.ModuleDict()

        # ## 分支（Branch）分类器（Student）
        # for i in range(2): #0：eeg分支分类器；1：eog分支分类器
        #     self.layer_classifiers['branch_classifier_' + str(i)] = copy.deepcopy(self.layer_classifier)
        # ## 主干（Backbone）分类器（Teacher）
        # self.layer_classifiers['final_classifier'] = self.layer_classifier


    def forward(self, x1,x2,x3, inference=False, inference_speed=0.1, training_stage=0):
        if inference:
            # ----------推理阶段，配置相对复杂-----------#
            # 定义基本参数
            batch_size = x1.shape[0]
            nb_class = self.num_class

            device = self.device

            # 定义全局参数，删除元素时，追踪每个样本的原始位置
            final_probs = torch.zeros((batch_size, nb_class), device=device) #
            uncertain_infos = torch.zeros((batch_size, 5), device=device) # uncertain < speed，输出文本
            positions = torch.arange(start=0, end=batch_size, device=device).long()
            all_probs = []

            # 样本通过eeg分类器
            eeg_cat = self.conv_layer(x=x1, last_layer_x=None, which_layer='eeg')
            eeg_logits = self.classify_layer(eeg_cat, which_classifiy_layer='student_eeg') #取消了softmax，真正的logits
            #求eeg分类器的uncertain
            eeg_prob = F.softmax(eeg_logits, dim=-1)
            eeg_log_prob = F.log_softmax(eeg_logits, dim=-1)
            eeg_uncertain = torch.sum(eeg_prob * eeg_log_prob, 1) / (-torch.log(self.num_class))

            eeg_enough_info = eeg_uncertain < inference_speed #[true, false, true,...]
            eeg_certain_positions = positions[eeg_enough_info]  # [0,2,...]
            # 能在eeg出来的sample的softmax分布在这里确定了最终结果
            final_probs[eeg_certain_positions] = eeg_prob[eeg_enough_info] #[0:[ppppp],1:[00000],2:[ppppp]]
            # 记录eeg出口的分类准确率
            all_probs.append(final_probs.clone())
            # eeg低于speed的uncertain值放在这里做记录
            uncertain_infos[eeg_certain_positions] = eeg_uncertain[eeg_enough_info].unsqueeze(1)  # [0:[uuuuu],1[00000],2:[uuuuu]]

            #没被eeg分类器分类出去的样本给到x2和x3
            x2=x2[~eeg_enough_info] #~enough_info=[False, True, False,...]
            x3 = x3[~eeg_enough_info]
            eeg_cat = eeg_cat[~eeg_enough_info]
            #如果全部分类完成，直接返回
            if x2.shape[0] == 0:
                return final_probs,0,uncertain_infos, all_probs
            #更新未分类出去的样本的位置集合
            positions = positions[~eeg_enough_info]


            #样本通过eog分类器
            inference_speed=1
            eog_cat = self.conv_layer(x=x2, last_layer_x=eeg_cat, which_layer='eog')
            eog_logits = self.classify_layer(eog_cat, which_classifiy_layer='student_eog')  # 取消了softmax，真正的logits
            # 求eog分类器的uncertain
            eog_prob = F.softmax(eog_logits, dim=-1)
            eog_log_prob = F.log_softmax(eog_logits, dim=-1)
            eog_uncertain = torch.sum(eog_prob * eog_log_prob, 1) / (-torch.log(self.num_class))


            eog_enough_info = eog_uncertain < inference_speed  # [true, false, true,...]
            eog_certain_positions = positions[eog_enough_info]  # [0,2,...]
            # 能在eog出来的sample的softmax分布在这里确定了最终结果
            final_probs[eog_certain_positions] = eog_prob[eog_enough_info] #[0:[ppppp],1:[00000],2:[ppppp]]
            # 记录eog出口的分类准确率
            all_probs.append(final_probs.clone())
            # eog低于speed的uncertain值放在这里做记录
            uncertain_infos[eog_certain_positions] = eog_uncertain[eog_enough_info].unsqueeze(1)  # [0:[uuuuu],1[00000],2:[uuuuu]]
            #没被eog分类器分类出去的样本给到x3
            x3=x3[~eog_enough_info] #~enough_info=[False, True, False,...]
            eog_cat = eog_cat[~eog_enough_info]
            # 如果全部分类完成，直接返回
            if x3.shape[0] == 0:
                return final_probs,1,uncertain_infos, all_probs
            #更新未分类出去的样本的位置集合
            positions = positions[~eog_enough_info]


            #样本通过emg分类器
            emg_cat = self.conv_layer(x=x3, last_layer_x=eog_cat, which_layer='emg')
            emg_logits = self.classify_layer(emg_cat, which_classifiy_layer='teacher')  # 取消了softmax，真正的logits

            # 求emg分类器的uncertain
            emg_prob = F.softmax(emg_logits, dim=-1)
            emg_log_prob = F.log_softmax(emg_logits, dim=-1)
            emg_uncertain = torch.sum(emg_prob * emg_log_prob, 1) / (-torch.log(self.num_class))

            emg_certain_positions = positions
            final_probs[emg_certain_positions] = emg_prob
            all_probs.append(final_probs.clone())
            uncertain_infos[emg_certain_positions] = emg_uncertain.unsqueeze(1)

            return final_probs,2, uncertain_infos, all_probs


        else:
            # ------训练阶段, 第一阶段初始训练, 第二阶段蒸馏训练--------#
            if training_stage == 0:
                eeg_cat = self.conv_layer(x=x1, last_layer_x=None, which_layer='eeg')
                eog_cat = self.conv_layer(x=x2, last_layer_x=eeg_cat, which_layer='eog')
                emg_cat = self.conv_layer(x=x3, last_layer_x=eog_cat, which_layer='emg')

                # eeg_cat = self.layers(x1, last_layer_x = None, which_layer = 'eeg')
                # eeg_eog_cat = self.layers(x2, last_layer_x = eeg_cat, which_layer = 'eog')
                # eeg_eog_emg_cat = self.layers(x3, last_layer_x = eeg_eog_cat, which_layer = 'emg')

                # x_final = self.layer_classifier(emg_cat, which_layer = 'emg')

                x_final = self.classify_layer(emg_cat, which_classifiy_layer='teacher')
                return x_final
                # x_final = self.classify_layer(eeg_cat, which_classifiy_layer='student_eeg') #消融
                # return x_final


            elif training_stage == 1:

                eeg_cat = self.conv_layer(x=x1, last_layer_x=None, which_layer='eeg')
                eog_cat = self.conv_layer(x=x2, last_layer_x=eeg_cat, which_layer='eog')
                emg_cat = self.conv_layer(x=x3, last_layer_x=eog_cat, which_layer='emg')

                all_logits = []
                eeg_logits = self.classify_layer(eeg_cat, which_classifiy_layer='student_eeg')
                all_logits.append(eeg_logits)
                eog_logits = self.classify_layer(eog_cat, which_classifiy_layer='student_eog')
                all_logits.append(eog_logits)
                emg_logits = self.classify_layer(emg_cat, which_classifiy_layer='teacher')
                all_logits.append(emg_logits)

                # todo(重要)，第一阶段使用softmax + CrossEntropy（外面）；第二阶段使用softmax + KL散度（外面）
                # loss = 0.0
                # teacher_log_prob = F.log_softmax(all_logits[-1], dim=-1)
                # for student_logits in all_logits[:-1]:
                #     student_prob = F.softmax(student_logits, dim=-1)
                #     student_log_prob = F.log_softmax(student_logits, dim=-1)
                #     uncertain = torch.sum(student_prob * student_log_prob, 1) / (-torch.log(self.num_class))
                #     # print('uncertain:', uncertain[0])
                #
                #     D_kl = torch.sum(student_prob * (student_log_prob - teacher_log_prob), 1)
                #     D_kl = torch.mean(D_kl)
                #     loss += D_kl
                return all_logits #里面包括[eeg_logits,eog_logits,emg_logits]三层结果

    def conv_layer(self, x, last_layer_x = None, which_layer = 'eeg'):
        self.batch_size = x.shape[0]

        if which_layer == 'eeg':
            x_EEG=self.EEG_feature(x).view(self.batch_size,2,64,24)
            x_EEG = self.EEG_feature_fusion(x_EEG)
            x_cat = x_EEG


        elif which_layer == 'eog':
            x.squeeze().unsqueeze(1)  #x是eog信号 [batch_size, channel, embedding]
            x_EOG = self.EOG_feature(x).view(self.batch_size, 1, 64, 24)
            x_cat = torch.cat((last_layer_x, x_EOG), dim=1)  # [256,2,64,24] 2->1eeg+1eog; last_layer_x = residual eeg

        elif which_layer == 'emg':
            x.squeeze().unsqueeze(1)  # [batch_size, channel, embedding]
            x_EMG = self.EMG_feature(x).view(self.batch_size, 1, 64, 24) #x是emg信号 [batch_size, channel, embedding]
            x_cat = torch.cat((last_layer_x, x_EMG), dim=1)  # [256,3,64,24] (2->)1eeg+1eog+1emg ; last_layer_x = residual eeg+eog

        return x_cat

    def classify_layer(self, x, which_classifiy_layer = 'teacher'):
        """
        分类器
        :param x: 应该是不同维度的矩阵，0:[eeg],1:[eeg, eog], 2:[eeg, eog, emg]
        :return:
        """
        self.batch_size = x.shape[0]

        if which_classifiy_layer == 'student_eeg': # 第0层就一层eeg，用不用se都一样，这里跳过
            x_afr_eeg = x
            x_afr_eeg = x_afr_eeg.view(self.batch_size, -1, 96)

            encoded_student_eeg_features = self.student_eeg_layer(x_afr_eeg.view(self.batch_size, -1, 96))
            # encoded_student_eeg_features = x_afr_eeg*encoded_student_eeg_features
            encoded_student_eeg_features = encoded_student_eeg_features.contiguous().view(encoded_student_eeg_features.shape[0], -1)

            x_student_eeg_final = self.linear_student_EEG1(encoded_student_eeg_features)
            x_student_eeg_final = self.linear_student_EEG2(x_student_eeg_final)

            return x_student_eeg_final

        elif which_classifiy_layer == 'student_eog':
            x_afr_eog = self.AFR_STUDENT_EOG(x)
            x_afr_eog = x_afr_eog.view(self.batch_size, -1, 96)

            encoded_student_eog_features = self.student_eog_layer(x_afr_eog.view(self.batch_size, -1, 96))
            encoded_student_eog_features = x_afr_eog*encoded_student_eog_features
            encoded_student_eog_features = encoded_student_eog_features.contiguous().view(encoded_student_eog_features.shape[0], -1)

            x_student_eog_final = self.linear_student_EOG1(encoded_student_eog_features)
            x_student_eog_final = self.linear_student_EOG2(x_student_eog_final)

            return x_student_eog_final


        elif which_classifiy_layer == 'teacher':
            x_afr_emg = self.AFR_TEACHER(x)
            x_afr_emg = x_afr_emg.view(self.batch_size, -1, 96)

            encoded_teacher_features = self.teacher_layer(x_afr_emg.view(self.batch_size, -1, 96))
            encoded_teacher_features = x_afr_emg * encoded_teacher_features
            encoded_teacher_features = encoded_teacher_features.contiguous().view(encoded_teacher_features.shape[0], -1)

            x_teacher_final = self.linear_teacher1(encoded_teacher_features)
            x_teacher_final = self.linear_teacher2(x_teacher_final)

            return x_teacher_final

        # x_afr = x_afr.view(self.batch_size, -1, 96)
        #
        # encoded_features = self.transformer_encoder(x_afr.view(self.batch_size, -1, 96))
        # encoded_features = x_afr * encoded_features
        # encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        #
        # x_final = self.linear1(encoded_features)
        # x_final = self.linear2(x_final)
        #
        # return x_final

    def _make_layer(self, block, planes, blocks, stride=4):  # makes residual SE block
        downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.inplanes, planes * block.expansion,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(planes * block.expansion),
        #     )

        self.inplanes = planes * block.expansion
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class DynamicSleepNet(BasicModel):
    def __init__(self, cfg):
        super(DynamicSleepNet, self).__init__()
        self.training_stage = cfg.training_stage
        self.inference_speed = cfg.inference_speed
        self.inference = cfg.inference
        self.fast_net_graph = DynamicSleepNetGraph(cfg=cfg)

    def forward(self, x1, x2, x3):
        x_final = self.fast_net_graph(x1, x2, x3, inference=self.inference, inference_speed=self.inference_speed, training_stage=self.training_stage)

        return x_final


# class MMASleepNet(BasicModel):
#     def __init__(self,cfg):#,d_model,afr_reduced_cnn_size):
#         super(MMASleepNet, self).__init__()
#
#         self.afr_reduced_cnn_size=cfg.afr_reduced_cnn_size -2
#         self.d_model = cfg.d_model
#         self.inplanes = cfg.inplanes
#         self.nhead=cfg.nhead
#         self.num_layers=cfg.num_layers
#         self.EEG_channels=cfg.EEG_channels
#
#         self.EEG_feature = Temporal_feature_EEG(channels=self.EEG_channels)
#         self.EOG_feature = Temporal_feature_multimodel(channels=1)
#         self.EMG_feature = Temporal_feature_multimodel(channels=1)
#
#         #试一试
#         self.EEG_feature_fusion = Spatial_EEG_Feature_Fusion()
#
#         self.linear1 = nn.Sequential(nn.Linear(self.d_model*self.afr_reduced_cnn_size*16,32),nn.ReLU(True))
#         self.linear2 = nn.Sequential(nn.Linear(32,5),nn.Softmax(dim=1))
#
#         self.AFR = self._make_layer(SEBasicBlock, self.afr_reduced_cnn_size, blocks = 1 ,stride=1)
#         encoder_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.nhead,batch_first=True,activation="relu")
#         #My TransformerEncoderLayer
#         # self.transformer_encoder = TransformerEncoderLayer(embed_dim=self.d_model, nhead=self.nhead,batch_first=True,activation="relu")
#         # self.transformer_encoder = TransformerEncoderLayer(embed_dim=self.d_model, nhead=self.nhead,dropout=0.1,batch_first=True,self_attention=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
#
#     def _make_layer(self, block, planes, blocks, stride=4):  # makes residual SE block
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x1,x2,x3):
#         self.batch_size=x1.shape[0]
#
#         # x2 = x2.unsqueeze(1)
#         # x3 = x3.unsqueeze(1)
#
#         x_EEG=self.EEG_feature(x1).view(self.batch_size,2,64,24)
#         x_EOG=self.EOG_feature(x2).view(self.batch_size,1,64,24)
#         x_EMG=self.EMG_feature(x3).view(self.batch_size,1,64,24)
#
#         x_EEG = self.EEG_feature_fusion(x_EEG)
#
#         x_cat=torch.cat((x_EEG,x_EOG,x_EMG),dim=1) #[256,4,64,24]
#         x_cat=x_EEG #[256,1,64,24]
#
#         x_afr = self.AFR(x_cat) #[256,4,64,24]
#
#         x_afr= x_afr.view(self.batch_size,-1,96)
#         x_afr= x_cat.contiguous().view(self.batch_size,-1,96)
#         # x_cat=x_cat.squeeze()
#         # print(x_afr.shape)
#         encoded_features=self.transformer_encoder(x_afr.view(self.batch_size,-1,96) )
#
#         # print(encoded_features.shape)
#
#         encoded_features=x_afr*encoded_features
#         encoded_features=encoded_features.contiguous().view(encoded_features.shape[0], -1)
#         # print(encoded_features.shape)
#         x_final = self.linear1(encoded_features)
#         x_final = self.linear2(x_final)
#         # print(x_final.shape)
#         return x_final









