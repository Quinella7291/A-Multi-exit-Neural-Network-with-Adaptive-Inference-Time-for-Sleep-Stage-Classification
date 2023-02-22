import torch
import numpy as np
from collections import Counter
from torch.nn import functional as F
from copy import deepcopy
from thop import profile
from thop import clever_format
from torchsummaryX import summary
device = "cuda" if torch.cuda.is_available() else "cpu"
from torchstat import stat
import  csv
from sklearn.metrics import f1_score, cohen_kappa_score

def train(dataloader, model, loss_fn, optimizer):
    pass

def test(dataloader, model, loss_fn):
    pass

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


# 2*EEG+EOG
def train_3ch(dataloader, model, loss_fn, optimizer, training_stage, inference):
    pass

def test_3ch(dataloader, model, loss_fn, training_stage, inference):
    pass


# 2*EEG+EOG+EMG
def train_4ch(dataloader, model, loss_fn, optimizer,training_stage, inference):
    size = len(dataloader.dataset)
    # print("train size:",size)
    model.train()
    # print(enumerate(dataloader).shape)
    for batch, (X_0,X_1,X_2,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print(X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        X1 = X_1.type(torch.FloatTensor).to(device)
        X2 = X_2.type(torch.FloatTensor).to(device)
        # Compute prediction error
        pred = model(X0,X1,X2)
        # print('y.shape',y.shape)
        if training_stage == 0 :
            loss = loss_fn(pred, y) #第一次训练，预测和y求loss
        elif training_stage == 1 :
            #蒸馏训练，student和teacher求loss
            loss =torch.tensor( 0.0).to(device)
            loss1 = torch.tensor( 0.0).to(device)
            loss2 = torch.tensor( 0.0).to(device)
            teacher_log_prob = F.log_softmax(pred[-1], dim=-1)
            for branch,student_logits in enumerate(pred[:-1]):
                student_prob = F.softmax(student_logits, dim=-1)
                student_log_prob = F.log_softmax(student_logits, dim=-1)
                # uncertain = torch.sum(student_prob * student_log_prob, 1) / (-torch.log(5)) # 5=self.num_class

                D_kl = torch.sum(student_prob * (student_log_prob - teacher_log_prob), 1)
                D_kl = torch.mean(D_kl)
                loss += D_kl
                if branch == 0:
                    loss1 = loss.clone()
                elif branch == 1:
                    loss2 = loss-loss1

            # --------------------不蒸馏看效果----------------------------
            # for branch,student_logits in enumerate(pred[:-1]):
            #     loss += loss_fn(pred[branch], y)



        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_0)
            if training_stage == 1:
                print(f"loss: {loss:>7f}  loss1:{loss1}  loss2:{loss2}  [{current:>5d}/{size:>5d}]") # training_stage == 1
            if training_stage == 0:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    if training_stage == 0:
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()/len(X_0)
        return loss.item(), 100*correct
    elif training_stage == 1:
        eeg_correct = (pred[0].argmax(1) == y).type(torch.float).sum().item()/len(X_0)
        eog_correct = (pred[1].argmax(1) == y).type(torch.float).sum().item()/len(X_0)
        emg_correct = (pred[2].argmax(1) == y).type(torch.float).sum().item()/len(X_0)
        return loss.item(), 100*eeg_correct, 100*eog_correct, 100*emg_correct

    # return loss, 100*correct

def test_4ch(dataloader, model, loss_fn, training_stage, inference):
    size = len(dataloader.dataset)
    # print("test size:",size)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    eeg_correct, eog_correct, emg_correct = 0,0,0
    with torch.no_grad():
        for X_0,X_1,X_2,y in dataloader:
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            X2 = X_2.type(torch.FloatTensor).to(device)
            pred = model(X0,X1,X2)
            if training_stage == 0:
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if training_stage == 1:
                # 蒸馏训练，student和teacher求loss

                teacher_log_prob = F.log_softmax(pred[-1], dim=-1)
                for student_logits in pred[:-1]:
                    student_prob = F.softmax(student_logits, dim=-1)
                    student_log_prob = F.log_softmax(student_logits, dim=-1)
                    # uncertain = torch.sum(student_prob * student_log_prob, 1) / (-torch.log(5)) # 5=self.num_class

                    D_kl = torch.sum(student_prob * (student_log_prob - teacher_log_prob), 1)
                    D_kl = torch.mean(D_kl)
                    test_loss += D_kl

                # --------------------不蒸馏看效果----------------------------
                # for branch, student_logits in enumerate(pred[:-1]):
                #     test_loss += loss_fn(pred[branch], y)


                eeg_correct += (pred[0].argmax(1) == y).type(torch.float).sum().item()
                eog_correct += (pred[1].argmax(1) == y).type(torch.float).sum().item()
                emg_correct += (pred[2].argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    if training_stage == 0:
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, 100 * correct
    elif training_stage == 1:
        eeg_correct /= size
        eog_correct /= size
        emg_correct /= size

        print(f"Test Error: \n EEG_Accuracy: {(100 * eeg_correct):>0.4f}%, EOG_Accuracy: {(100 * eog_correct):>0.4f}%, EMG_Accuracy: {(100 * emg_correct):>0.4f}%,  Avg loss: {test_loss:>8f} \n")
        return test_loss, 100 * eeg_correct, 100 * eog_correct,  100 * emg_correct


def inference_4ch(dataloader, model, loss_fn, training_stage, inference):
    size = len(dataloader.dataset)

    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    eeg_sumcorrect, eog_sumcorrect, emg_sumcorrect = 0,0,0
    eeg_sumsize,eog_sumsize,emg_sumsize = 0,0,0

    y_true = []
    y_predict=[]
    y_eeg_true = []
    y_eeg_predict=[]
    y_eog_true = []
    y_eog_predict=[]
    y_emg_true = []
    y_emg_predict=[]
    with torch.no_grad():
        for X_0,X_1,X_2,y in dataloader:
            eeg = []
            eog = []
            emg = []
            eeg_correct, eog_correct, emg_correct = 0, 0, 0
            eeg_size, eog_size, emg_size = 0, 0, 0
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            X2 = X_2.type(torch.FloatTensor).to(device)
            pred, output_layer_num, uncertain_infos, all_probs = model(X0,X1,X2)
            # inference最终acc结果
            if training_stage == -1 and inference:
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                # y_true = np.concatenate((y_true,y.cpu().numpy()),axis=0)
                # y_predict = np.concatenate((y_predict,pred.argmax(1).cpu().numpy()),axis=0)
            # 查看每一层的分类acc
            # 用到了final分类器（可能性最大）
            if output_layer_num >= 0:
                #原版本
                prob_label = zip(all_probs[0], y)
                for probs, label in prob_label:
                    if probs[0] == probs[1] == probs[2] == probs[3] == probs[4] == 0:
                        pass
                    else:
                        eeg_correct += (probs.argmax(0) == label).type(torch.float).item()
                        eeg_size += 1

                eeg_sumcorrect += eeg_correct
                eeg_sumsize +=eeg_size

                #一个更好的做法
                # prob_label = list(zip(all_probs[0], y))
                # for probs, label in prob_label:
                #     if probs[0] == probs[1] == probs[2] == probs[3] == probs[4] == 0:
                #         pass
                #     else:
                #         eeg.append((probs.argmax(0).cpu().numpy(),label.cpu().numpy()))
                #         y_eeg_true.append(label.cpu().numpy())
                #         y_eeg_predict.append(probs.argmax(0).cpu().numpy())

            if output_layer_num >= 1:
                prob_label = zip(all_probs[1], y)
                for probs, label in prob_label:
                    if probs[0] == probs[1] == probs[2] == probs[3] == probs[4] == 0:
                        pass
                    else:
                        eog_correct += (probs.argmax(0) == label).type(torch.float).item()
                        eog_size += 1
                eog_correct -=  eeg_correct
                eog_size -=  eeg_size
                eog_sumcorrect +=eog_correct
                eog_sumsize += eog_size


                # prob_label = list(zip(all_probs[1], y))
                # for probs, label in prob_label:
                #     if probs[0] == probs[1] == probs[2] == probs[3] == probs[4] == 0:
                #         pass
                #     else:
                #         eog.append((probs.argmax(0).cpu().numpy(),label.cpu().numpy()))
                # for i in eeg:
                #     eog.remove(i)
                # for pre,t in eog:
                #     y_eog_true.append(t)
                #     y_eog_predict.append(pre)

            if output_layer_num >= 2:
                prob_label = zip(all_probs[2], y)
                for probs, label in prob_label:
                    if probs[0] == probs[1] == probs[2] == probs[3] == probs[4] == 0:
                        assert False , '不应该还有没概率的标签'
                    else:
                        emg_correct += (probs.argmax(0) == label).type(torch.float).item()
                        emg_size += 1
                emg_correct -=  (eeg_correct + eog_correct)
                emg_size -=  (eeg_size + eog_size)
                emg_sumcorrect += emg_correct
                emg_sumsize += emg_size


                # prob_label = list(zip(all_probs[2], y))
                # for probs, label in prob_label:
                #     emg.append((probs.argmax(0).cpu().numpy(),label.cpu().numpy()))
                # for i in eeg:
                #     emg.remove(i)
                # for i in eog:
                #     emg.remove(i)
                # for pre,t in emg:
                #     y_emg_true.append(t)
                #     y_emg_predict.append(pre)


            assert output_layer_num != 0 or output_layer_num != 1, f'some batch exit earlier than excepted, exit layer: {output_layer_num}'




    print(f" total_size:{size}, total_correct:{correct}\n eeg_size:{eeg_sumsize}, eeg_correct:{eeg_sumcorrect}\n eog_size: {eog_sumsize} eog_correct:{eog_sumcorrect}\n emg_size: {emg_sumsize}, emg_correct: {emg_sumcorrect}")
    correct /= size
    if eeg_sumsize != 0:
        eeg_sumcorrect /= eeg_sumsize
    else:
        eeg_sumcorrect = 0
    if eog_sumsize != 0:
        eog_sumcorrect /= eog_sumsize
    else:
        eog_sumcorrect=0
    if emg_sumsize != 0:
        emg_sumcorrect /= emg_sumsize
    else:
        emg_sumcorrect=0



    # print(f"Test Error: \n Total_Accuracy:{100*correct}%, Layer1_Accuracy: {(100 * eeg_sumcorrect):>0.4f}%, Layer2_Accuracy: {(100 * eog_sumcorrect):>0.4f}%, LayerFin_Accuracy: {(100 * emg_sumcorrect):>0.4f}%")
    # return 100*correct, 100 * eeg_sumcorrect, 100 * eog_sumcorrect,  100 * emg_sumcorrect

    # f1score = f1_score(y_true, y_predict, average=None)
    # kappa = cohen_kappa_score(y_true, y_predict)
    # eeg_f1score = f1_score(y_eeg_true, y_eeg_predict, average='macro')
    # eeg_kappa = cohen_kappa_score(y_eeg_true, y_eeg_predict)
    # eog_f1score = f1_score(y_eog_true, y_eog_predict, average='macro')
    # eog_kappa = cohen_kappa_score(y_eog_true, y_eog_predict)
    # emg_f1score = f1_score(y_emg_true, y_emg_predict, average='macro')
    # emg_kappa = cohen_kappa_score(y_emg_true, y_emg_predict)
    # if len(y_eog_true) == 0:
    #     eog_kappa = 0
    #     eog_f1score = 0
    # if len(y_emg_true) == 0:
    #     emg_kappa = 0
    #     emg_f1score = 0
    # return eeg_f1score,eeg_kappa,eog_f1score,eog_kappa,emg_f1score,emg_kappa


    # return eeg_sumsize*100/size,eog_sumsize*100/size,emg_sumsize*100/size
    print(eeg_sumcorrect*100)
    assert 0

def compute_params(dataloader, model):

    model.eval()

    for X_0, X_1, X_2, y in dataloader:

        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        X1 = X_1.type(torch.FloatTensor).to(device)
        X2 = X_2.type(torch.FloatTensor).to(device)


        # summary(model,X0,X1,X2)

        flops, params = profile(model, inputs=(X0,X1,X2))
        flops, params = clever_format([flops, params], '% .3f')
        print(flops,params)
        break
    return flops,params


def train_salient(dataloader, model, loss_fn, optimizer):
    pass

def test_salient(dataloader, model, loss_fn):
    pass

def train_1ch(dataloader, model, loss_fn, optimizer):
    pass

def test_1ch(dataloader, model, loss_fn):
    pass

def train_printnet(dataloader, model, loss_fn, optimizer):
    pass

def test_printnet(dataloader, model, loss_fn):
    pass
