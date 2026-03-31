import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import pickle
from termcolor import colored
import math
import random
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
import argparse
import torch.backends.cudnn as cudnn
import gc

cudnn.benchmark = True
SEED = 6996
print("Seed was: ", SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda:1")


def genData(file, max_len):
    aa_dict = {
        "A": 1,
        "R": 2,
        "N": 3,
        "D": 4,
        "C": 5,
        "Q": 6,
        "E": 7,
        "G": 8,
        "H": 9,
        "I": 10,
        "L": 11,
        "K": 12,
        "M": 13,
        "F": 14,
        "P": 15,
        "S": 16,
        "T": 17,
        "W": 18,
        "Y": 19,
        "V": 20,
        "X": 21,
    }
    with open(file, "r") as inf:
        lines = inf.read().splitlines()

    long_pep_counter = 0
    pep_codes = []
    labels = []
    pep_seq = []
    for pep in lines:
        pep, label = pep.split(",")
        labels.append(int(label))
        input_seq = " ".join(pep)
        input_seq = re.sub(r"[UZOB]", "X", input_seq)
        pep_seq.append(input_seq)
        if not len(pep) > max_len:
            current_pep = []
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(torch.tensor(current_pep))
        else:
            long_pep_counter += 1
    print("length > 33:", long_pep_counter)
    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)

    return data, torch.tensor(labels), pep_seq


def get_prelabel(data_iter, net, seq2vec):
    prelabel, relabel = [], []
    net.eval()
    with torch.no_grad():
        for x, y, z in data_iter:
            x, y = x.to(device), y.to(device)
            seq_to_emb = {item['sequence']: item for item in seq2vec}
            for i in range(len(z)):
                if i == 0:
                    if z[0].replace(' ', '') in seq_to_emb:
                        vec = seq_to_emb[z[0].replace(' ', '')]['embedding'].unsqueeze(0).to(device)
                else:
                    if z[i].replace(' ', '') in seq_to_emb:
                        vec = torch.cat((vec, seq_to_emb[z[i].replace(' ', '')]['embedding'].unsqueeze(0).to(device)), dim=0)
            outputs = net.trainModel(x, vec)
            prelabel.append(outputs.argmax(dim=1).cpu().numpy())
            relabel.append(y.cpu().numpy())
    return prelabel, relabel


def caculate_metric(pred_y, labels, pred_prob):

    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if int(labels[index]) == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    ACC = float(tp + tn) / test_num

    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (
            np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        )

    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    labels = list(map(int, labels))
    pred_prob = list(map(float, pred_prob))
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)
    AUC = auc(fpr, tpr)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(
        labels, pred_prob, pos_label=1
    )
    AP = average_precision_score(
        labels, pred_prob, average="macro", pos_label=1, sample_weight=None
    )

    metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC])

    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return metric, roc_data, prc_data


class MyDataSet(Data.Dataset):
    def __init__(self, data, label, seq):
        self.data = data
        self.label = label
        self.seq = seq

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.seq[idx]


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1, bias=False)

    def forward(self, H, mask=None):
        scores = self.attn(H).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        H_att = torch.sum(H * weights.unsqueeze(-1), dim=1)
        return H_att, weights


class PhoSARteModel(nn.Module):
    def __init__(self, vocab_size=22, max_len=33, num_heads=8, num_layers=1, pretained_dims=1024):
        super().__init__()
        self.hidden_dim = 128
        self.emb_dim = 512
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pretained_dims = pretained_dims

        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        self.pos_enc = nn.Parameter(torch.zeros(1, self.max_len, self.emb_dim))
        nn.init.normal_(self.pos_enc, std=0.02)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=self.num_heads,
            dim_feedforward=1024,
            dropout=0.3,
            activation='gelu',
            batch_first=True,
            norm_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.num_layers,
        )

        self.gru = nn.GRU(
            self.emb_dim,
            self.hidden_dim,
            num_layers=self.num_layers+1,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        self.attn_pool = AttentionPooling(self.hidden_dim * 2)

        self.gruplm = nn.GRU(
            self.pretained_dims,
            self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 2),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0)

    def forward(self, x):

        padding_mask = (x == 0)
        
        x_emb = self.embedding(x)
        x_emb = x_emb + self.pos_enc[:, :x.size(1), :]
        
        trans_output = self.transformer_encoder(
            x_emb, 
            src_key_padding_mask=padding_mask
        )
        
        gru_output, _ = self.gru(trans_output)
        
        h_att, _ = self.attn_pool(gru_output, mask=padding_mask)
        
        return h_att

    def trainModel(self, x, pep):
        seq_features = self.forward(x)

        gru_pep_output, _ = self.gruplm(pep)
        pep_features, _ = self.attn_pool(gru_pep_output)
        
        combined_features = torch.cat((seq_features, pep_features), dim=1)
        
        logits = self.fusion_classifier(combined_features)
        
        return logits



class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive



def collate(batch):
    seq1_ls = []
    seq2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    pep1_ls = []
    pep2_ls = []
    batch_size = len(batch)
    for i in range(int(batch_size / 2)):
        seq1, label1, pep_seq1 = batch[i][0], batch[i][1], batch[i][2]
        seq2, label2, pep_seq2 = (
            batch[i + int(batch_size / 2)][0],
            batch[i + int(batch_size / 2)][1],
            batch[i + int(batch_size / 2)][2],
        )
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        pep1_ls.append(pep_seq1)
        pep2_ls.append(pep_seq2)
        label = label1 ^ label2
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
    seq1 = torch.cat(seq1_ls) 
    seq2 = torch.cat(seq2_ls) 
    label = torch.cat(label_ls)
    label1 = torch.cat(label1_ls) 
    label2 = torch.cat(label2_ls) 
    return seq1, seq2, label, label1, label2, pep1_ls, pep2_ls


def evaluate_accuracy(data_iter, net, seq2vec):
    acc_sum, n = 0.0, 0
    net.eval()
    with torch.no_grad():
        for x, y, z in data_iter:
            x, y = x.to(device), y.to(device)
            seq_to_emb = {item['sequence']: item for item in seq2vec}
            for i in range(len(z)):
                if i == 0:
                    if z[0].replace(' ', '') in seq_to_emb:
                        vec = seq_to_emb[z[0].replace(' ', '')]['embedding'].unsqueeze(0).to(device)
                else:
                    if z[i].replace(' ', '') in seq_to_emb:
                       vec = torch.cat((vec, seq_to_emb[z[i].replace(' ', '')]['embedding'].unsqueeze(0).to(device)), dim=0)
            outputs = net.trainModel(x, vec)

            acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

num_heads = [8, 32, 16]
num_layers = [1, 2, 4]
data_list = ['A549', 'VeroE6', 'Combined']
pretrained_list = ['prot_t5_xl_bfd', 'prot_t5_xl_uniref50', 'prot_t5_xxl_bfd', 'prot_t5_xxl_uniref50', 'prot_albert', 'prot_bert', 'prot_bert_bfd', 'prot_xlnet']
k_folds = [1, 2, 3, 4, 5]

for num_head in num_heads:
    for num_layer in num_layers:
        for data_name in data_list:
            for pretrained_type in pretrained_list:
                for kf in k_folds:
                    # Load data
                    train_data, train_label, train_seq = genData(
                        "./data/{}/{}_train_fold_{}.csv".format(data_name, data_name, kf), 33
                    )
                    test_data, test_label, test_seq = genData(
                        "./data/{}/{}_val_fold_{}.csv".format(data_name, data_name, kf), 33
                    )
                    print(train_data.shape, train_label.shape)
                    print(test_data.shape, test_label.shape)
                    train_dataset = MyDataSet(train_data, train_label, train_seq)
                    test_dataset = MyDataSet(test_data, test_label, test_seq)
                    batch_size = 128
                    train_iter_cont = torch.utils.data.DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
                    )
                    test_iter = torch.utils.data.DataLoader(
                        test_dataset, batch_size=batch_size, shuffle=False
                    )
                    seq2vec_train = torch.load("./embeddings/{}/{}_Train_{}.pt".format(data_name, data_name, pretrained_type))
                    print("Loaded data for:", data_name)
                    print("Training with:", pretrained_type, num_layer, num_head)
                    print("Training fold:", kf)
                    # Define model
                    if pretrained_type == 'prot_albert':
                        pretained_dim = 4096
                    else:
                        pretained_dim = 1024
                    net = PhoSARteModel(num_heads=num_head, num_layers=num_layer, pretained_dims = pretained_dim).to(device)
                    lr = 0.0001

                    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-5)
                    criterion = ContrastiveLoss()

                    criterion_model = nn.CrossEntropyLoss(reduction="sum")
                    best_mcc = 0
                    best_metrics = None
                    EPOCH = 50
                    save_path = "./kfcv_models/{}/{}_{}_{}_{}.pt".format(data_name, pretrained_type, num_layer, num_head, kf)

                    for epoch in range(EPOCH):
                        loss_ls = []
                        loss1_ls = []
                        loss2_3_ls = []

                        t0 = time.time()
                        net.train()

                        for seq1, seq2, label, label1, label2, pep1, pep2 in train_iter_cont:

                            for i in range(len(pep1)):

                                if i == 0:
                                    seq_to_emb = {item['sequence']: item for item in seq2vec_train}
                                    if pep1[0].replace(' ', '') in seq_to_emb:
                                        pep1_2vec = seq_to_emb[pep1[0].replace(' ', '')]['embedding'].unsqueeze(0).to(device)
                                    if pep2[0].replace(' ', '') in seq_to_emb:
                                        pep2_2vec = seq_to_emb[pep2[0].replace(' ', '')]['embedding'].unsqueeze(0).to(device)
                                else:
                                    seq_to_emb = {item['sequence']: item for item in seq2vec_train}
                                    if pep1[i].replace(' ', '') in seq_to_emb:
                                        pep1_2vec = torch.cat((pep1_2vec, seq_to_emb[pep1[i].replace(' ', '')]['embedding'].unsqueeze(0).to(device)), dim=0)
                                    if pep2[i].replace(' ', '') in seq_to_emb:
                                        pep2_2vec = torch.cat((pep2_2vec, seq_to_emb[pep2[i].replace(' ', '')]['embedding'].unsqueeze(0).to(device)), dim=0)

                            seq1_cuda = seq1.to(device)
                            seq2_cuda = seq2.to(device)
                            label_cuda = label.to(device)
                            label1_cuda = label1.to(device)
                            label2_cuda = label2.to(device)
                            pep1_2vec_cuda = pep1_2vec.to(device)
                            pep2_2vec_cuda = pep2_2vec.to(device)
                            output1 = net(seq1_cuda)
                            output2 = net(seq2_cuda)
                            output3 = net.trainModel(
                                seq1_cuda, pep1_2vec_cuda
                            )
                            output4 = net.trainModel(
                                seq2_cuda, pep2_2vec_cuda
                            )
                            seq1_cuda, seq2_cuda = None, None
                            pep1_2vec_cuda, pep2_2vec_cuda = None, None
                            loss1 = criterion(
                                output1, output2, label_cuda
                            )
                            loss2 = criterion_model(
                                output3, label1_cuda
                            ) 
                            loss3 = criterion_model(
                                output4, label2_cuda
                            )
                            loss = loss1 + loss2 + loss3
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            loss_ls.append(loss.item())
                            loss1_ls.append(loss1.item())
                            loss2_3_ls.append((loss2 + loss3).item())
                            label_cuda, label1_cuda, label2_cuda = None, None, None
                            output1, output2, output3, output4 = None, None, None, None

                            del (
                                seq1_cuda,
                                seq2_cuda,
                                label_cuda,
                                label1_cuda,
                                label2_cuda,
                                pep1_2vec_cuda,
                                pep2_2vec_cuda,
                                output1,
                                output2,
                                output3,
                                output4,
                            )
                            torch.cuda.empty_cache()

                        net.eval()
                        with torch.no_grad():
                            test_acc = evaluate_accuracy(test_iter, net, seq2vec_train)
                            A, B = get_prelabel(test_iter, net, seq2vec_train)
                            A = [np.concatenate(A)]
                            B = [np.concatenate(B)]
                            A = np.array(A)
                            B = np.array(B)
                            A = A.reshape(-1, 1)
                            B = B.reshape(-1, 1)

                            df1 = pd.DataFrame(A, columns=["prelabel"])
                            df2 = pd.DataFrame(B, columns=["realabel"])
                            df4 = pd.concat([df1, df2], axis=1)

                            acc_sum, n = 0.0, 0
                            outputs = []
                            for x, y, z in test_iter:
                                x, y = x.to(device), y.to(device)
                                seq_to_emb = {item['sequence']: item for item in seq2vec_train}
                                for i in range(len(z)):
                                    if i == 0:
                                        if z[0].replace(' ', '') in seq_to_emb:
                                            vec = seq_to_emb[z[0].replace(' ', '')]['embedding'].unsqueeze(0).to(device)
                                    else:
                                        if z[i].replace(' ', '') in seq_to_emb:
                                            vec = torch.cat((vec, seq_to_emb[z[i].replace(' ', '')]['embedding'].unsqueeze(0).to(device)), dim=0)
                                output = torch.softmax(net.trainModel(x, vec), dim=1)
                                outputs.append(output)
                                del x, y, z, vec, output
                                torch.cuda.empty_cache()
                            outputs = torch.cat(outputs, dim=0)
                            pre_pro = outputs[:, 1]
                            pre_pro = np.array(pre_pro.cpu().detach().numpy())
                            pre_pro = pre_pro.reshape(-1)
                            df3 = pd.DataFrame(pre_pro, columns=["pre_pro"])
                            df5 = pd.concat([df4, df3], axis=1)
                            real1 = df5["realabel"]
                            pre1 = df5["prelabel"]
                            pred_pro1 = df5["pre_pro"]
                            metric1, roc_data1, prc_data1 = caculate_metric(pre1, real1, pred_pro1)

                        torch.cuda.empty_cache()
                        results = f"epoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}\n"
                        results += f'\ttest_acc: {colored(test_acc, "red")}, time: {time.time() - t0:.2f}'
                        print(results)

                        if metric1[6] > best_mcc:
                            best_mcc = metric1[6]
                            best_metrics = metric1
                            torch.save(
                            {"best_mcc": best_mcc, "metric": metric1, "model": net.state_dict()},
                            save_path,
                            )
                            print(f"best_MCC: {best_mcc}, metric: {metric1}")
                            with open('./kfcv_results/{}/{}_{}_{}_{}_best_metrics.txt'.format(data_name, pretrained_type, num_head, num_layer, kf), 'w') as f:
                                f.write(f'Best MCC: {best_mcc}\n')
                                f.write(f'Accuracy: {best_metrics[0]}\n')
                                f.write(f'Precision: {best_metrics[1]}\n')
                                f.write(f'Sensitivity: {best_metrics[2]}\n')
                                f.write(f'Specificity: {best_metrics[3]}\n')
                                f.write(f'F1-score: {best_metrics[4]}\n')
                                f.write(f'AUC: {best_metrics[5]}\n')
                                f.write(f'MCC: {best_metrics[6]}\n')
                            output_df = pd.DataFrame({
                                'Sample': test_seq,
                                'Real_Label': real1,
                                'Predicted_Probability': pred_pro1,
                                'Predicted_Class': pre1.astype(int)
                            })
                            output_df.to_csv('./kfcv_results/{}/{}_{}_{}_{}.csv'.format(data_name, pretrained_type, num_head, num_layer, kf), index=False)
                        print("Current best_mcc:", best_mcc)
                print(f"best_mcc: {best_mcc},metric:{best_metrics}")
