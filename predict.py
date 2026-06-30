import os
import re
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
import torch.backends.cudnn as cudnn
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    auc,
)
from transformers import (
    T5Tokenizer,
    T5EncoderModel,
)
import random
import gc
import warnings

warnings.filterwarnings("ignore")
cudnn.benchmark = True
SEED = 6996
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
gc.collect()
torch.cuda.empty_cache()


# ──────────────────────────────────────────────
# Embedding extraction 
# ──────────────────────────────────────────────

# Default local paths for each PLM (edit as needed)
PLM_PATHS = {
    "prot_t5_xl_bfd": "ProtTrans_models/models--Rostlab--prot_t5_xl_bfd/snapshots/7ae1d5c1d148d6c65c7e294cc72807e5b454fdb7",
    "prot_t5_xxl_uniref50": "ProtTrans_models/models--Rostlab--prot_t5_xxl_uniref50/snapshots/31a40d7b55caf68d7a8a8dfd913b779b99dc09a9",
}

# Mapping from PLM name to embedding dimension
PLM_DIMS = {
    "prot_t5_xl_bfd": 1024,
    "prot_t5_xxl_uniref50": 1024,
}

# Model-specific defaults
MODEL_CONFIGS = {
    "A549": {
        "num_head": 16,
        "num_layer": 1,
        "pretrained_type": "prot_t5_xxl_uniref50",
        "model_file": "PhoSARte_A549.pt",
    },
    "VeroE6": {
        "num_head": 32,
        "num_layer": 1,
        "pretrained_type": "prot_t5_xl_bfd",
        "model_file": "PhoSARte_VeroE6.pt",
    },
    "Generic": {
        "num_head": 8,
        "num_layer": 1,
        "pretrained_type": "prot_t5_xxl_uniref50",
        "model_file": "PhoSARte_Generic.pt",
    },
}


def load_plm(plm_name, device):
    """Load tokenizer and PLM model based on name."""
    plm_path = PLM_PATHS[plm_name]
    if plm_name.startswith("prot_t5"):
        tokenizer = T5Tokenizer.from_pretrained(plm_path, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(plm_path).to(device).eval()
    else:
        raise ValueError(f"Unsupported PLM: {plm_name}")
    return tokenizer, plm_model


def encode_sequence(sequence, tokenizer, plm_model, plm_name, device, max_len=33):
    """Extract embedding for a single sequence using the loaded PLM."""
    seq_spaced = " ".join(list(sequence)).upper()
    seq_spaced = re.sub(r"[UZOB]", "X", seq_spaced)

    inputs = tokenizer(
        seq_spaced,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if plm_name.startswith("prot_t5"):
            if "decoder_input_ids" in inputs:
                del inputs["decoder_input_ids"]
            outputs = plm_model(**inputs)
            embeddings = outputs.last_hidden_state.squeeze(0)

    embeddings = embeddings[:max_len]
    return embeddings.cpu()


def extract_embeddings_for_sequences(sequences, tokenizer, plm_model, plm_name, device, max_len=33):
    """Extract embeddings for a list of sequences. Returns list of dicts matching seq2vec format."""
    embeddings_data = []
    for seq in sequences:
        raw_seq = seq.replace(" ", "")
        emb = encode_sequence(raw_seq, tokenizer, plm_model, plm_name, device, max_len)
        embeddings_data.append({"sequence": raw_seq, "embedding": emb})
    return embeddings_data


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def genData(file, max_len):
    aa_dict = {
        "A": 1, "R": 2, "N": 3, "D": 4, "C": 5, "Q": 6, "E": 7,
        "G": 8, "H": 9, "I": 10, "L": 11, "K": 12, "M": 13, "F": 14,
        "P": 15, "S": 16, "T": 17, "W": 18, "Y": 19, "V": 20, "X": 21,
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


class MyDataSet(Data.Dataset):
    def __init__(self, data, label, seq):
        self.data = data
        self.label = label
        self.seq = seq

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.seq[idx]


# ──────────────────────────────────────────────
# Model architecture
# ──────────────────────────────────────────────

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
    """
    Improved phosphorylation prediction model combining:
    - Embedding layer with positional encoding
    - Transformer encoder for capturing long-range dependencies
    - Bidirectional GRU for sequential information
    - Attention pooling for feature aggregation
    - Fusion classifier combining sequence and PLM features
    """
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
            d_model=self.emb_dim, nhead=self.num_heads, dim_feedforward=1024,
            dropout=0.3, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.num_layers,
        )

        self.gru = nn.GRU(
            self.emb_dim, self.hidden_dim, num_layers=self.num_layers + 1,
            bidirectional=True, batch_first=True, dropout=0.3,
        )

        self.attn_pool = AttentionPooling(self.hidden_dim * 2)

        self.gruplm = nn.GRU(
            self.pretained_dims, self.hidden_dim, num_layers=self.num_layers,
            bidirectional=True, batch_first=True,
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.3),
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
        trans_output = self.transformer_encoder(x_emb, src_key_padding_mask=padding_mask)
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


# ──────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────

def get_prelabel(data_iter, net, seq2vec, device):
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


def evaluate_accuracy(data_iter, net, seq2vec, device):
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


def caculate_metric(pred_y, labels, pred_prob):
    test_num = len(labels)
    tp = fp = tn = fn = 0
    for index in range(test_num):
        if int(labels[index]) == 1:
            if labels[index] == pred_y[index]:
                tp += 1
            else:
                fn += 1
        else:
            if labels[index] == pred_y[index]:
                tn += 1
            else:
                fp += 1

    ACC = float(tp + tn) / test_num
    Precision = float(tp) / (tp + fp) if (tp + fp) else 0
    Recall = Sensitivity = float(tp) / (tp + fn) if (tp + fn) else 0
    Specificity = float(tn) / (tn + fp) if (tn + fp) else 0
    MCC = (float(tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
           if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0)
    F1 = 2 * Recall * Precision / (Recall + Precision) if (Recall + Precision) else 0

    labels = list(map(int, labels))
    pred_prob = list(map(float, pred_prob))
    fpr, tpr, _ = roc_curve(labels, pred_prob, pos_label=1)
    AUC = auc(fpr, tpr)
    precision_arr, recall_arr, _ = precision_recall_curve(labels, pred_prob, pos_label=1)
    AP = average_precision_score(labels, pred_prob, average="macro", pos_label=1)

    metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC])
    roc_data = [fpr, tpr, AUC]
    prc_data = [recall_arr, precision_arr, AP]
    return metric, roc_data, prc_data


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PhoSARte end-to-end prediction (embedding extraction + inference)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the input CSV file. Each line: <sequence>,<label>",
    )
    parser.add_argument(
        "--model_type", type=str, default="Generic",
        choices=["A549", "VeroE6", "Generic"],
        help="Which PhoSARte model to use (default: Generic)",
    )
    parser.add_argument(
        "--model_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "final_models"),
        help="Directory containing PhoSARte .pt model files (default: final_models)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for computation (default: cuda:0 or cpu)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size for inference (default: 128)",
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Directory to save prediction results (default: results)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    cfg = MODEL_CONFIGS[args.model_type]

    plm_name = cfg["pretrained_type"]
    pretained_dim = PLM_DIMS[plm_name]

    # 1. Load input data
    print(f"[1/4] Loading input data from {args.input} ...")
    test_data, test_label, test_seq = genData(args.input, 33)
    print(f"  Data shape: {test_data.shape}, Labels shape: {test_label.shape}")

    # 2. Load PLM and extract embeddings on-the-fly
    print(f"[2/4] Loading PLM '{plm_name}' and extracting embeddings ...")
    tokenizer, plm_model = load_plm(plm_name, device)
    seq2vec = extract_embeddings_for_sequences(
        test_seq, tokenizer, plm_model, plm_name, device, max_len=33
    )
    print(f"  Extracted embeddings for {len(seq2vec)} sequences")
    # Free PLM memory
    del plm_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Load PhoSARte model
    model_path = os.path.join(args.model_dir, cfg["model_file"])
    print(f"[3/4] Loading PhoSARte model from {model_path} ...")
    net = PhoSARteModel(
        num_heads=cfg["num_head"],
        num_layers=cfg["num_layer"],
        pretained_dims=pretained_dim,
    ).to(device)
    net.load_state_dict(
        torch.load(model_path, map_location=device)["model"]
    )
    net.eval()

    # 4. Run prediction
    print("[4/4] Running prediction ...")
    test_dataset = MyDataSet(test_data, test_label, test_seq)
    test_iter = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        # Get predicted labels
        A, B = get_prelabel(test_iter, net, seq2vec, device)
        A = np.concatenate(A).reshape(-1, 1)
        B = np.concatenate(B).reshape(-1, 1)
        df1 = pd.DataFrame(A, columns=["prelabel"])
        df2 = pd.DataFrame(B, columns=["realabel"])
        df4 = pd.concat([df1, df2], axis=1)

        # Get predicted probabilities
        outputs_all = []
        for x, y, z in test_iter:
            x, y = x.to(device), y.to(device)
            seq_to_emb = {item['sequence']: item for item in seq2vec}
            for i in range(len(z)):
                if i == 0:
                    if z[0].replace(' ', '') in seq_to_emb:
                        vec = seq_to_emb[z[0].replace(' ', '')]['embedding'].unsqueeze(0).to(device)
                else:
                    if z[i].replace(' ', '') in seq_to_emb:
                        vec = torch.cat((vec, seq_to_emb[z[i].replace(' ', '')]['embedding'].unsqueeze(0).to(device)), dim=0)
            output = torch.softmax(net.trainModel(x, vec), dim=1)
            outputs_all.append(output)

        outputs_all = torch.cat(outputs_all, dim=0)
        pre_pro = outputs_all[:, 1].cpu().numpy().reshape(-1)
        df3 = pd.DataFrame(pre_pro, columns=["pre_pro"])
        df5 = pd.concat([df4, df3], axis=1)

        real1 = df5["realabel"]
        pre1 = df5["prelabel"]
        pred_pro1 = df5["pre_pro"]

    # Compute metrics
    metric1, roc_data1, prc_data1 = caculate_metric(pre1, real1, pred_pro1)
    ACC, Precision, Sensitivity, Specificity, F1, AUC_val, MCC = metric1

    pred_cls = pre_pro > 0.5
    tn, fp, fn, tp = confusion_matrix(real1, pred_cls).ravel()

    mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-6)
    sn = float(tp) / (tp + fn + 1e-6)
    sp = float(tn) / (tn + fp + 1e-6)
    acc = float(tp + tn) / (tn + fp + fn + tp + 1e-6)
    fpr, tpr, _ = roc_curve(real1, pre_pro, pos_label=1)
    auc_ = auc(fpr, tpr)
    precision_arr, recall_arr, _ = precision_recall_curve(real1, pre_pro, pos_label=1)
    aupr = auc(recall_arr, precision_arr)

    print("\n" + "=" * 60)
    print(f"  Model: PhoSARte_{args.model_type}  |  PLM: {plm_name}")
    print("=" * 60)
    print(f"  ACC:         {ACC:.4f}")
    print(f"  Precision:   {Precision:.4f}")
    print(f"  Sensitivity: {Sensitivity:.4f}")
    print(f"  Specificity: {Specificity:.4f}")
    print(f"  F1:          {F1:.4f}")
    print(f"  AUC:         {AUC_val:.4f}")
    print(f"  MCC:         {MCC:.4f}")
    print(f"  AUPR:        {aupr:.4f}")
    print("=" * 60)
    print(f"  MCC={mcc:.4f}  SN={sn:.4f}  SP={sp:.4f}  ACC={acc:.4f}  AUC={auc_:.4f}  AUPR={aupr:.4f}")
    print(confusion_matrix(real1, pred_cls))

    # Optionally save results
    if args.output:
        output_df = pd.DataFrame({
            "Sample": [seq.replace(" ", "") for seq in test_seq],
            "Real_Label": real1,
            "Predicted_Probability": pre_pro,
            "Predicted_Class": pred_cls.astype(int),
        })
        output_df.to_csv(os.path.join(args.output, f"PhoSARte_{args.model_type}_Results.csv"), index=False)
        print(f"\nResults saved to {os.path.join(args.output, f'PhoSARte_{args.model_type}_Results.csv')}")


if __name__ == "__main__":
    main()
