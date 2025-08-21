import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

import argparse

# ===========================
# Utility functions
# ===========================
def pad_array(lst, size=30):
    arr = np.zeros(size, dtype=np.float32)
    length = min(len(lst), size)
    arr[:length] = lst[:length]
    return arr

# ===========================
# Dataset
# ===========================
class EyeFeatureDataset(Dataset):
    def __init__(self, X, y, channel_first=False):
        self.X = X
        self.y = y
        self.channel_first = channel_first # 1DCNN의 경우 [Channel, Length] 순서 필요

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feats = self.X[idx]  # shape=(30,8)
        label = self.y[idx]
        
        feats_t = torch.tensor(feats, dtype=torch.float32) # [30, 8]
        if self.channel_first:
            feats_t = feats_t.permute(1, 0) # now [8, 30]

        return feats_t, torch.tensor(label, dtype=torch.long)


def prepare_dataloader(data_path, model_type, test_size=0.2, random_state=42, batch_size=32):
    arr = np.load(data_path, allow_pickle=True)
    data_list = arr.tolist()

    X = []
    y = []

    for item in data_list:
        left_feats  = item["eye_left_feats"] # dict
        right_feats = item["eye_right_feats"]
        label       = item["label"]

        edge_l = left_feats.get("edge", [])
        shadow_l = left_feats.get("shadow", [])
        refl_l = left_feats.get("reflection", [])
        freq_l = left_feats.get("freq", [])

        edge_l_pad = pad_array(edge_l, 30)
        shadow_l_pad = pad_array(shadow_l, 30)
        refl_l_pad = pad_array(refl_l, 30)
        freq_l_pad = pad_array(freq_l, 30)

        left_merged = np.concatenate(
            [edge_l_pad, shadow_l_pad, refl_l_pad, freq_l_pad], axis=0
        )

        edge_r = right_feats.get("edge", [])
        shadow_r = right_feats.get("shadow", [])
        refl_r = right_feats.get("reflection", [])
        freq_r = right_feats.get("freq", [])
        
        edge_r_pad = pad_array(edge_r, 30)
        shadow_r_pad = pad_array(shadow_r, 30)
        refl_r_pad = pad_array(refl_r, 30)
        freq_r_pad = pad_array(freq_r, 30)

        right_merged= np.concatenate(
            [edge_r_pad, shadow_r_pad, refl_r_pad, freq_r_pad], axis=0
        )

        feature_vec = np.concatenate([left_merged, right_merged], axis=0)
        feature_mat = feature_vec.reshape(30, 8)  # (30,8)

        X.append(feature_mat)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    if model_type == '1dcnn':
        channel_first = True
    else:
        channel_first = False

    train_ds = EyeFeatureDataset(X_train, y_train, channel_first)
    val_ds   = EyeFeatureDataset(X_test, y_test, channel_first)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    return train_loader, val_loader

# ===========================
# Model
# ===========================
class EyeFeature1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(32)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(32*15, 2)

    def forward(self, x):
        """
        x : [b, 8, 30]
        """
        x = self.conv1(x)   # => [B,16,30]
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)   # => [B,32,30]
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.pool(x)    # => [B,32,15]
        
        x = x.view(x.size(0), -1)  # => [B,32*15]
        logits = self.fc(x)        # => [B,2]
        return logits


class TimeSeriesTransformer(nn.Module):
    def __init__(self, 
                 input_dim=8,
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=256,
                 dropout=0.1,
                 num_classes=2,
                 seq_len=30):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  #
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x : [b, 30, 8]
        """
        batch_size, seq_len, input_dim = x.shape
        
        x = self.input_proj(x) # => [B, 30, d_model]

        x = x + self.pos_embedding[:, :seq_len, :] #  => [1, 30, d_model] + [B, 30, d_model]

        out = self.transformer_encoder(x) # => [B, 30, d_model]

        out = out.mean(dim=1)  # => [B, d_model]

        logits = self.fc(out) # => [B, 2]
        return logits



# ===========================
# train
# ===========================

def train(model, train_loader, val_loader, device, epochs=5, lr=1e-3):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        train_loss_sum = 0
        train_correct  = 0
        train_total    = 0

        for feats, labels in train_loader:
            feats  = feats.to(device)   # (batch, 8, 30)
            labels = labels.to(device)  # (batch,)
            
            optimizer.zero_grad()
            outputs = model(feats)      # => (batch,2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            train_correct  += (preds==labels).sum().item()
            train_total    += len(labels)

        train_loss = train_loss_sum/train_total
        train_acc  = train_correct/train_total

        model.eval()
        val_loss_sum=0
        val_correct=0
        val_total=0
        
        with torch.no_grad():
            for feats, labels in val_loader:
                feats  = feats.to(device)
                labels = labels.to(device)
                outputs= model(feats)
                loss   = criterion(outputs, labels)
                val_loss_sum += loss.item()*len(labels)
                preds = outputs.argmax(dim=1)
                val_correct += (preds==labels).sum().item()
                val_total   += len(labels)
        val_loss = val_loss_sum/val_total
        val_acc  = val_correct/val_total

        print(f"Epoch[{epoch}/{epochs}] "
              f"TrainLoss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"ValLoss={val_loss:.4f} Acc={val_acc:.4f}")

    return model
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='1dcnn') # 1dcnn, transformer
    parser.add_argument('--data_path', type=str, default='/purestorage/AILAB/AI_1/tyk/3_CUProjects/fas_lab/mask_eye_fas/output_npy/final_data.npy')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = prepare_dataloader(args.data_path, args.model_type, batch_size=args.batch_size)

    if args.model_type == '1dcnn':
        model = EyeFeature1DCNN()
    elif args.model_type == 'transformer':
        model = TimeSeriesTransformer()
    else:
        pass

    model.to(device)

    model = train(model, train_loader, val_loader, device=device, epochs=args.epochs)

    # Test : test 데이터 구하기
    model.eval()
    test_correct=0
    test_total=0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats = feats.to(device)
            labels= labels.to(device)
            out   = model(feats)
            preds = out.argmax(dim=1)
            test_correct+= (preds==labels).sum().item()
            test_total+= len(labels)
    test_acc= test_correct/test_total
    print("Test Acc=", test_acc)
