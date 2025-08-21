import torch
import torch.nn as nn
import math

def pad_array(lst, size=30):
    arr = np.zeros(size, dtype=np.float32)
    length = min(len(lst), size)
    arr[:length] = lst[:length]
    return arr


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

        out = self.transformer_encoder(x) # => 

        # mean-pool across seq_len => (batch,d_model)
        out = out.mean(dim=1)  # (batch, d_model)

        # final linear => (batch,2)
        logits = self.fc(out)
        return logits


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EyeFeatureDataset(Dataset):
    """
    X: shape=(N,30,8) float
    y: shape=(N,) int
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        feats = self.X[idx]  # shape=(30,8)
        label = self.y[idx]
        # convert to torch
        feats_t = torch.tensor(feats, dtype=torch.float32)  # (30,8)
        return feats_t, torch.tensor(label, dtype=torch.long)

import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

def train_transformer(model, train_loader, val_loader, epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        # train
        model.train()
        train_loss_sum=0
        train_correct=0
        train_total=0
        for feats, labels in train_loader:
            feats  = feats.to(device)   # (batch,30,8)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs= model(feats)      # => (batch,2)
            loss   = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()*len(labels)
            preds = outputs.argmax(dim=1)
            train_correct+= (preds==labels).sum().item()
            train_total  += len(labels)
        train_loss= train_loss_sum/train_total
        train_acc = train_correct/train_total

        # val
        model.eval()
        val_loss_sum=0
        val_correct=0
        val_total=0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                labels= labels.to(device)
                outputs= model(feats)
                loss  = criterion(outputs, labels)
                val_loss_sum+= loss.item()*len(labels)
                preds= outputs.argmax(dim=1)
                val_correct+= (preds==labels).sum().item()
                val_total+= len(labels)
        val_loss= val_loss_sum/val_total
        val_acc= val_correct/val_total

        print(f"Epoch[{epoch}/{epochs}] "
              f"TrainLoss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"ValLoss={val_loss:.4f} Acc={val_acc:.4f}")

    return model



if __name__=="__main__":
    # import random
    # random.seed(42)
    # data_list = []
    # for i in range(1000):
    #     is_attack = 1 if (i<200) else 0  # e.g. 200개의 attack, 800개의 real
    #     feats = np.random.randn(30,8).astype(np.float32)  # dummy
    #     data_list.append({'features':feats, 'label': is_attack})
    
    npy_path = "/purestorage/AILAB/AI_1/tyk/3_CUProjects/fas_lab/mask_eye_fas/output_npy/final_data.npy"
    arr = np.load(npy_path, allow_pickle=True)
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

        # feature_vec = np.concatenate([left_merged, right_merged], axis=0) # shape=(120,)
        
        # X.append(feature_vec)
        # y.append(label)
        feature_vec = np.concatenate([left_merged, right_merged], axis=0)
        feature_mat = feature_vec.reshape(30, 8)  # (30,8)

        X.append(feature_mat)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)


    test_size=0.2
    random_state=42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )


    # 3) Dataloader
    from torch.utils.data import DataLoader
    train_ds = EyeFeatureDataset(X_train, y_train)
    val_ds   = EyeFeatureDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)


    # 4) model
    model = TimeSeriesTransformer()
    
    # 5) train
    model = train_transformer(model, train_loader, val_loader, epochs=5, lr=1e-3)



    model.eval()
    test_correct=0
    test_total=0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats = feats.to(model.pos_embedding.device)
            labels= labels.to(model.pos_embedding.device)
            out   = model(feats)
            preds = out.argmax(dim=1)
            test_correct+= (preds==labels).sum().item()
            test_total+= len(labels)
    test_acc= test_correct/test_total
    print("Test Acc=", test_acc)