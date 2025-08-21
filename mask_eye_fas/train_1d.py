import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

def pad_array(lst, size=30):
    arr = np.zeros(size, dtype=np.float32)
    length = min(len(lst), size)
    arr[:length] = lst[:length]
    return arr


class EyeFeatureDataset(Dataset):
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        feats = self.x_list[idx]
        label = self.y_list[idx]
        
        # to torch.Tensor
        feats_t = torch.tensor(feats, dtype=torch.float32)  # (30,8)
        # 1D CNN expects (channels, length) => (8,30)
        feats_t = feats_t.permute(1, 0)  # now shape=(8,30)

        return feats_t, torch.tensor(label, dtype=torch.long)





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
        x = self.conv1(x)   # => (batch,16,30)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)   # => (batch,32,30)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.pool(x)    # => (batch,32,15)
        
        x = x.view(x.size(0), -1)  # => (batch,32*15)
        logits = self.fc(x)        # => (batch,2)
        return logits


def train_1dcnn(model, train_loader, val_loader, epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs+1):
        # TRAIN
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
        
        # VAL
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
    model = EyeFeature1DCNN()
    
    # 5) train
    model = train_1dcnn(model, train_loader, val_loader, epochs=5, lr=1e-3)
    
    # 6) test
    # (여기서는 val_loader로 대체)
    model.eval()
    test_correct=0
    test_total=0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats= feats.cuda() if torch.cuda.is_available() else feats
            labels= labels.cuda() if torch.cuda.is_available() else labels
            out= model(feats)
            preds= out.argmax(dim=1)
            test_correct += (preds==labels).sum().item()
            test_total   += len(labels)
    test_acc= test_correct/test_total
    print("Test accuracy=", test_acc)
