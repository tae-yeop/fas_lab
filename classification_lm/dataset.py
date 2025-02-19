import torch
from torch.utils.data import Dataset, DataLoader
import os
import json

class LandmarkDataset(Dataset):
    def __init__(self, json_path):
        if not os.path.exists(json_path):
            self.data = None

        with open(json_path, "r") as f:
            self.data = json.load(f)
 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        landmarks = self.data[idx]["landmarks"]
        label = self.data[idx]["photo_label"]
        landmarks = torch.tensor(landmarks)
        label = torch.tensor(label)
        landmarks = preprocess_landmarks(landmarks)

        return landmarks, label



def get_dataloader(json_path, batch_size):
    dataset = LandmarkDataset(json_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader