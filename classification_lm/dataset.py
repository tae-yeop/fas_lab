import torch
from torch.utils.data import Dataset, DataLoader
import torch_geometric.loader as pyg
import os
import json


def preprocess_landmarks(landmarks):

    lm_list = [
            162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
            296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
            380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87
        ]

    nose_idx = lm_list.index(4)
    nose_x, nose_y = landmarks[nose_idx][0], landmarks[nose_idx][1]

    tensor = torch.tensor(landmarks)

    landmarks_centered = tensor - torch.tensor([nose_x, nose_y])

    return landmarks_centered


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

def get_pyg_dataloader():
    # 실제로는 모든 랜드마크 연결
    face_adjacency = [
        (0,1), (1,2), (2,3), (3,4), # ...
        (36,37), (37,38), (38,39), # ...
    ]

    batch_landmarks = random_face_landmarks(batch_size=4, num_frames=10)
    labels = torch.tensor([0,1,0,1], dtype=torch.float32)

    data_list = []
    for i in range(batch_landmarks.shape[0]):
        lm = batch_landmarks[i]  # [num_frames, 68, 2]
        graph_data = build_st_graph(lm, face_adjacency, temporal=True)
        graph_data.y = labels[i].unsqueeze(0)  # PyG convention (shape [1], or [1, out_dim])
        data_list.append(graph_data)


    return pyg.DataLoader(data_list, batch_size=4, shuffle=True)


def random_face_landmarks(batch_size=4, num_frames=10):
    return torch.randn(batch_size, num_frames, 68, 2)