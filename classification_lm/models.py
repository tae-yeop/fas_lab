import torch
import torch.nn as nn
import timm
from torch_geometric.nn import GCNConv, global_mean_pool

class SimpleModel(nn.Module):
    def __init__(
              self, 
              input_size=136, 
              hidden_size=64, 
              num_layers=2, 
              num_classes=2
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size *2, num_classes)

    def forward(self, x):
        """
        x: (batch_size, 68, 2) → (batch_size, 136)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        print('x.shape', x.shape) # x.shape torch.Size([32, 30, 136])
        lstm_out, _ = self.lstm(x) # lstm_out shape torch.Size([32, 128])

        print('lstm_out shape', lstm_out.shape)
        out = self.fc(lstm_out)

        return out



class ViT_Model(nn.Module):
    def __init__(
        self, 
         num_classes=2, 
         patch_size=1, 
         embed_dim=128
    ):
        super().__init__()

        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=num_classes
        )

        self.vit.patch_embed.proj = nn.Linear(136, embed_dim)

    def forward(self, x):
        out = self.vit(x)
        return out


class LandmarkGNN(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=64, out_dim=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 그래프 풀링 (여기서는 mean-pool)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            # 단일 그래프이면 전체 노드 평균
            x = x.mean(dim=0, keepdim=True)
        
        # 분류기
        out = self.fc(x)  # shape [batch_size, out_dim]
        return out