import torch
import torch.nn as nn
import timm

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