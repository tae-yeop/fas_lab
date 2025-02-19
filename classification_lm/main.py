import argparse
import wandb

from config import TraningConfig
from datasets import get_dataloader
from models import SimpleModel, ViT_Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--ngpus', type=int, default=1)
    args = parser.parse_args()

    cfg = TraningConfig()

    dataloader = get_dataloader(cfg.json_path, cfg.batch_size)

    fabric = Fabric(
        accelerator='cuda', 
        num_nodes=args.nnodes,
        devices=args.ngpus,
        strategy=cfg.pl_strategy, 
        precision=cfg.pl_precision,
    )

    fabric.launch()
    fabric.seed_everything(cfg.seed)

    if cfg.model == 'simple':
        model = SimpleModel().cuda()
    elif cfg.model == 'vit':
        model = ViT_Model().cuda()
    else:
        pass

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        for batch in dataloader:
            lm, label = batch
            lm = lm.float().cuda()
            label = label.long().cuda()

            optimizer.zero_grad()
            outputs = model(lm)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
