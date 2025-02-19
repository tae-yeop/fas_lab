import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

# class DataLoaderX(DataLoader):
# 	def __iter__(self):
# 		return BackgroundGenerator(super().__iter__())

class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            self.batch['clean_img'] = self.batch['clean_img'].to(device=self.local_rank, non_blocking=True)
            self.batch['moire_img'] = self.batch['moire_img'].to(device=self.local_rank, non_blocking=True)


    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch