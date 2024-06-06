import torch
from time import time
from improve.wrapper import dict_util as du

class DataPrefetcher:
    def __init__(self, loader, device):
        self.device = device
        self.loader = loader
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()

        self.preload()

    def preload(self):
        try:
            # Dataloader will prefetch data to cpu so this step is very quick
            self.batch = next(self.iter)
        except StopIteration:
            self.batch = None
            self.iter = iter(self.loader)
            return
        with torch.cuda.stream(self.stream):
            self.batch = du.apply(self.batch, lambda x: x.to(self.device, non_blocking=True))

    def next(self):
        clock = time()
        batch = self.batch
        if batch is not None:
            record = lambda x: x.record_stream(torch.cuda.current_stream())
            _ = du.apply(batch, lambda x: record(x) if x is not None else x)

        self.preload()
        return batch, time() - clock

    def next_without_none(self):
        batch, time = self.next()
        if batch is None:
            batch, time = self.next()
        return batch, time


