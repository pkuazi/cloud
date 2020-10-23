import torch

class tiledata_prefetcher():
    def __init__(self, loader, use_cuda=True):
        self.loader = iter(loader)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_inputs, self.next_targets, self.next_index = next(self.loader)
#             self.next_inputs, self.next_index = next(self.loader)
        except StopIteration:
            self.next_inputs = None
            self.next_targets = None
            self.next_index = None
            return
        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                self.next_inputs = self.next_inputs.cuda(non_blocking=True)
                self.next_targets = self.next_targets.cuda(non_blocking=True)

    def next(self):
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_inputs
        targets = self.next_targets
        index = self.next_index
        self.preload()
        return inputs, targets, index
#         return inputs, index
    
class data_prefetcher():
    def __init__(self, loader, use_cuda=True):
        self.loader = iter(loader)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
#             self.next_inputs, self.next_targets, self.next_index = next(self.loader)
            self.next_inputs, self.next_index = next(self.loader)
        except StopIteration:
            self.next_inputs = None
            self.next_targets = None
            self.next_index = None
            return
        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                self.next_inputs = self.next_inputs.cuda(non_blocking=True)
                self.next_targets = self.next_targets.cuda(non_blocking=True)

    def next(self):
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_inputs
#         targets = self.next_targets
        index = self.next_index
        self.preload()
#         return inputs, targets, index
        return inputs, index
