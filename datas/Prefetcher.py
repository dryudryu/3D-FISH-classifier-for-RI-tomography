# https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
import torch

class Prefetcher():
    def __init__(self, loader, device):
        self.loader = loader
        self.dataset = loader.dataset
        self.device = device
        self.stream = torch.cuda.Stream(device)
              
    def preload(self):
        self.next_input, self.next_target, self.path = next(self.iter_loader)
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.to(self.device, non_blocking=True)
            self.next_target = self.next_target.to(self.device, non_blocking=True)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input_ = self.next_input
        target = self.next_target
        path = self.path
        self.preload()
        return input_, target, path

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.iter_loader = iter(self.loader)
        self.preload()
        for i in range(1, len(self.loader)):
            yield self.next()

        input_, target, path = next(self.iter_loader)
        input_ = input_.to(self.device)
        target = target.to(self.device, non_blocking=True)
        return input_, target, path

# Original Repo
class data_prefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, path = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.path = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.to(device, non_blocking=True)
            self.next_target = self.next_target.to(device, non_blocking=True)
            self.path = path

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        path = self.path
        self.preload()
        return input, target, path

if __name__ == "__main__":
    from preprocess3d import TRAIN_AUGS_3D
    from CSVLoader import CSVLoader
    import time

    device = torch.device("cuda")
    data_path = "/data2/DW/181121_Lympho/lympho_dataset/190110_mice/190206_mice_add_3split_"
    train_loader = CSVLoader(data_path + "train.csv", 32, sampler=False,
                             transform=TRAIN_AUGS_3D, aug_rate=0.5,
                             num_workers=16, shuffle=True, drop_last=True)

    train_prefetch = Prefetcher(train_loader, device)

    t1 = time.time()    
    for i, (input_, target_, path) in enumerate(train_prefetch):
        if i == 0:
            print("t1 0 input : ", input_.shape, target_.shape)
    print("Prefetcher1 : ", i, time.time() - t1)

    t3 = time.time()
    for i, (input_, target_, path) in enumerate(train_loader):
        input_ = input_.to(device)
        target_ = target_.to(device, non_blocking=True)
        if i == 0:
            print("t3 0 input : ", input_.shape, target_.shape)
    print("Loader3 i : ", i, time.time() - t3)

    
    prefetcher = data_prefetcher(train_loader, device)
    t5 = time.time()
    input_, target, path = prefetcher.next()
    i = 0
    while input_ is not None:
        if i == 0:
            print("t5 0 input_ : ", input_.shape, target_.shape)

        i += 1
        input_, target, path = prefetcher.next()
    print("data_prefetcher i : ", i, time.time() - t5)