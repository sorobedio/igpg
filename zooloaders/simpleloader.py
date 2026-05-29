import pytorch_lightning as pl
from torch.utils.data import DataLoader

from zoodatasets.get_zoo_data import ZooDataset


class ZooDataModule(pl.LightningDataModule):
    def __init__(self,  root='modelzoos', dataset="Llama-3.2-3B-Instruct", split='train', topk=None,
                 scale=1.0, transform=None, normalize=False, tgt=None, exd=None, length=3072,
                 zoo_file='zoo_config.simple_config.yaml', batch_size=32, num_workers=4):
        super().__init__()
        self.root = root
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.topk = topk
        self.normalize = normalize
        self.scale=scale
        self.length=length
        self.tgt = tgt
        self.exd = exd
        self.transform = transform
        self.split = split
        self.zoo_file = zoo_file


    def prepare_data(self):
        pass
        # datasets.CIFAR10(self.data_root, train=True, download=True)
        # datasets.CIFAR10(self.data_root, train=False, download=True)

    def setup(self, stage):

        if stage == "fit":
            self.trainset = ZooDataset(root=self.root, dataset=self.dataset, split='train',
                                       scale=self.scale, topk=self.topk, normalize=self.normalize,
                                       tgt=self.tgt, exd=self.exd, length=self.length,  zoo_file=self.zoo_file,
                                       transform=self.transform)
            self.valset = ZooDataset(root=self.root, dataset=self.dataset, split='train',
                                       scale=self.scale, topk=1000, normalize=self.normalize,
                                       tgt=self.tgt, exd=self.exd, length=self.length,  zoo_file=self.zoo_file,
                                       transform=self.transform)

        if stage == "test":
            pass
            # self.testset = ZooDataset(zoo_root=self.zoo_root, zoo_name=self.zoo_name, split='train',
            #                            scale=self.scale, topk=10, normalize=self.normalize,
            #                            tgt=self.tgt, exd=self.exd)
        if stage == "predict":
            pass
            # pass

    def train_dataloader(self):
        # sampler = DistributedSampler(self.trainset, shuffle=True)
        # sampler = DistributedSampler(self.trainset)
        return DataLoader(
            self.trainset,
            # sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        pass
        # return DataLoader(
        #     self.testset,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=False,
        # )

    def predict_dataloader(self):
        pass
        # return DataLoader(
        #     self.cifar10_predict,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=False,
        # )

