import torch

if __name__ == "__main__":
    model_list=[ 'cifar10_repvgg_a0',
 'cifar10_repvgg_a1',
 'cifar10_repvgg_a2',]
    for model_name in model_list:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True)
        print(model)
        print('------------------------------------------------------')
