import torch

if __name__ == "__main__":
    model_list=[ 'cifar10_vgg11_bn',
 'cifar10_vgg13_bn',
 'cifar10_vgg16_bn',
 'cifar10_vgg19_bn',]
    for model_name in model_list:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True)
        print(model)
        print('------------------------------------------------------')
