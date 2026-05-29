from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pprint import pprint

if __name__ == "__main__":


    pprint(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))