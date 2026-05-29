from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pprint import pprint

if __name__ == "__main__":


    # pprint(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))
    olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
    print(olmo)