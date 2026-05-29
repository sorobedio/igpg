from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pprint import pprint

if __name__ == "__main__":


    # pprint(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))
    olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
    print(olmo)
    std = olmo.state_dict()
    ws =[]
    for key in std:
        vas = std[key].reshape(1, -1).cpu()
        ws.append(vas)
        print(f'{key}:---shape--{vas.shape}--mean--{vas.mean()}--std--{vas.std()}')
    ws = torch.cat(ws, dim=-1)
    print(f'all params shape:{ws.shape}==mean:{ws.mean()}==std:{ws.std()}')
    # print(ws)