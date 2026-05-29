from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    olmo = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3.1-32B-Instruct")
    print(olmo)