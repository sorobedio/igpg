import torch
from torch import nn
from transformers import T5Tokenizer, T5EncoderModel

# --- Include MLPResidualBlock and HFEmbedder as defined (with the bugfixes) ---
class MLPResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pre_layer_norm, post_dropout):
        super().__init__()
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(input_size))
        layers += [
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size, output_size),
            nn.SiLU(),
        ]
        if post_dropout:
            layers.append(nn.Dropout(0.05))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.mlp(x)


class HFEmbedder(nn.Module):
    def __init__(self, method: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.method = method
        self.max_length = max_length
        self.is_clip = False
        self.is_qwen = False

        if "clip" in method:
            self.is_clip = True
            from transformers import CLIPTextModel, CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(method)
            self.hf_module = CLIPTextModel.from_pretrained(method, **hf_kwargs)
        elif "t5" in method:
            self.tokenizer = T5Tokenizer.from_pretrained(method)
            self.hf_module = T5EncoderModel.from_pretrained(method, **hf_kwargs)
        elif "qwen3" in method:
            self.is_qwen = True
            from sentence_transformers import SentenceTransformer
            self.hf_module = SentenceTransformer(method)
        else:
            raise NotImplementedError
        if not self.is_qwen:
            self.hf_module.eval().requires_grad_(False)
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

    def forward(self, text: list[str]) -> torch.Tensor:
        if self.is_qwen:
            return self.hf_module.encode(text, convert_to_tensor=True)
        else:
            batch_encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                padding="max_length",
            )
            outputs = self.hf_module(
                input_ids=batch_encoding["input_ids"].cuda(),
                attention_mask=batch_encoding["attention_mask"].cuda(),
            )
            return outputs[self.output_key][:, 0].float()  # CLS-style token


# --- Conditioner Module ---
class Conditioner(nn.Module):
    def __init__(self, hf_method, hf_max_length, arch_emb_dim: int, layer_emb_dim: int,
                 chunk_emb_dim: int, output_emb_dim: int = 4096,
                 max_num_chunks: int = 5220, to_img=False, out_ch=1, out_size=64):
        super().__init__()
        self.chunk_embedder = nn.Embedding(max_num_chunks, chunk_emb_dim)
        self.layer_embedder = HFEmbedder(hf_method, hf_max_length).cuda()
        self.arch_embedder = nn.Identity()

        # self.chunk_proj = nn.Linear(chunk_emb_dim, chunk_emb_dim)
        # self.layer_proj = nn.Linear(layer_emb_dim, layer_emb_dim)
        # self.arch_proj = nn.Linear(arch_emb_dim, arch_emb_dim)

        mix_dim = arch_emb_dim + layer_emb_dim + chunk_emb_dim
        self.mix_proj = nn.Linear(mix_dim, output_emb_dim)

        self.mlp = MLPResidualBlock(output_emb_dim, output_emb_dim * 2, output_emb_dim,
                                    pre_layer_norm=True, post_dropout=True)
        self.to_img = to_img
        self.out_ch = out_ch
        self.out_size = out_size

    def forward(self, arch_emb: torch.Tensor, layer_info: list[str], chunk_idx: torch.Tensor) -> torch.Tensor:

        # print('===============================================')
        e_c = self.chunk_embedder(chunk_idx.cuda())                          # (B, chunk_emb_dim)
        with torch.no_grad():
            e_l = self.layer_embedder(layer_info)                     # (B, layer_emb_dim)
        e_a = self.arch_embedder(arch_emb.cuda())                           # (B, arch_emb_dim)
        # print(e_c.shape, e_l.shape, e_a.shape)
        # exit()
        mixed = torch.cat([e_a, e_l, e_c], dim=-1)              # (B, mix_dim)
        # print(mixed.shape)
        x_m = self.mix_proj(mixed)                              # (B, output_emb_dim)
        emb = self.mlp(x_m)
        # print(x_m.shape, emb.shape)# (B, output_emb_dim)
        if self.to_img:
            emb = emb.reshape(-1, self.out_ch, self.out_size, self.out_size)
        return emb







#499900 maximum chunk for 65536

