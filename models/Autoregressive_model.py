from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn

class AutoregressiveModel(nn.Module):
    def __init__(self, prefix_hidden_dim, backbone_name="gpt2",prefix_len=10, d_visual=2048, hidden_layers=1):
        super().__init__()
        # language model
        if backbone_name == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(backbone_name)
            self.model = GPT2LMHeadModel.from_pretrained(backbone_name)
            self.input_size = self.model.config.n_embd
            self.vocab_size = self.model.config.vocab_size
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented")
        self.prefix_len = prefix_len
        self.dropout = nn.Dropout(0.1)
        # freeze LM
        for p in self.model.parameters():
            p.requires_grad = False
        # mapping network
        layers = []
        in_dim = d_visual
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, prefix_hidden_dim))
            layers.append(nn.ReLU())
            in_dim = prefix_hidden_dim
        layers.append(nn.Linear(prefix_hidden_dim, self.input_size))
        self.prefix_mlp = nn.Sequential(*layers)
    def prefix_projection(self, x):
        prefix_embs = self.prefix_mlp(x)
        prefix = prefix_embs.view(-1, self.prefix_len, self.model.config.n_embd) # (B, prefix_len, D)
        return self.dropout(prefix)
    def forward(self, x_visual, captions=None, max_length=50):
        """
        x_visual : Tensor[B, d_visual]
        captions : None (inference) or list[str] (training)
        """
        B = x_visual.shape[0]
        prefix_embs = self.prefix_projection(x_visual) # (B, prefix_len, D)
        if captions is not None:
            # tokenise the ground-truth captions
            enc = self.tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(x_visual.device)  # (B, T)
            caption_ids = enc.input_ids                 # (B, T)

            # embed captions
            caption_embs = self.model.transformer.wte(caption_ids)  # (B,T,D)

            # prepend the prefix token
            inputs_embs = torch.cat([prefix_embs, caption_embs], dim=1)  # (B,prefix_len+T,D)

            # labels: ignore the prefix token (pad) and shift captions by 1
            labels = torch.full((B, self.prefix_len), -100, device=x_visual.device, dtype=torch.long)
            labels = torch.cat([labels, caption_ids], dim=1)  # [B, prefix_len + T]

            out = self.model(inputs_embeds=inputs_embs, labels=labels)
            return out.loss
        else:
            # start with a single BOS token (the prefix will be the *first* token)
            bos_id = self.tokenizer.bos_token_id
            bos_emb = self.model.wte.weight[bos_id:bos_id+1] # (1, D)
            bos_emb = bos_emb.expand(B, 1, -1)
            start_embs = bos_emb + prefix_embs[:, :1, :] # bias first token
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs_embeds=start_embs,
                    max_length=max_length + 1,      # +1 because we already fed BOS
                    do_sample=True,                # greedy
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    top_p=0.9,
                    temperature=0.7,
                    use_cache=True,                 # safe – HF handles the cache
                )

            # remove the BOS token that we fed manually
            return self.tokenizer.batch_decode(
                generated_ids[:, 1:], skip_special_tokens=True
            )

if __name__ == '__main__':
    model = AutoregressiveModel(prefix_hidden_dim=512, backbone_name="gpt2", d_visual=2048, hidden_layers=1)
    print(model.tokenizer.pad_token_id, model.tokenizer.eos_token_id, model.model.config.vocab_size)