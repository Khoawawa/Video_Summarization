from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn

class AutoregressiveModel(nn.Module):
    def __init__(self, prefix_hidden_dim, backbone_name="gpt2", d_visual=2048, hidden_layers=1):
        super().__init__()
        # language model
        if backbone_name == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(backbone_name)
            self.model = GPT2LMHeadModel.from_pretrained(backbone_name)
            self.input_size = self.model.config.n_embd
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented")
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
    def __generate_caption(self, prefix_embs, max_length):
        B = prefix_embs.shape[0]
        generated = torch.full((B,1), self.tokenizer.bos_token_id, device=prefix_embs.device)
        # autoregressively generate caption
        for _ in range(max_length):
            caption_embs = self.model.transformer.wte(generated)
            input_embs = torch.cat([prefix_embs, caption_embs], dim=1)
            outputs = self.model(inputs_embeds=input_embs)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token == self.tokenizer.eos_token_id):
                break;
        return self.tokenizer.batch_decode(generated[:, 1:], skip_special_tokens=True)
    def forward(self,x_visual, captions=None):
        # x_visual: [B, d_visual]        
        prefix_embs = self.prefix_mlp(x_visual) # [B, gpt2_emb]
        prefix_embs = prefix_embs.unsqueeze(1) # [B, 1, gpt2_emb]
        
        if captions is not None:
            # training
            caption_ids = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
            caption_ids = caption_ids["input_ids"].to(x_visual.device)
            captions_embs = self.model.transformer.wte(caption_ids) # [B, T, gpt2_emb]
            
            inputs_embs = torch.cat([prefix_embs, captions_embs], dim=1) # [B, T+1, gpt2_emb]
            # padding for labels
            B, T = caption_ids.shape
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            prefix_padding = torch.full((B, 1), pad_token_id, dtype=torch.long, device=x_visual.device)
            labels = torch.cat([prefix_padding, caption_ids], dim=1) # [B, T+1]
            
            out = self.model(inputs_embeds=inputs_embs, labels=labels) # [B, T+1, vocab_size]
            return out.loss
        else:
            # inference
            return self.__generate_caption(prefix_embs, max_length=50)