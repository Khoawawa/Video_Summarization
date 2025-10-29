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
            self.vocab_size = self.model.config.vocab_size
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
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
        B, _, D = prefix_embs.shape
        device = prefix_embs.device
        vocab_size = self.model.config.vocab_size

        # Sanitize prefix
        prefix_embs = torch.nan_to_num(prefix_embs, nan=0.0, posinf=0.0, neginf=0.0)
        prefix_embs = prefix_embs.clamp(min=-100, max=100)

        # Start with BOS
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_length):
            # Embed generated tokens
            caption_embs = self.model.transformer.wte(generated)
            caption_embs = torch.nan_to_num(caption_embs, nan=0.0, posinf=0.0, neginf=0.0)
            caption_embs = caption_embs.clamp(min=-100, max=100)

            # Prepend prefix
            input_embs = torch.cat([prefix_embs, caption_embs], dim=1)  # (B, 1+T, D)

            # NO CACHE. NO PAST. NO PROBLEMS.
            with torch.no_grad():
                outputs = self.model(inputs_embeds=input_embs, use_cache=False)

            logits = outputs.logits[:, -1, :]  # (B, V)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=-88.0, neginf=-88.0)

            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            next_token = next_token.clamp(min=0, max=vocab_size - 1)

            generated = torch.cat([generated, next_token], dim=1)

            if torch.all(next_token == self.tokenizer.eos_token_id):
                break

        return self.tokenizer.batch_decode(generated[:, 1:], skip_special_tokens=True)
    def forward(self,x_visual, captions=None):
        # x_visual: [B, d_visual]    
        B = x_visual.shape[0]    
        
        prefix_embs = self.prefix_mlp(x_visual) # [B, gpt2_emb]
        prefix_embs = torch.nan_to_num(prefix_embs, nan=0.0, posinf=0.0, neginf=0.0)
        prefix_embs = prefix_embs.clamp(min=-10, max=10)
        prefix_embs = prefix_embs.unsqueeze(1) # [B, 1, gpt2_emb]
        if captions is not None:
            # training
            caption_ids = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=50).to(x_visual.device)
            caption_ids = caption_ids["input_ids"]
            captions_embs = self.model.transformer.wte(caption_ids) # [B, T, gpt2_emb]
            
            inputs_embs = torch.cat([prefix_embs, captions_embs], dim=1) # [B, T+1, gpt2_emb]
            labels = caption_ids.clone()
            labels = torch.cat([
                torch.full((B, 1), self.tokenizer.pad_token_id, device=x_visual.device, dtype=torch.long),
                labels
            ], dim=1)
            out = self.model(inputs_embeds=inputs_embs, labels=labels) # [B, T+1, vocab_size]
            return out.loss
        else:
            # inference
            return self.__generate_caption(prefix_embs, max_length=50)

if __name__ == '__main__':
    model = AutoregressiveModel(prefix_hidden_dim=512, backbone_name="gpt2", d_visual=2048, hidden_layers=1)
    print(model.tokenizer.pad_token_id, model.tokenizer.eos_token_id, model.model.config.vocab_size)