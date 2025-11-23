from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
import numpy as np
custom_beam = False
class AutoregressiveModel(nn.Module):
    def __init__(self, prefix_hidden_dim, max_length=20,backbone_name="gpt2",prefix_len=10, d_visual=2048, hidden_layers=1, stop_token='.',beam_size=5, temperature=1.0):
        super().__init__()
        # language model
        if backbone_name == "gpt2":
            self.model = GPT2LMHeadModel.from_pretrained(backbone_name)
            self.input_size = self.model.config.n_embd
            self.vocab_size = self.model.config.vocab_size
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
        layers.append(nn.Linear(prefix_hidden_dim, prefix_len * self.input_size))
        self.prefix_mlp = nn.Sequential(*layers)
        # stop token
        self.tokenizer = GPT2Tokenizer.from_pretrained(backbone_name)
        self.stop_token_idx = self.tokenizer.encode(stop_token)[0]
        # temperature
        self.temperature = temperature
        # beam size
        self.beam_size = beam_size
        # max length
        self.max_length = max_length
        
    def prefix_projection(self, x):
        prefix_embs = self.prefix_mlp(x)
        prefix = prefix_embs.view(-1, self.prefix_len, self.model.config.n_embd) # (B, prefix_len, D)
        return self.dropout(prefix)
         
    def forward(self, x_visual, caption_tokens=None,mask=None):
        """
        x_visual : Tensor[B, d_visual]
        captions : None (inference) or list[str] (training)
        """
        B = x_visual.shape[0]
        prefix_embs = self.prefix_projection(x_visual) # (B, prefix_len, D)
        assert prefix_embs.shape == (B, self.prefix_len, self.model.config.n_embd)
        if caption_tokens is not None:
            # get embeddings of captions token
            caption_embs = self.model.transformer.wte(caption_tokens) # (B, T, D)
            # prepend the prefix token
            inputs_embs = torch.cat([prefix_embs, caption_embs], dim=1)  # (B,prefix_len+T,D)
            # crafting the labels
            prefix_dummy_token = torch.zeros(B, self.prefix_len, device=x_visual.device, dtype=torch.long)
            labels = torch.cat([prefix_dummy_token, caption_tokens], dim=1) # (B, prefix_len+T)
            
            ce_out = self.model(inputs_embeds=inputs_embs, labels=labels, attention_mask=mask)
            # aligment loss
            
            txt_embs = self.model.transformer(inputs_embeds=caption_embs).last_hidden_state.mean(dim=1).detach()
            vis_embs = prefix_embs.mean(dim=1)
            
            align_loss = 1 - torch.cosine_similarity(txt_embs, vis_embs, dim=-1).mean() #  
            return ce_out, align_loss
        else:
            generated = prefix_embs.unsqueeze(1).repeat(1, self.beam_size, 1, 1) # (B, beam_size, prefix_len, D)
            if custom_beam:
                return self.beam_search(generated,beam_size=self.beam_size,batch_size=B, max_length=self.max_length,device=x_visual.device)
            else:
                output = self.model.generate(
                    inputs_embeds=prefix_embs,
                    max_length=self.max_length + self.prefix_len,
                    num_beams=self.beam_size,
                    temperature=self.temperature,
                    early_stopping=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                return self.tokenizer.batch_decode(output, skip_special_tokens=True)
            

if __name__ == '__main__':
    model = AutoregressiveModel(prefix_hidden_dim=512, backbone_name="gpt2", d_visual=2048, hidden_layers=1)
    print(model.tokenizer.pad_token_id, model.tokenizer.eos_token_id, model.model.config.vocab_size)