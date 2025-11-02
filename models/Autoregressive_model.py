from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
import numpy as np
custom_beam = False
class AutoregressiveModel(nn.Module):
    def __init__(self, prefix_hidden_dim, backbone_name="gpt2",prefix_len=10, d_visual=2048, hidden_layers=1, stop_token='.',beam_size=5, temperature=1.0):
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
    def prefix_projection(self, x):
        prefix_embs = self.prefix_mlp(x)
        prefix = prefix_embs.view(-1, self.prefix_len, self.model.config.n_embd) # (B, prefix_len, D)
        return self.dropout(prefix)
    def beam_search(self, generated,beam_size, batch_size, max_length, device):
        # generated (B, beam_size, prefix_len, D)
        #remap generated to (B*beam_size, prefix_len, D)
        generated = generated.view(batch_size*beam_size, self.prefix_len, -1)
        
        seq_lengths = torch.ones(batch_size,self.beam_size, device=device)
        is_stopped = torch.zeros(batch_size,self.beam_size, device=device, dtype=torch.bool)
        scores = None
        tokens = None
        for i in range(max_length):
            outputs = self.model(inputs_embeds=generated)
            logits = outputs.logits[:,-1,:] / self.temperature # (B*beam_size, vocab_size)
            #reshape logits to (B, beam_size, vocab_size)
            logits = logits.view(batch_size, beam_size, -1)
            log_probs = torch.log_softmax(logits, dim=-1) # (B, beam_size, vocab_size)
            
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1) # (B, beam_size)
                next_tokens = next_tokens.view(batch_size*beam_size,1) # (B*beam_size, 1)
                scores = scores.view(batch_size*beam_size) # (B*beam_size,)
                tokens = next_tokens # (B*beam_size, 1)
            else:
                log_probs[is_stopped] = -float(np.inf)
                log_probs[is_stopped,:, 0] = 0
                scores_sum = scores.view(batch_size,beam_size, 1) + log_probs
                seq_lengths[~is_stopped] += 1
                scores_sum_avg = scores_sum / seq_lengths.unsqueeze(-1) # (B, beam_size, vocab_size)
                scores_sum_avg, next_tokens = scores_sum_avg.view(batch_size, -1).topk(beam_size, dim=-1) # (B, beam_size)
                next_tokens_source = next_tokens // log_probs.shape[2]
                next_tokens = next_tokens % log_probs.shape[2]
                next_tokens = next_tokens.view(batch_size*beam_size,1) # (B*beam_size, 1)
                # update tokens
                tokens = tokens.view(batch_size, beam_size, -1) # (B, beam_size, prefix_len)
                tokens = tokens.gather(1, next_tokens_source.unsqueeze(-1).repeat(1,1,tokens.shape[-1])) # (B, beam_size, T)
                tokens = tokens.view(batch_size*beam_size, -1) # (B*beam_size, prefix_len)
                tokens = torch.cat((tokens, next_tokens), dim=1) # (B*beam_size, T+1)
                generated = generated.view(batch_size, beam_size, -1) # (B, beam_size, prefix_len, D)
                generated = generated.gather(1, next_tokens_source.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, generated.shape[2], generated.shape[3]))  # Shape: (B, beam_size, T, D)    
                generated = generated.view(batch_size*beam_size, -1) # (B*beam_size, T+1, D)
                scores = scores_sum_avg.view(batch_size*beam_size)
                seq_lengths = seq_lengths.gather(1, next_tokens_source)
                is_stopped = is_stopped.gather(1, next_tokens_source)
            next_token_embed = self.model.transformer.wte(next_tokens) # (B*beam_size, D)
            generated = torch.cat((generated, next_token_embed.unsqueeze(1)), dim=1) # (B*beam_size, T+1, D)
            is_stopped = is_stopped | next_tokens.view(batch_size, beam_size).eq(self.stop_token_idx)
            
            if is_stopped.all():
                break
        scores = scores.view(batch_size, beam_size) / seq_lengths  # Shape: (B, beam_size)
        tokens = tokens.view(batch_size, beam_size, -1)  # Shape: (B, beam_size, T)
        output_texts = []
        for b in range(batch_size):
            batch_tokens = tokens[b].cpu().numpy()  # Shape: (beam_size, T)
            batch_lengths = seq_lengths[b].cpu().numpy()  # Shape: (beam_size)
            batch_scores = scores[b].cpu().numpy()  # Shape: (beam_size)
            # Get top-scoring sequence
            order = batch_scores.argsort()[::-1]  # Sort descending
            top_text = self.tokenizer.decode(batch_tokens[order[0], :int(batch_lengths[order[0]])])
            output_texts.append(top_text)

        return output_texts # list[str] * B
                
                
    def forward(self, x_visual, caption_tokens=None,mask=None, max_length=50):
        """
        x_visual : Tensor[B, d_visual]
        captions : None (inference) or list[str] (training)
        """
        B = x_visual.shape[0]
        prefix_embs = self.prefix_projection(x_visual) # (B, prefix_len, D)
        if caption_tokens is not None:
            # get embeddings of captions token
            caption_embs = self.model.transformer.wte(caption_tokens) # (B, T, D)
            # prepend the prefix token
            inputs_embs = torch.cat([prefix_embs, caption_embs], dim=1)  # (B,prefix_len+T,D)
            # crafting the labels
            prefix_dummy_token = torch.zeros(B, self.prefix_len, device=x_visual.device, dtype=torch.long)
            labels = torch.cat([prefix_dummy_token, caption_tokens], dim=1) # (B, prefix_len+T)
            
            out = self.model(inputs_embeds=inputs_embs, labels=labels, attention_mask=mask)
            return out
        else:
            generated = prefix_embs.unsqueeze(1).repeat(1, self.beam_size, 1, 1) # (B, beam_size, prefix_len, D)
            if custom_beam:
                return self.beam_search(generated,beam_size=self.beam_size,batch_size=B, max_length=max_length,device=x_visual.device)
            else:
                return self.model.generate(
                    inputs_embs=prefix_embs,
                    max_length=max_length + self.prefix_len,
                    num_beams=self.beam_size,
                    temperature=self.temperature,
                    early_stopping=True,
                    eos_token_id=self.stop_token_idx,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            

if __name__ == '__main__':
    model = AutoregressiveModel(prefix_hidden_dim=512, backbone_name="gpt2", d_visual=2048, hidden_layers=1)
    print(model.tokenizer.pad_token_id, model.tokenizer.eos_token_id, model.model.config.vocab_size)