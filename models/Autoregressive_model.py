from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
import numpy as np
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
            generated = prefix_embs
            scores = None
            tokens = None
            seq_lengths = torch.ones(self.beam_size, device=x_visual.device)
            is_stopped = torch.zeros(self.beam_size, device=x_visual.device, dtype=torch.bool)
            for i in range(max_length):
                outputs = self.model(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / self.temperature # (B, vocab_size)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(self.beam_size, -1)
                    generated = generated.expand(self.beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(self.beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:,None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_avg = scores_sum / seq_lengths[:,None]
                    scores_sum_avg, next_tokens = scores_sum_avg.view(-1).topk(self.beam_size, -1)
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_avg * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = self.model.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(self.stop_token_idx).squeeze()
                if is_stopped.all():
                    break
            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [
                self.tokenizer.decode(
                    output[: int(length)]
                ) for output, length in zip(output_list, seq_lengths)
            ]
            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order]
            return output_texts

if __name__ == '__main__':
    model = AutoregressiveModel(prefix_hidden_dim=512, backbone_name="gpt2", d_visual=2048, hidden_layers=1)
    print(model.tokenizer.pad_token_id, model.tokenizer.eos_token_id, model.model.config.vocab_size)