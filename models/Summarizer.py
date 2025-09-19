import torch
from torch import nn
from transformers import BertConfig, BertModel
batch_first=False
bidirectional=False

class Summarizer(nn.Module):
    def __init__(self,bert_attention_heads, bert_hidden_size, bert_num_hidden_layers, pad_token_id, vocab_size):
        super().__init__()
        
        self.bert_config = BertConfig(
            num_attention_heads=bert_attention_heads,
            hidden_size = bert_hidden_size,
            num_hidden_layers = bert_num_hidden_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )
        
        self.temporal_transformer = BertModel(self.bert_config)
        
    