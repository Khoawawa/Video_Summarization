import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
class TextDecoder(nn.Module):
    def __init__(self,backbone_name="facebook/bart-base",d_visual=2048):
        super().__init__()
        if backbone_name == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(backbone_name)
            self.model = GPT2LMHeadModel.from_pretrained(backbone_name)
        elif "bart" in backbone_name:
            self.tokenizer = BartTokenizer.from_pretrained(backbone_name)
            self.model = BartForConditionalGeneration.from_pretrained(backbone_name)
            
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented")
        
        self.visual_proj = nn.Linear(d_visual, self.model.config.d_model)
        
        
    def forward(self, x, captions=None):
        '''
        visual_feats: tensor of shape (B,F,d)
        captions: tensor of shape (B,T) or None
        return: caption
        '''
        B, F, d = x.shape 
        
        
        if captions is not None:
            captions = self.tokenizer(
                captions,
                return_tensors='pt',
                padding=True,
            )
            out = self.model(
                inputs_embeds=x,
                labels=captions['input_ids'],
            )
            return out
        else:
            generated = self.model.generate(
                inputs_embeds=x,
                max_length=50,
            )
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        
if __name__ == '__main__':
    
    B, F = 2, 8
    temporal_feats = torch.randn(B, F, 2048)
    
    model = TextDecoder(backbone_name="facebook/bart-base",d_visual=2048)
    dummy_captions = ['A dog is playing in the park.', 'A cat is sitting on the sofa.']
    out = model(temporal_feats, captions=dummy_captions)
    
    print("training loss:", out.loss.item())
    
    generated = model(temporal_feats)
    print("Generated text:", generated)