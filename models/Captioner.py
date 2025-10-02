import torch
import torch.nn as nn

from models.Visual_encoder import VisualEncoder
from models.Text_decoder import TextDecoder

class Captioner(nn.Module):
    def __init__(self,
                 visual_backbone='resnet50',
                 temporal_backbone='default',
                 text_backbone='facebook/bart-base',
                 pretrained=True):
        super().__init__()
        
        self.visual_encoder = VisualEncoder(
            spatial_backbone=visual_backbone,
            temporal_backbone=temporal_backbone,
            pretrained=pretrained,
        )
        
        self.text_decoder = TextDecoder(
            backbone_name=text_backbone,
            d_visual=self.visual_encoder.d,
        )
        
        self.linear = nn.Linear(self.visual_encoder.d, self.text_decoder.model.config.d_model)
    
    def forward(self, video_clips, captions=None):
        '''
        video_clips: tensor of shape (C,F,H,W)
        captions: tensor of shape (B,T) or None
        '''
        visual_feats = self.visual_encoder(video_clips) # (F,d)
        visual_feats :torch.Tensor = self.linear(visual_feats) # (F, d_model)
        visual_feats = visual_feats.unsqueeze(0) # (1,F,d_model)
        if captions is not None:
            visual_feats = visual_feats.expand(len(captions), -1, -1) # [no_captions, F, d_model]
        
        return self.text_decoder(visual_feats, captions) # depend on the captions input will output either loss or generated captions
    
if __name__ == '__main__':
    model = Captioner()
    x = torch.randn(3, 8, 224, 224) # (C,F,H,W)
    caption = "a cat is playing with a ball"
    tokenizer = model.text_decoder.tokenizer
    out = model(x, captions=[caption])
    print(out.loss.item())
    
    model.eval()
    generated = model(x)
    print(generated)
    print(type(generated))
    
    