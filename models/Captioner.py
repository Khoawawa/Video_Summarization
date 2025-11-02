import torch
import torch.nn as nn

from models.Visual_encoder import VisualEncoder
from models.Autoregressive_model import AutoregressiveModel
# for image captioning the bread and butter are visual encoder and text decoder
# tryout the clipcap method
# for visual encoder we can use transfer learning of resnet
# for the captioning generation we use prefix embedding + caption --> gpt2#
class Captioner(nn.Module):
    def __init__(self,
                 visual_backbone='resnet50',
                 prefix_hidden_dim=512,
                 text_backbone_name="gpt2",
                 mapping_hidden_layers=1,
                 pretrained=True,
                 unfreeze_layer=0,
                 prefix_len=10,
                 beam_size=5,
                 temperature=1.0,
                 max_length=20):
        super().__init__()
        self.visual_encoder = VisualEncoder(visual_backbone, pretrained, unfreeze_layer)
        d_visual = self.visual_encoder.out_dim
        self.text_decoder = AutoregressiveModel(prefix_hidden_dim, backbone_name=text_backbone_name,max_length=max_length, prefix_len=prefix_len, d_visual=d_visual, hidden_layers=mapping_hidden_layers, beam_size=beam_size, temperature=temperature)
    
    def forward(self, images, caption_tokens=None, mask=None):
        '''
        images: tensor of shape (B,C,H,W)
        captions: tensor of shape (B,T) or None
        '''
        visual_feats = self.visual_encoder(images) # [B, d_visual]
        return self.text_decoder(visual_feats, caption_tokens, mask)
    
if __name__ == '__main__':
    model = Captioner()
    x = torch.randn(2, 3, 224, 224)
    captions = ["A dog is playing in the park.", "A cat is sitting on the sofa."]
    output = model(x, captions)
    print(output)
    
    with torch.no_grad():
        output = model(x)
        print(output)
    