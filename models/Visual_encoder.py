import torch
import torch.nn as nn
import torchvision.models as models

batch_first = True
class VisualEncoder(torch.nn.Module):
    '''
    Module for encoding visual and temporal features from video clips
    
    it should receive a tensor of shape (C, F, H, W)
    return a tensor of shape (F, d) # d is embedding dimension of the visual encoder
    '''
    def __init__(self, spatial_backbone='resnet50', temporal_backbone='default', pretrained=True):
        super().__init__()
        
        # Spatial encoder
        if spatial_backbone == 'resnet50':
            spatial_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            d = spatial_model.fc.in_features # 2048
        elif spatial_backbone == 'resnet18':
            spatial_model = models.resnet18(pretrained=models.ResNet18_Weights.DEFAULT if pretrained else None)
            d = spatial_model.fc.in_features # 512
        
        else:
            raise NotImplementedError(f'Spatial backbone {spatial_backbone} not implemented')
        
        if 'resnet' in spatial_backbone:
            self.spatial_model = nn.Sequential(*list(spatial_model.children())[:-1]) # remove the classification head
            self.d = d
            self.is_clip = False
        
        # Temporal encoder
        if temporal_backbone == 'default':
            self.temporal_model = TemporalEncoder(d_model=self.d)
        else:
            raise NotImplementedError(f'Temporal backbone {temporal_backbone} not implemented')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: tensor of shape (C,F,H,W)
        return: tensor of shape (F,d)
        '''
        C, F, H, W = x.shape
        
        if not self.is_clip:
            # handle CNN backbones
            
            x = x.permute(1,0,2,3) # (F,C,H,W)
            
            features = self.spatial_model(x) # (F,D,1,1)
            features = features.view(F, self.d) # (F,D)

            out = self.temporal_model(features.unsqueeze(0)) # (1,F,D)
            
        return out.squeeze(0) # (F,D)

class TemporalEncoder(nn.Module):
    def __init__(self, d_model, nhead=8, n_layers=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=d_model*4,
            batch_first=batch_first,
        )      
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self,x):
        '''
        x: (B,F,d)
        '''
        return self.transformer(x)
            
if __name__ == '__main__':
    model = VisualEncoder(spatial_backbone='resnet50', pretrained=False)
    x = torch.randn(3, 8, 224, 224) # (C,F,H,W)
    with torch.no_grad():
        out = model(x)
    print(out.shape) # (F,d)