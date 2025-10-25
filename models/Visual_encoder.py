import torch
import torch.nn as nn
import torchvision.models as models

class VisualEncoder(torch.nn.Module):
    '''
    Module for encoding visual features
    '''
    def __init__(self, spatial_backbone='resnet50', pretrained=True, unfreeze_layer=0):
        super().__init__()
        # spatial_model
        if spatial_backbone == 'resnet50':
            spatial_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.out_dim = 2048
        else:
            raise NotImplementedError(f'Spatial backbone {spatial_backbone} not implemented')
        
        if 'resnet' in spatial_backbone:
            self.spatial_model = nn.Sequential(*list(spatial_model.children())[:-1]) # remove the classification head
            for param in self.spatial_model.parameters():
                param.requires_grad = False
            if unfreeze_layer > 0:
                blocks = [self.spatial_model[-2][i] for i in range(-unfreeze_layer, 0)]
                for block in blocks:
                    for param in block.parameters():
                        param.requires_grad = True
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: tensor of shape (B,C,H,W)
        return: tensor of shape (F,d)
        '''
        B,C, H, W = x.shape
        output = self.spatial_model(x)
        output = self.pool(output)
        return output.flatten(1) # (B, d)
        
        
