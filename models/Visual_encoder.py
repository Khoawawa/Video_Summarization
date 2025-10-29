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
            self.spatial_model = nn.Sequential(*list(spatial_model.children())[:-2])  # up to last conv
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            for param in self.spatial_model.parameters():
                param.requires_grad = False
            if unfreeze_layer > 0:
                layers_to_unfreeze = list(self.spatial_model.children())[-unfreeze_layer:]
                for layer in layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: tensor of shape (B,C,H,W)
        return: tensor of shape (F,d)
        '''
        output = self.spatial_model(x) # (B, 2048, H', W')
        output = self.pool(output) # (B, 2048, 1, 1)
        return output.flatten(1) # (B, d)
        
        
