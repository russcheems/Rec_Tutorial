import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .senet import SENet # 直接复用
from .bilinear_interaction import BilinearInteraction

class FiBiNET(nn.Module):

    def __init__(self, 
                 num_fields, 
                 embedding_dim, 
                 hidden_dims=[256, 128, 64], 
                 dropout_rate=0.2, 
                 reduction_ratio=3,
                 bilinear_type='field_interaction'): # 这个type暂时先不用

        super(FiBiNET, self).__init__()
        

        self.senet = SENet(num_fields, reduction_ratio)
        
        self.bilinear_interaction = BilinearInteraction(
            embedding_dim, num_fields, bilinear_type
        )
        
        self.se_bilinear_interaction = BilinearInteraction(
            embedding_dim, num_fields, bilinear_type
        )

        # if bilinear_type == 'field_all':
        #     bilinear_dim = num_fields * num_fields
        # else:  # field_interaction
        #     bilinear_dim = (num_fields * (num_fields - 1)) // 2
        
        # 组合层
        # 输入包括：原始特征交互 + SENet加权特征交互
        combined_dim = bilinear_dim * 2
        

        dnn_layers = []
        input_dim = combined_dim
        
        for hidden_dim in hidden_dims:
            dnn_layers.append(nn.Linear(input_dim, hidden_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.BatchNorm1d(hidden_dim))
            dnn_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        self.dnn_layers = nn.Sequential(*dnn_layers)

        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, feature_embeddings):
        senet_output = self.senet(feature_embeddings)
        bilinear_output = self.bilinear_interaction(feature_embeddings)
        se_bilinear_output = self.se_bilinear_interaction(senet_output)
        combined = torch.cat([bilinear_output, se_bilinear_output], dim=1)
        dnn_output = self.dnn_layers(combined)
        output = torch.sigmoid(self.output_layer(dnn_output))
        
        return output.squeeze(1)


