import torch
import torch.nn as nn

class SENet(nn.Module):

    def __init__(self, num_fields, reduction_ratio=3):

        super(SENet, self).__init__()
        
        #先确保压缩后的维度至少为1
        reduced_size = max(1, num_fields // reduction_ratio)

        self.se_layers = nn.Sequential(
            nn.Linear(num_fields, reduced_size),  # 压缩
            nn.ReLU(),
            nn.Linear(reduced_size, num_fields),  # 恢复
            nn.Sigmoid()  # 归一化为权重
        )
    
    def forward(self, inputs):

        # 算平均值
        squeezed = torch.mean(inputs, dim=2)
        

        weights = self.se_layers(squeezed)  # (batch_size, num_fields)
        
        # 恢复维度
        weights = weights.unsqueeze(2)  # (batch_size, num_fields, 1)
        # 加权
        reweighted_inputs = inputs * weights  # (batch_size, num_fields, embedding_dim)
        
        return reweighted_inputs

