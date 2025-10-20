import torch
import torch.nn as nn
import torch.nn.functional as F

class LHUC_PPNet(nn.Module):

    def __init__(self, 
                 user_dim, 
                 item_dim, 
                 hidden_dims=[128, 64], 
                 dropout_rate=0.2):

        super(LHUC_PPNet, self).__init__()
        
        self.user_layers = nn.ModuleList()
        input_dim = user_dim
        
        for hidden_dim in hidden_dims:
            self.user_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.item_layers = nn.ModuleList()
        input_dim = item_dim
        
        for hidden_dim in hidden_dims:
            self.item_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.lhuc_layers = nn.ParameterList()
        for hidden_dim in hidden_dims:
            self.lhuc_layers.append(nn.Parameter(torch.zeros(hidden_dim)))

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_dims = hidden_dims
    
    def forward(self, user_features, item_features):

        user_repr = user_features
        item_repr = item_features

        for i in range(len(self.hidden_dims)):
            # 用户特征表示
            user_repr = F.relu(self.user_layers[i](user_repr))
            user_repr = self.dropout(user_repr)
            
            # 物品特征
            item_repr = F.relu(self.item_layers[i](item_repr))
            item_repr = self.dropout(item_repr)
            

            lhuc_weights = 2 * torch.sigmoid(self.lhuc_layers[i])
            
            # 用哈达玛积
            combined_repr = user_repr * item_repr * lhuc_weights
            user_repr = combined_repr
        output = self.output_layer(user_repr)
        
        return output.squeeze(1)


