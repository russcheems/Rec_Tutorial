import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionLayer(nn.Module):
    # 论文里的激活函数似乎并不是直接用的relu
    def __init__(self, embedding_dim, attention_hidden_units=[80, 40]):
        super(AttentionLayer, self).__init__()
     
        layers = []
        input_dim = embedding_dim * 4  # [query, key, query*key, query-key]
        
        for hidden_dim in attention_hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.attention_net = nn.Sequential(*layers)
    
    def forward(self, query, keys, keys_length):

        batch_size, max_seq_len, dim = keys.shape
        queries = query.unsqueeze(1).expand(-1, max_seq_len, -1)  # (batch_size, max_seq_len, embedding_dim)
        product = queries * keys
        difference = queries - keys
        
        attention_input = torch.cat([queries, keys, product, difference], dim=-1)  # (batch_size, max_seq_len, embedding_dim*4)
        attention_input = attention_input.view(-1, attention_input.size(-1))  # (batch_size * max_seq_len, embedding_dim*4)
        
        # 注意力得分
        attention_scores = self.attention_net(attention_input)  # (batch_size * max_seq_len, 1)
        attention_scores = attention_scores.view(batch_size, max_seq_len)  # (batch_size, max_seq_len)
        mask = torch.arange(max_seq_len, device=keys_length.device).unsqueeze(0) < keys_length.unsqueeze(1)
        attention_scores = attention_scores * mask.float()

        paddings = torch.ones_like(attention_scores) * (-2**32 + 1)
        attention_scores = torch.where(mask, attention_scores, paddings)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, max_seq_len)
        
        weighted_keys = keys * attention_weights.unsqueeze(-1)  # (batch_size, max_seq_len, embedding_dim)
        
        interest_representation = weighted_keys.sum(dim=1)  # (batch_size, embedding_dim)
        
        return interest_representation, attention_weights


class DIN(nn.Module):
    def __init__(self, 
                 num_users, 
                 num_items, 
                 embedding_dim=64, 
                 mlp_hidden_units=[200, 80],
                 attention_hidden_units=[80, 40],
                 dropout_rate=0.2,
                 max_seq_len=50):
        super(DIN, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.attention_layer = AttentionLayer(embedding_dim, attention_hidden_units)
        mlp_layers = []
        mlp_input_dim = embedding_dim * 3
        
        for hidden_dim in mlp_hidden_units:
            mlp_layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.Dropout(dropout_rate))
            mlp_input_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(mlp_input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, user_ids, item_ids, history_item_ids, history_length):
        """
            history_item_ids: (batch_size, max_seq_len)
            history_length: (batch_size,) 
        """
        user_emb = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        target_item_emb = self.item_embedding(item_ids)  # (batch_size, embedding_dim)

        history_emb = self.item_embedding(history_item_ids)  # (batch_size, max_seq_len, embedding_dim)
        interest_emb, attention_weights = self.attention_layer(target_item_emb, history_emb, history_length)
        
        concat_feature = torch.cat([user_emb, target_item_emb, interest_emb], dim=1)
        
        output = self.mlp(concat_feature)
        prediction = torch.sigmoid(output).squeeze(1)
        
        return prediction, attention_weights # 点击率+权重



