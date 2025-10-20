import torch
import torch.nn as nn

class BilinearInteraction(nn.Module):

    def __init__(self, embedding_dim, num_fields, bilinear_type='field_interaction'):

        super(BilinearInteraction, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.bilinear_type = bilinear_type


        nn.init.xavier_normal_(self.bilinear_matrices)
    
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        
        if self.bilinear_type == 'field_all':
            bilinear_list = []
            for i in range(self.num_fields):
                # 获取第i个特征向量
                vi = inputs[:, i, :].unsqueeze(1)  # (batch_size, 1, embedding_dim)
                
                bilinear_i = torch.matmul(vi, self.bilinear_matrices[i])
                

                interaction = torch.matmul(bilinear_i, inputs.transpose(1, 2))
                bilinear_list.append(interaction)

            bilinear_output = torch.cat(bilinear_list, dim=1)  # (batch_size, num_fields, num_fields)
            bilinear_output = bilinear_output.view(batch_size, -1)  # (batch_size, num_fields*num_fields)
        
        else:  
            bilinear_list = []
            index = 0
            # 只计算不同特征域之间的交互i！=j
            for i in range(self.num_fields):
                for j in range(i+1, self.num_fields):
                    # 获取特征向量
                    vi = inputs[:, i, :].unsqueeze(1)  # (batch_size, 1, embedding_dim)
                    vj = inputs[:, j, :].unsqueeze(1)  # (batch_size, 1, embedding_dim)

                    bilinear_ij = torch.matmul(vi, self.bilinear_matrices[index])
                    
                    interaction = torch.matmul(bilinear_ij, vj.transpose(1, 2))
                    bilinear_list.append(interaction.squeeze(2))  # (batch_size, 1)
                    index += 1
            bilinear_output = torch.cat(bilinear_list, dim=1)  # (batch_size, num_interaction)
        
        return bilinear_output

