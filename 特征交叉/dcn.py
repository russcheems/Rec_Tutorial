import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import random

class MovieLensDataset(Dataset):

    def __init__(self, ratings_file, train=True, test_size=0.2, negative_sampling=3):
        # 加载数据
        self.ratings = pd.read_csv(ratings_file)
        self.negative_sampling = negative_sampling
        
        # 维护用户和物品的映射
        self.user_ids = self.ratings['userId'].unique()
        self.item_ids = self.ratings['movieId'].unique()
        
        self.user_map = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_map = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        # 构建用户-物品交互字典，用于负采样
        self.user_items = {}
        for _, row in self.ratings.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            if user_id not in self.user_items:
                self.user_items[user_id] = set()
            self.user_items[user_id].add(item_id)
        
        # 分割训练集和测试集
        train_data, test_data = train_test_split(
            self.ratings, test_size=test_size, random_state=42
        )
        
        self.data = train_data if train else test_data
    
    def __len__(self):
        return len(self.data) * (1 + self.negative_sampling)
    
    def __getitem__(self, idx):

        pos_idx = idx // (1 + self.negative_sampling)
        is_positive = idx % (1 + self.negative_sampling) == 0
        
        user_id = self.data.iloc[pos_idx]['userId']
        
        if is_positive:
            item_id = self.data.iloc[pos_idx]['movieId']
            rating = self.data.iloc[pos_idx]['rating']
            label = 1.0
        else:
            # 负样本随机选择用户未交互过的物品
            pos_item_id = self.data.iloc[pos_idx]['movieId']
            item_id = self.sample_negative(user_id, pos_item_id)
            rating = 0.0
            label = 0.0
        
        # 转换为模型内部索引
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]
        

        
        sample = {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'item_idx': torch.tensor(item_idx, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.float)
        }
        
        return sample
    
    def sample_negative(self, user_id, pos_item_id):
        user_pos_items = self.user_items.get(user_id, set())
        
        while True:
            # 随机选择一个物品
            neg_item_id = random.choice(self.item_ids)
            # 确保是用户未交互过的，且不是当前正样本
            if neg_item_id != pos_item_id and neg_item_id not in user_pos_items:
                return neg_item_id
    
    def get_num_users_items(self):
        return len(self.user_ids), len(self.item_ids)


class CrossNetwork(nn.Module):

    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.weights = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        """
        计算交叉网络的输出
        x_0: 初始输入
        x_l+1 = x_0 * x_l^T * w_l + b_l + x_l
        """
        x0 = x
        xl = x
        
        for i in range(self.num_layers):
            xl_w = self.weights[i](xl)  # (batch_size, 1)
            xl = x0 * xl_w + self.biases[i] + xl
        
        return xl


class DeepNetwork(nn.Module):
    """
    深度网络：捕获特征的非线性组合
    """
    def __init__(self, input_dim, hidden_dims):
        super(DeepNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.deep_network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.deep_network(x)


class DCN(nn.Module):
    """
    Deep & Cross Network：结合深度学习和特征交叉
    """
    def __init__(self, num_users, num_items, embedding_dim=64, 
                 num_cross_layers=3, deep_hidden_dims=[128, 64]):
        super(DCN, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 输入维度：用户嵌入 + 物品嵌入
        total_input_dim = embedding_dim * 2
        
        # 交叉网络
        self.cross_network = CrossNetwork(total_input_dim, num_cross_layers)
        
        # 深度网络
        self.deep_network = DeepNetwork(total_input_dim, deep_hidden_dims)
        
        # 组合层：Cross网络输出 + Deep网络输出
        deep_output_dim = deep_hidden_dims[-1] if deep_hidden_dims else 0
        
        # 最后的预测层
        self.prediction_layer = nn.Linear(total_input_dim + deep_output_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        # 初始化嵌入层
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # 初始化预测层
        nn.init.xavier_normal_(self.prediction_layer.weight)
        nn.init.zeros_(self.prediction_layer.bias)
    
    def forward(self, user_idx, item_idx):
        # 获取嵌入
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        
        # 拼接特征
        x = torch.cat([user_emb, item_emb], dim=1)
        
        # 交叉网络
        cross_output = self.cross_network(x)
        
        # 深度网络
        deep_output = self.deep_network(x)
        
        # 组合交叉网络和深度网络的输出
        final_input = torch.cat([cross_output, deep_output], dim=1)
        
        # 最终预测
        prediction = self.prediction_layer(final_input)
        
        return torch.sigmoid(prediction).squeeze(1)


class DCNRecommender:
    """
    DCN推荐系统封装类
    """
    def __init__(self, 
                 path="ml-100k/ratings.csv", 
                 embedding_dim=64,
                 num_cross_layers=3,
                 deep_hidden_dims=[128, 64],
                 batch_size=256,
                 learning_rate=0.001,
                 weight_decay=0.0001,
                 n_epochs=20,
                 test_size=0.2,
                 negative_sampling=4,
                 device=None):
        
        self.path = path
        self.embedding_dim = embedding_dim
        self.num_cross_layers = num_cross_layers
        self.deep_hidden_dims = deep_hidden_dims
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.test_size = test_size
        self.negative_sampling = negative_sampling
        
        if device:
            self.device = device
        else:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        
        self.train_dataset = None
        self.test_dataset = None
        self.model = None
        print(f"Using device: {self.device}")
    
    def load_data(self):
        self.train_dataset = MovieLensDataset(
            self.path, 
            train=True, 
            test_size=self.test_size,
            negative_sampling=self.negative_sampling
        )
        self.test_dataset = MovieLensDataset(
            self.path, 
            train=False, 
            test_size=self.test_size,
            negative_sampling=self.negative_sampling
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size
        )
        
        n_users, n_items = self.train_dataset.get_num_users_items()
        self.model = DCN(
            num_users=n_users, 
            num_items=n_items,
            embedding_dim=self.embedding_dim,
            num_cross_layers=self.num_cross_layers,
            deep_hidden_dims=self.deep_hidden_dims
        )
        self.model.to(self.device)
        
        return True
    
    def train(self):
        """训练模型"""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        train_losses = []
        test_losses = []
        
        for epoch in range(self.n_epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch in self.train_dataloader:
                # 获取数据
                user_idx = batch['user_idx'].to(self.device)
                item_idx = batch['item_idx'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                outputs = self.model(user_idx, item_idx)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * user_idx.size(0)
            
            # 计算平均训练损失
            train_loss /= len(self.train_dataloader.dataset)
            train_losses.append(train_loss)
            
            self.model.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for batch in self.test_dataloader:
                    user_idx = batch['user_idx'].to(self.device)
                    item_idx = batch['item_idx'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(user_idx, item_idx)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item() * user_idx.size(0)
                
                test_loss /= len(self.test_dataloader.dataset)
                test_losses.append(test_loss)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.n_epochs}], "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Test Loss: {test_loss:.4f}")
        

        
        return {
            'train_loss': train_losses[-1],
            'test_loss': test_losses[-1],
        }
    
    def recommend_for_user(self, user_id, n_rec=10, exclude_rated=True):
        user_map, item_map = self.train_dataset.user_map, self.train_dataset.item_map
        
        if user_id not in user_map:
            return []
        
        user_idx = user_map[user_id]
        
        user_rated_items = self.train_dataset.user_items.get(user_id, set())
        
        all_items = list(item_map.keys())
        if exclude_rated:
            all_items = [item_id for item_id in all_items if item_id not in user_rated_items]
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            batch_size = 512
            for i in range(0, len(all_items), batch_size):
                batch_items = all_items[i:i+batch_size]
                batch_item_indices = [item_map[item_id] for item_id in batch_items]
                
                batch_user_indices = [user_idx] * len(batch_items)
                
                user_tensor = torch.tensor(batch_user_indices, dtype=torch.long).to(self.device)
                item_tensor = torch.tensor(batch_item_indices, dtype=torch.long).to(self.device)

                scores = self.model(user_tensor, item_tensor)

                for j, item_id in enumerate(batch_items):
                    predictions.append((item_id, scores[j].item()))

        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_rec]
    



if __name__ == "__main__":
    # 使用DCN进行推荐
    dcn = DCNRecommender(
        path="ml-100k/ratings.csv",
        embedding_dim=64,
        num_cross_layers=3,
        deep_hidden_dims=[128, 64],
        batch_size=256,
        learning_rate=0.001,
        weight_decay=0.0001,
        n_epochs=20,
        test_size=0.2,
        negative_sampling=4
    )
    
    if dcn.load_data():
        results = dcn.train()
        print(f"Train Loss = {results['train_loss']:.4f}, Test Loss = {results['test_loss']:.4f}")
        
        # 为指定用户推荐物品
        user_id = 1
        recommendations = dcn.recommend_for_user(user_id, n_rec=10)
        
        print(f"\n为用户 {user_id} 的推荐物品:")
        for i, (item_id, score) in enumerate(recommendations):
            print(f"{i+1}. 物品ID: {item_id}, 预测得分: {score:.4f}")
        

