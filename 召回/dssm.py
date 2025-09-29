import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import os
import random

class MovieLensDataset(Dataset):
    # 继承后需要重写len和getitem方法
    def __init__(self, ratings_file,
                 train=True,
                 test_size=0.2, 
                 negative_sampling=3):
        # 先用pointwise，1个正样本对应3个负样本
        self.ratings = pd.read_csv(ratings_file)
        self.negative_sampling = negative_sampling
        
        # 维护用户和物品的映射
        self.user_ids = self.ratings['userId'].unique() # 取唯一id
        self.item_ids = self.ratings['movieId'].unique() 

        self.user_map = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        # 返回一个字典，key是原始user_id，value是从0开始的索引
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
        # 要加上负样本的数量
        return len(self.data) * (1 + self.negative_sampling)
    
    def __getitem__(self, idx):
        # 确定是正样本还是负样本
        pos_idx = idx // (1 + self.negative_sampling)
        is_positive = idx % (1 + self.negative_sampling) == 0 # 每个正样本后面跟negative_sampling个负样本
        
        user_id = self.data.iloc[pos_idx]['userId']
        
        if is_positive:
            item_id = self.data.iloc[pos_idx]['movieId']
            label = 1.0
        else:
            # 负样本随机选择用户未交互过的物品
            pos_item_id = self.data.iloc[pos_idx]['movieId']
            item_id = self.sample_negative(user_id, pos_item_id)
            label = 0.0
        
        # 转换为模型内部索引
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]
        
        sample = {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'item_idx': torch.tensor(item_idx, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }
        
        return sample
    
    def sample_negative(self, user_id, pos_item_id):
        """采样用户未交互的物品作为负样本"""
        user_pos_items = self.user_items.get(user_id, set())
        
        while True:
            # 随机选择一个物品
            neg_item_id = random.choice(self.item_ids)
            # 确保是用户未交互过的，且不是当前正样本
            if neg_item_id != pos_item_id and neg_item_id not in user_pos_items:
                return neg_item_id
    
    def get_num_users_items(self):
        return len(self.user_ids), len(self.item_ids)
    
    def get_user_item_maps(self):
        return self.user_map, self.item_map
    
    def get_original_ids(self, user_idx=None, item_idx=None):
        # 通过反向映射获取原始ID
        inv_user_map = {v: k for k, v in self.user_map.items()}
        inv_item_map = {v: k for k, v in self.item_map.items()}
        
        user_id = inv_user_map[user_idx] if user_idx is not None else None
        item_id = inv_item_map[item_idx] if item_idx is not None else None
        return user_id, item_id
        
    
class DSSM(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128, hidden_layers=[512, 256]):

        super(DSSM, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 用户塔先用mlp
        user_layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_layers:
            user_layers.append(nn.Linear(input_dim, hidden_dim))
            user_layers.append(nn.ReLU())
            user_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        self.user_tower = nn.Sequential(*user_layers)
        
        # 物品塔
        item_layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_layers:
            item_layers.append(nn.Linear(input_dim, hidden_dim))
            item_layers.append(nn.ReLU())
            item_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        self.item_tower = nn.Sequential(*item_layers)
        
        self.output_dim = hidden_layers[-1] if hidden_layers else embedding_dim

        nn.init.normal_(self.user_embedding.weight, std=0.01) # 这里能不能用xavier
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # MLP层初始化
        for layer in self.user_tower:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        for layer in self.item_tower:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    

    
    def forward(self, user_idx, item_idx):

        # 获取嵌入向量
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        
        # 通过两个塔得到用户和物品的表示
        user_vector = self.user_tower(user_emb)
        item_vector = self.item_tower(item_emb)
        
        # 计算cossim
        user_vector = nn.functional.normalize(user_vector, p=2, dim=1) # p=2表示L2范数
        item_vector = nn.functional.normalize(item_vector, p=2, dim=1)

        cosine_sim = torch.sum(user_vector * item_vector, dim=1)
        
        return cosine_sim
    
    def get_user_embedding(self, user_idx):
        user_emb = self.user_embedding(user_idx)
        user_vector = self.user_tower(user_emb)
        return nn.functional.normalize(user_vector, p=2, dim=1)
    
    def get_item_embedding(self, item_idx):

        item_emb = self.item_embedding(item_idx)
        item_vector = self.item_tower(item_emb)
        return nn.functional.normalize(item_vector, p=2, dim=1)


class DSSMRecommender:
    def __init__(self, 
                 path="ml-100k/ratings.csv", 
                 embedding_dim=128,
                 hidden_layers=[512, 256],
                 batch_size=256,
                 learning_rate=0.001,
                 weight_decay=0.0001,
                 n_epochs=20,
                 test_size=0.2,
                 negative_sampling=4,
                 device=None):

        self.path = path
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
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
            else:
                self.device = torch.device("cpu")
        
        self.train_dataset = None
        self.test_dataset = None
        self.model = None
    
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
        self.model = DSSM(
            num_users=n_users, 
            num_items=n_items,
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers
        )
        self.model.to(self.device)
        
        return True
    
    def train(self):
        
        # 用二元交叉熵
        criterion = nn.BCEWithLogitsLoss()
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

        

        user_map, item_map = self.train_dataset.get_user_item_maps()
        user_idx = user_map[user_id]
        
        user_ratings = self.train_dataset.user_items.get(user_id, set())
        
        # 所有物品ID列表
        all_items = list(item_map.keys())
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            user_idx_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            user_embedding = self.model.get_user_embedding(user_idx_tensor)
            
            # 批量预测所有物品
            batch_size = 1024 # 这里batch_size的大小
            for i in range(0, len(all_items), batch_size):
                batch_items = all_items[i:i+batch_size]
                

                if exclude_rated:
                    batch_items = [item_id for item_id in batch_items if item_id not in user_ratings]
                
                if not batch_items:  # 如果没有物品（所有物品都已评分）
                    continue
                
                batch_item_idx = [item_map[item_id] for item_id in batch_items]
                
                batch_item_tensor = torch.tensor(batch_item_idx, dtype=torch.long).to(self.device)
                item_embeddings = self.model.get_item_embedding(batch_item_tensor)
                batch_scores = torch.mm(user_embedding, item_embeddings.t()).squeeze(0)
                
                for j, item_id in enumerate(batch_items):
                    predictions.append((item_id, batch_scores[j].item()))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_rec]
    

    
if __name__ == "__main__":
    dssm = DSSMRecommender(
        path="ml-100k/ratings.csv",
        embedding_dim=128,
        hidden_layers=[512, 256],
        batch_size=256,
        learning_rate=0.001,
        weight_decay=0.0001,
        n_epochs=20,
        test_size=0.2,
        negative_sampling=4
    )

    if dssm.load_data():
        results = dssm.train()
        print(f"Train Loss = {results['train_loss']:.4f}, Test Loss = {results['test_loss']:.4f}")
        
        
        user_id = 1  
        recommendations = dssm.recommend_for_user(user_id, n_rec=10)
        
        print(f"\n为用户 {user_id} 的推荐物品:")
        for item_id, score in recommendations:
            print(f"物品ID: {item_id}, 相似度: {score:.4f}")
        
    else:
        print(111)