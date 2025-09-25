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

class MovieLensDataset(Dataset):
    # 继承后需要重写len和getitem方法
    def __init__(self, ratings_file, train=True, test_size=0.2):

        self.ratings = pd.read_csv(ratings_file)
        print(len(self.ratings))
        
        # 维护用户和物品的映射
        self.user_ids = self.ratings['userId'].unique() # 取唯一id
        self.item_ids = self.ratings['movieId'].unique() 

        self.user_map = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        # 返回一个字典，key是原始user_id，value是从0开始的索引
        print(self.user_map)
        self.item_map = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        train_data, test_data = train_test_split(
            self.ratings, test_size=test_size, random_state=42
        )
        
        self.data = train_data if train else test_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_id = self.data.iloc[idx]['userId']
        item_id = self.data.iloc[idx]['movieId']
        rating = self.data.iloc[idx]['rating']
        
        # 转换为模型内部索引
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]
        
        sample = {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'item_idx': torch.tensor(item_idx, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float)
        }
        
        return sample
    
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
    





class MatrixCompletionModel(nn.Module):
    def __init__(self, n_users, n_items, emb_size=50):
        super(MatrixCompletionModel, self).__init__()
        # 结构很简单，就是两个嵌入层
        # 用户嵌入层
        self.user_embedding = nn.Embedding(n_users, emb_size)
        self.item_embedding = nn.Embedding(n_items, emb_size)
        
        # 初始化嵌入层参数 
        nn.init.xavier_uniform_(self.user_embedding.weight) # xavier初始化
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, user_idx, item_idx):
        """
        user_idx: 用户索引
        item_idx: 物品索引
        
        返回预测评分
        """
        # uid -> emb, iid -> emb
        user_embedding = self.user_embedding(user_idx)
        item_embedding = self.item_embedding(item_idx)
        
        prediction = torch.sum(user_embedding * item_embedding, dim=1) # inner product
        
        return prediction
    
    def get_user_embedding(self, user_idx):
        # uid -> emb
        return self.user_embedding(user_idx)
    
    def get_item_embedding(self, item_idx):
        return self.item_embedding(item_idx)

class MatrixCompletion:
    def __init__(self, 
                 path="ml-100k/ratings.csv", 
                 emb_size=50,
                 batch_size=64,
                 learning_rate=0.001,
                 weight_decay=0.00001,
                 n_epochs=20,
                 test_size=0.2):
        self.path = path
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.test_size = test_size
        self.train_dataset = None
        self.test_dataset = None
        self.model = None
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"使用计算设备: {self.device}")
        

    

    def load_data(self):
        self.train_dataset = MovieLensDataset(self.path, train=True, test_size=self.test_size)
        self.test_dataset = MovieLensDataset(self.path, train=False, test_size=self.test_size)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size)

        n_users, n_items = self.train_dataset.get_num_users_items()
        print(f"用户数: {n_users}, 物品数: {n_items}")
        
        self.model = MatrixCompletionModel(n_users, n_items, self.emb_size)
        self.model.to(self.device)
        
        return True
    
    def train(self):


        criterion = nn.MSELoss() # loss = (ground_truth - prediction)^2
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        train_losses = []
        test_losses = []
        
        
        for epoch in range(self.n_epochs):
            self.model.train()
            train_loss = 0.0 # 先随便定义一个变量
            
            for batch in self.train_dataloader:
                user_idx = batch['user_idx'].to(self.device)
                item_idx = batch['item_idx'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                # 前向传播
                predictions = self.model(user_idx, item_idx)
                
                # 计算损失
                loss = criterion(predictions, ratings)
                
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
                    ratings = batch['rating'].to(self.device)
                    
                    predictions = self.model(user_idx, item_idx)
                    loss = criterion(predictions, ratings)
                    
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
            'rmse': np.sqrt(test_losses[-1]),
        }
    

    def predict(self, user_id, item_id):
        user_map, item_map = self.train_dataset.get_user_item_maps()
        
  
        user_idx = user_map[user_id]
        item_idx = item_map[item_id]
        self.model.eval()
        with torch.no_grad():
            user_idx_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            item_idx_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
            
            prediction = self.model(user_idx_tensor, item_idx_tensor).item()
        
        return prediction
    
    def recommend_for_user(self, user_id, n_rec=10, exclude_rated=True):

        
        user_map, item_map = self.train_dataset.get_user_item_maps()
        
        user_idx = user_map[user_id]
        
        # 获取用户已评分物品
        user_ratings = self.train_dataset.ratings[self.train_dataset.ratings['userId'] == user_id]
        rated_items = set(user_ratings['movieId'])
        
        all_items = list(item_map.keys())
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            user_idx_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            user_embedding = self.model.get_user_embedding(user_idx_tensor)

            batch_size = 1024
            for i in range(0, len(all_items), batch_size):
                batch_items = all_items[i:i+batch_size]
                batch_item_idx = [item_map[item_id] for item_id in batch_items]
                
                # 过滤已评分物品
                if exclude_rated:
                    filtered_items = []
                    filtered_idx = []
                    for j, item_id in enumerate(batch_items):
                        if item_id not in rated_items:
                            filtered_items.append(item_id)
                            filtered_idx.append(batch_item_idx[j])
                    batch_items = filtered_items
                    batch_item_idx = filtered_idx
                

                
                batch_item_tensor = torch.tensor(batch_item_idx, dtype=torch.long).to(self.device)
                item_embeddings = self.model.get_item_embedding(batch_item_tensor)
                
                batch_scores = torch.mm(user_embedding, item_embeddings.t()).squeeze(0)
                
                for j, item_id in enumerate(batch_items):
                    predictions.append((item_id, batch_scores[j].item()))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前N个推荐
        return predictions[:n_rec]
    
    

if __name__ == "__main__":
    # 初始化矩阵补全算法
    mc = MatrixCompletion(
        path="ml-100k/ratings.csv",
        emb_size=50,
        batch_size=256,
        learning_rate=0.001,
        weight_decay=0.00001,
        n_epochs=20,
        test_size=0.2
    )
    
    # 加载数据
    mc.load_data()
    results = mc.train()
    print(f"训练结果: RMSE = {results['rmse']:.4f}")

    user_id = 1  
    recommendations = mc.recommend_for_user(user_id, n_rec=10)
    
    print(f"\n为用户 {user_id} 的推荐物品:")
    for item_id, score in recommendations:
        print(f"物品ID: {item_id}, 预测评分: {score:.4f}")
    
