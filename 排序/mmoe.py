import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class MovieLensDataset(Dataset):
    def __init__(self, ratings_file, train=True, test_size=0.2):
        self.ratings = pd.read_csv(ratings_file)
        
        self.user_ids = self.ratings['userId'].unique()
        self.item_ids = self.ratings['movieId'].unique()
        
        self.user_map = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_map = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        train_data, test_data = train_test_split(
            self.ratings, test_size=test_size, random_state=42
        )
        
        self.data = train_data if train else test_data
        

        self.rating_mean = self.data['rating'].mean()
        self.rating_std = self.data['rating'].std()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        user_id = row['userId']
        item_id = row['movieId']
        rating = row['rating']
        

        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]

        normalized_rating = (rating - self.rating_mean) / self.rating_std

        click_label = 1.0 if rating >= 4.0 else 0.0
        
        sample = {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'item_idx': torch.tensor(item_idx, dtype=torch.long),
            'click_label': torch.tensor(click_label, dtype=torch.float),
            'rating_label': torch.tensor(normalized_rating, dtype=torch.float)
        }
        
        return sample
    
    def get_num_users_items(self):
        return len(self.user_ids), len(self.item_ids)
    
    def get_rating_stats(self):
        return self.rating_mean, self.rating_std

class Expert(nn.Module):

    def __init__(self, input_dim, hidden_dims):
        super(Expert, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.expert_net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.expert_net(x)

class MMoE(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=64, 
                 expert_hidden_dims=[128, 64], gate_hidden_dims=[64], 
                 task_hidden_dims=[64], num_experts=3, num_tasks=2):
        super(MMoE, self).__init__()
        
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        

        input_dim = embedding_dim * 2

        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dims) 
            for _ in range(num_experts)
        ])
        
        expert_output_dim = expert_hidden_dims[-1]
        
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, gate_hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(gate_hidden_dims[0], num_experts),
                nn.Softmax(dim=1)
            ) for _ in range(num_tasks)
        ])
        
        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_output_dim, task_hidden_dims[0]),
                nn.ReLU(),
                nn.BatchNorm1d(task_hidden_dims[0]),
                nn.Dropout(0.2),
                nn.Linear(task_hidden_dims[0], 1)
            ) for _ in range(num_tasks)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        # 嵌入层
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        for expert in self.experts:
            for module in expert.expert_net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # 初始化门控网络
        for gate_net in self.gate_networks:
            for i, layer in enumerate(gate_net):
                if isinstance(layer, nn.Linear):
                    if i == len(gate_net) - 2:  # 最后一层使用小的权重
                        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    else:
                        nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        for tower in self.task_towers:
            for layer in tower:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, user_idx, item_idx):


        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        

        features = torch.cat([user_emb, item_emb], dim=1)

        expert_outputs = [expert(features) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, expert_output_dim)

        task_outputs = []
        for task_id in range(self.num_tasks):
            gate_output = self.gate_networks[task_id](features)  # (batch_size, num_experts)
            
            gate_output = gate_output.unsqueeze(2)  # (batch_size, num_experts, 1)

            combined_experts = (expert_outputs * gate_output).sum(dim=1)  # (batch_size, expert_output_dim)

            task_output = self.task_towers[task_id](combined_experts)
            task_outputs.append(task_output)
        
        click_pred = torch.sigmoid(task_outputs[0]).squeeze(-1)
        rating_pred = task_outputs[1].squeeze(-1)  # 预测的是归一化评分
        
        return click_pred, rating_pred

class MMoERecommender:

    def __init__(self, 
                 path="ml-100k/ratings.csv", 
                 embedding_dim=64,
                 expert_hidden_dims=[128, 64], 
                 gate_hidden_dims=[64],
                 task_hidden_dims=[64],
                 num_experts=3,
                 batch_size=256,
                 learning_rate=0.001,
                 weight_decay=0.0001,
                 n_epochs=20,
                 test_size=0.2,
                 device=None):
        
        self.path = path
        self.embedding_dim = embedding_dim
        self.expert_hidden_dims = expert_hidden_dims
        self.gate_hidden_dims = gate_hidden_dims
        self.task_hidden_dims = task_hidden_dims
        self.num_experts = num_experts
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.test_size = test_size
        
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
            test_size=self.test_size
        )
        self.test_dataset = MovieLensDataset(
            self.path, 
            train=False, 
            test_size=self.test_size
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
        self.model = MMoE(
            num_users=n_users, 
            num_items=n_items,
            embedding_dim=self.embedding_dim,
            expert_hidden_dims=self.expert_hidden_dims,
            gate_hidden_dims=self.gate_hidden_dims,
            task_hidden_dims=self.task_hidden_dims,
            num_experts=self.num_experts
        )
        self.model.to(self.device)
        
        return True
    
    def train(self):

        # 两个任务的损失函数
        click_criterion = nn.BCELoss()
        rating_criterion = nn.MSELoss()
        
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        

        train_click_losses = []
        train_rating_losses = []
        test_click_losses = []
        test_rating_losses = []
        train_total_losses = []
        test_total_losses = []
        
        # 两个任务的权重
        click_weight = 0.5
        rating_weight = 0.5
        
        for epoch in range(self.n_epochs):
            self.model.train()
            train_click_loss = 0.0
            train_rating_loss = 0.0
            
            for batch in self.train_dataloader:
                user_idx = batch['user_idx'].to(self.device)
                item_idx = batch['item_idx'].to(self.device)
                click_labels = batch['click_label'].to(self.device)
                rating_labels = batch['rating_label'].to(self.device)
                
                click_preds, rating_preds = self.model(user_idx, item_idx)
                
                click_loss = click_criterion(click_preds, click_labels)
                rating_loss = rating_criterion(rating_preds, rating_labels)
                
                total_loss = click_weight * click_loss + rating_weight * rating_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # 累计损失
                train_click_loss += click_loss.item() * user_idx.size(0)
                train_rating_loss += rating_loss.item() * user_idx.size(0)

            train_click_loss /= len(self.train_dataloader.dataset)
            train_rating_loss /= len(self.train_dataloader.dataset)
            train_total_loss = click_weight * train_click_loss + rating_weight * train_rating_loss
            
            train_click_losses.append(train_click_loss)
            train_rating_losses.append(train_rating_loss)
            train_total_losses.append(train_total_loss)

            self.model.eval()
            test_click_loss = 0.0
            test_rating_loss = 0.0
            
            with torch.no_grad():
                for batch in self.test_dataloader:
                    user_idx = batch['user_idx'].to(self.device)
                    item_idx = batch['item_idx'].to(self.device)
                    click_labels = batch['click_label'].to(self.device)
                    rating_labels = batch['rating_label'].to(self.device)
                    
                    click_preds, rating_preds = self.model(user_idx, item_idx)
                    
                    click_loss = click_criterion(click_preds, click_labels)
                    rating_loss = rating_criterion(rating_preds, rating_labels)
                    
                    test_click_loss += click_loss.item() * user_idx.size(0)
                    test_rating_loss += rating_loss.item() * user_idx.size(0)
                
                test_click_loss /= len(self.test_dataloader.dataset)
                test_rating_loss /= len(self.test_dataloader.dataset)
                test_total_loss = click_weight * test_click_loss + rating_weight * test_rating_loss
                
                test_click_losses.append(test_click_loss)
                test_rating_losses.append(test_rating_loss)
                test_total_losses.append(test_total_loss)
            
            # 打印训练进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.n_epochs}], "
                      f"Train: [Click: {train_click_loss:.4f}, Rating: {train_rating_loss:.4f}, Total: {train_total_loss:.4f}], "
                      f"Test: [Click: {test_click_loss:.4f}, Rating: {test_rating_loss:.4f}, Total: {test_total_loss:.4f}]")
        

        
        return {
            'train_click_loss': train_click_losses[-1],
            'train_rating_loss': train_rating_losses[-1],
            'train_total_loss': train_total_losses[-1],
            'test_click_loss': test_click_losses[-1],
            'test_rating_loss': test_rating_losses[-1],
            'test_total_loss': test_total_losses[-1]
        }
    


if __name__ == "__main__":
    mmoe = MMoERecommender(
        path="ml-100k/ratings.csv",
        embedding_dim=64,
        expert_hidden_dims=[128, 64],
        gate_hidden_dims=[64],
        task_hidden_dims=[64],
        num_experts=3,
        batch_size=256,
        learning_rate=0.001,
        weight_decay=0.0001,
        n_epochs=20
    )
    
    if mmoe.load_data():
        results = mmoe.train()
        print(f"训练集点击损失: {results['train_click_loss']:.4f}")
        print(f"训练集评分损失: {results['train_rating_loss']:.4f}")
        print(f"测试集点击损失: {results['test_click_loss']:.4f}")
        print(f"测试集评分损失: {results['test_rating_loss']:.4f}")
        
        # 为指定用户推荐物品
        user_id = 1
        recommendations = mmoe.recommend_for_user(user_id, top_n=10, recommendation_type='hybrid')
        
        print(f"\n为用户 {user_id} 的推荐物品:")
        for i, rec in enumerate(recommendations):
            print(f"{i+1}. 物品ID: {rec['item_id']}, 点击概率: {rec['click_prob']:.4f}, 预测评分: {rec['rating']:.2f}")
