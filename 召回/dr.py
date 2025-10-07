import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random

class MovieLensDataset(Dataset):
    def __init__(self, ratings_file,
                 train=True,
                 test_size=0.2,
                 num_paths_per_item=3,  # 每个物品对应的路径数量
                 path_length=3):        # 每条路径的节点数量
        
        self.ratings = pd.read_csv(ratings_file)
        self.num_paths_per_item = num_paths_per_item
        self.path_length = path_length
        
        # 维护用户和物品的映射
        self.user_ids = self.ratings['userId'].unique() 
        self.item_ids = self.ratings['movieId'].unique() 
        
        self.user_map = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_map = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        # 构建用户-物品交互字典
        self.user_items = {}
        for _, row in self.ratings.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            if user_id not in self.user_items:
                self.user_items[user_id] = set()
            self.user_items[user_id].add(item_id)

        train_data, test_data = train_test_split(
            self.ratings, test_size=test_size, random_state=42
        )
        
        self.data = train_data if train else test_data
        

        self.num_nodes_per_level = 100  # 每一层节点的总数
        self.total_nodes = self.num_nodes_per_level * self.path_length
        
        self.generate_path_mappings()
        
    def generate_path_mappings(self):
        self.item_to_paths = {}  # 物品到路径的映射
        self.path_to_items = {}  # 路径到物品的映射
        
        for item_id in self.item_ids:
            self.item_to_paths[item_id] = []
            
            # 为每个物品生成多条路径
            for _ in range(self.num_paths_per_item):
                path = []
                # 对于每一层，随机选择一个节点
                for level in range(self.path_length):
                    node_id = random.randint(0, self.num_nodes_per_level - 1) + level * self.num_nodes_per_level # 确保不同层的节点ID不重复
                    path.append(node_id)
                
                # 将路径转换为元组以便作为字典键
                path_tuple = tuple(path)
                self.item_to_paths[item_id].append(path_tuple)
                
                # 更新路径到物品的映射
                if path_tuple not in self.path_to_items:
                    self.path_to_items[path_tuple] = []
                self.path_to_items[path_tuple].append(item_id)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_id = self.data.iloc[idx]['userId']
        item_id = self.data.iloc[idx]['movieId']
        
        user_idx = self.user_map[user_id]
        
        item_paths = self.item_to_paths[item_id]
    
        target_path = random.choice(item_paths)
        
        sample = {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'target_path': torch.tensor(target_path, dtype=torch.long),
            'item_id': item_id
        }
        
        return sample
    
    def get_num_users_nodes(self):
        return len(self.user_ids), self.total_nodes
    
    def get_user_item_maps(self):
        return self.user_map, self.item_map
    
    def get_original_ids(self, user_idx=None):
        inv_user_map = {v: k for k, v in self.user_map.items()}
        user_id = inv_user_map[user_idx] if user_idx is not None else None
        return user_id

    def get_item_to_paths(self):
        return self.item_to_paths
    
    def get_path_to_items(self):
        return self.path_to_items


class DeepRetrieval(nn.Module):
    def __init__(self, num_users, num_nodes, embedding_dim=128, hidden_dims=[256, 128]):
        super(DeepRetrieval, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # 第一层网络，输入用户特征，输出第一层节点的概率分布
        self.layer1 = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], num_nodes // 3)  # 第一层的节点数量
        )
        
        # 第二层网络，输入用户特征 + 第一层节点嵌入，输出第二层节点的概率分布
        self.layer2 = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], num_nodes // 3)  # 第二层的节点数量
        )
        
        # 一样
        self.layer3 = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], num_nodes // 3)  
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.node_embedding.weight, std=0.01)
        
        for layer in [self.layer1, self.layer2, self.layer3]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self, user_idx, target_paths=None, beam_size=5):

        batch_size = user_idx.size(0)

        user_emb = self.user_embedding(user_idx)  # [batch_size, embedding_dim]
        
        if self.training and target_paths is not None:
            level1_logits = self.layer1(user_emb)  
            
            level1_nodes = target_paths[:, 0]  # [batch_size]
            level1_emb = self.node_embedding(level1_nodes)  # [batch_size, embedding_dim]
            
            # emb_u concat emb_1
            level2_input = torch.cat([user_emb, level1_emb], dim=1)  # [batch_size, embedding_dim*2]
            level2_logits = self.layer2(level2_input)  # [batch_size, num_nodes//3]
            

            level2_nodes = target_paths[:, 1]  
            level2_emb = self.node_embedding(level2_nodes)  # [batch_size, embedding_dim]
            
    
            level3_input = torch.cat([user_emb, level1_emb, level2_emb], dim=1)  # [batch_size, embedding_dim*3]
            level3_logits = self.layer3(level3_input)  # [batch_size, num_nodes//3]
            

            level1_indices = level1_nodes % (self.node_embedding.num_embeddings // 3)
            level2_indices = level2_nodes % (self.node_embedding.num_embeddings // 3)
            level3_indices = target_paths[:, 2] % (self.node_embedding.num_embeddings // 3)
            
            level1_probs = torch.softmax(level1_logits, dim=1)
            level1_log_probs = torch.log(level1_probs.gather(1, level1_indices.unsqueeze(1)).squeeze(1) + 1e-10)
            
            level2_probs = torch.softmax(level2_logits, dim=1)
            level2_log_probs = torch.log(level2_probs.gather(1, level2_indices.unsqueeze(1)).squeeze(1) + 1e-10)
            
            level3_probs = torch.softmax(level3_logits, dim=1)
            level3_log_probs = torch.log(level3_probs.gather(1, level3_indices.unsqueeze(1)).squeeze(1) + 1e-10)
            
            path_log_probs = level1_log_probs + level2_log_probs + level3_log_probs
            
            return {
                'path_log_probs': path_log_probs,
                'level1_logits': level1_logits,
                'level2_logits': level2_logits,
                'level3_logits': level3_logits
            }
        
        else:
            return self._beam_search_inference(user_emb, beam_size)
    
    def _beam_search_inference(self, user_emb, beam_size=5):
        batch_size = user_emb.size(0)
        device = user_emb.device
        num_nodes_per_level = self.node_embedding.num_embeddings // 3
        
        level1_logits = self.layer1(user_emb)  # [batch_size, num_nodes//3]
        level1_probs = torch.softmax(level1_logits, dim=1)  # [batch_size, num_nodes//3]
        
        # 每个用户选择 beam_size 个最可能的节点
        level1_top_probs, level1_top_indices = torch.topk(level1_probs, beam_size, dim=1)

        level1_top_indices_abs = level1_top_indices + 0 * num_nodes_per_level
        
        # 为每个 batch 中的每个路径创建占位符
        batch_paths = []
        batch_probs = []
        
        for b in range(batch_size):
            paths = []
            path_probs = []
            

            for i in range(beam_size):
                # 获取第一层节点的嵌入
                node1_idx = level1_top_indices_abs[b, i]
                node1_emb = self.node_embedding(node1_idx)
                
                level2_input = torch.cat([user_emb[b:b+1], node1_emb.unsqueeze(0)], dim=1)
                level2_logits = self.layer2(level2_input)
                level2_probs = torch.softmax(level2_logits, dim=1)
                
                level2_top_probs, level2_top_indices = torch.topk(level2_probs, beam_size, dim=1)
                
                level2_top_indices_abs = level2_top_indices + 1 * num_nodes_per_level
                

                for j in range(beam_size):
                    # 获取第二层节点的嵌入
                    node2_idx = level2_top_indices_abs[0, j]
                    node2_emb = self.node_embedding(node2_idx)
                    
                    # 计算第三层的输入
                    level3_input = torch.cat([user_emb[b:b+1], node1_emb.unsqueeze(0), node2_emb.unsqueeze(0)], dim=1)
                    level3_logits = self.layer3(level3_input)
                    level3_probs = torch.softmax(level3_logits, dim=1)
                    
                    level3_top_probs, level3_top_indices = torch.topk(level3_probs, beam_size, dim=1)
                    
                    level3_top_indices_abs = level3_top_indices + 2 * num_nodes_per_level

                    for k in range(beam_size):
                        node3_idx = level3_top_indices_abs[0, k]
                        
                        path = [node1_idx.item(), node2_idx.item(), node3_idx.item()]
                        # 乘积
                        path_prob = level1_top_probs[b, i] * level2_top_probs[0, j] * level3_top_probs[0, k]
                        
                        paths.append(path)
                        path_probs.append(path_prob.item())
            
            path_probs_tensor = torch.tensor(path_probs, device=device)
            top_k_probs, top_k_indices = torch.topk(path_probs_tensor, min(beam_size, len(paths)), dim=0) # 这里能不能直接用topk
            
            top_paths = [paths[idx] for idx in top_k_indices]
            top_probs = top_k_probs.tolist()
            
            batch_paths.append(top_paths)
            batch_probs.append(top_probs)
        
        return {
            'paths': batch_paths,  # [batch_size, beam_size, path_length]
            'probs': batch_probs   #batch_size, beam_size
        }


class DeepRetrievalRecommender:
    def __init__(self, 
                 path="ml-100k/ratings.csv",
                 embedding_dim=128,
                 hidden_dims=[256, 128],
                 batch_size=256,
                 learning_rate=0.001,
                 weight_decay=0.0001,
                 n_epochs=20,
                 test_size=0.2,
                 num_paths_per_item=3,
                 path_length=3,
                 device=None):
        
        self.path = path
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.test_size = test_size
        self.num_paths_per_item = num_paths_per_item
        self.path_length = path_length
        
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
            num_paths_per_item=self.num_paths_per_item,
            path_length=self.path_length
        )
        self.test_dataset = MovieLensDataset(
            self.path, 
            train=False, 
            test_size=self.test_size,
            num_paths_per_item=self.num_paths_per_item,
            path_length=self.path_length
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
        
        n_users, n_nodes = self.train_dataset.get_num_users_nodes()
        self.model = DeepRetrieval(
            num_users=n_users, 
            num_nodes=n_nodes,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims
        )
        self.model.to(self.device)
        
        return True
    
    def train(self):
        # 使用负对数似然损失
        criterion = nn.NLLLoss()
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

                user_idx = batch['user_idx'].to(self.device)
                target_path = batch['target_path'].to(self.device)

                outputs = self.model(user_idx, target_path)
                path_log_probs = outputs['path_log_probs']
                

                loss = -path_log_probs.mean()
                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * user_idx.size(0)
            

            train_loss /= len(self.train_dataloader.dataset)
            train_losses.append(train_loss)
            
            self.model.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for batch in self.test_dataloader:
                    user_idx = batch['user_idx'].to(self.device)
                    target_path = batch['target_path'].to(self.device)
                    
                    outputs = self.model(user_idx, target_path)
                    path_log_probs = outputs['path_log_probs']
                    
                    loss = -path_log_probs.mean()
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
    
    def recommend_for_user(self, user_id, n_rec=10, beam_size=5):

        user_map, _ = self.train_dataset.get_user_item_maps()
        user_idx = user_map[user_id]
        
        user_ratings = self.train_dataset.user_items.get(user_id, set())
        
        path_to_items = self.train_dataset.get_path_to_items()
        
        self.model.eval()
        
        with torch.no_grad():
            user_idx_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            
            path_results = self.model(user_idx_tensor, beam_size=beam_size)
            
            paths = path_results['paths'][0]  # 第一个用户的路径
            probs = path_results['probs'][0]  # 第一个用户的路径概率
            

            item_scores = {}
            
            for i, path in enumerate(paths):
                path_prob = probs[i]
                path_tuple = tuple(path)

                if path_tuple in path_to_items:
                    items = path_to_items[path_tuple]
                    for item_id in items:
                        if item_id not in user_ratings:  # 排除已评分物品
                            if item_id not in item_scores:
                                item_scores[item_id] = 0.0
                            item_scores[item_id] += path_prob
            
            # 按分数排序物品
            ranked_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n_rec]
            
            return ranked_items

if __name__ == "__main__":
    dr = DeepRetrievalRecommender(
        path="ml-100k/ratings.csv",
        embedding_dim=128,
        hidden_dims=[256, 128],
        batch_size=256,
        learning_rate=0.001,
        weight_decay=0.0001,
        n_epochs=20,
        test_size=0.2,
        num_paths_per_item=3,
        path_length=3
    )
    
    if dr.load_data():
        results = dr.train()
        print(f"Train Loss = {results['train_loss']:.4f}, Test Loss = {results['test_loss']:.4f}")
        
        user_id = 1  # 示例用户ID
        recommendations = dr.recommend_for_user(user_id, n_rec=10, beam_size=5)
        
        print(f"\n为用户 {user_id} 的推荐物品:")
        for item_id, score in recommendations:
            print(f"物品ID: {item_id}, 得分: {score:.4f}")
    else:
        print("数据加载失败")
