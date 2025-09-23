import pandas as pd
import numpy as np
from collections import defaultdict
import math

class ItemCF:
    def __init__(self, 
                 dataset_path="../dataset/ml-latest-small/ratings.csv", 
                 top_k=10,
                 lastn=30):
        """
        取最相似的10个物品，保留用户最近交互的30个物品
        """
        self.dataset_path = dataset_path
        self.top_k = top_k
        self.lastn = lastn
        self.ratings = None
        self.item_sim = None
        self.item_sim_sorted = None
        self.user_items = None
        self.user_item_index = None
        self.item_item_index = None
    
    def load_data(self):
        try:
            self.ratings = pd.read_csv(self.dataset_path)
            print(f"共{len(self.ratings)}条数据")
            return True
        except FileNotFoundError:
            # print(111)
            return False
    
    def compute_item_similarity(self):
        # 维护用户-物品 和 物品-用户 的映射
        # 是否需要维护物品-用户映射？
        self.user_items = defaultdict(dict)  # 用户对物品的评分
        item_users = defaultdict(dict)       

        for _, row in self.ratings.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            rating = row['rating']
            self.user_items[user_id][item_id] = rating
            item_users[item_id][user_id] = rating

        self.item_sim = defaultdict(dict)
        self.item_sim_sorted = defaultdict(list)  # 存储每个物品的top_k个最相似物品

        total_items = len(item_users)
        
        for i, (item_i, users_i) in enumerate(item_users.items()):
            
            similarities = []
            
            for item_j, users_j in item_users.items():
                if item_i == item_j:
                    continue
                
                # 同时喜欢两个物品的用户集合
                common_users = set(users_i.keys()) & set(users_j.keys())
                
                if len(common_users) == 0:
                    continue
                    
                # 相似度计算：同时喜欢两个物品的人数 / sqrt(喜欢物品i的人数 * 喜欢物品j的人数)
                similarity = len(common_users) / math.sqrt(len(users_i) * len(users_j))
                self.item_sim[item_i][item_j] = similarity
                similarities.append((similarity, item_j))
            
            # 直接对所有相似度排序，选择top_k个
            similarities.sort(reverse=True)
            self.item_sim_sorted[item_i] = similarities[:self.top_k]
    
    def recommend_for_user(self, user_id, n_rec=10):

        interacted_items = self.user_items[user_id]
        item_scores = defaultdict(float)

        for item_i, rating_i in interacted_items.items():
            # 用户对历史交互物品的兴趣 * 待计算物品和该物品的相似度
            for item_j, sim_ij in self.item_sim[item_i].items():
                if item_j in interacted_items:  # 跳过用户已交互的物品
                    continue
                # Item CF 公式: 用户对历史交互物品的兴趣 * 待计算物品和该物品的相似度
                item_scores[item_j] += rating_i * sim_ij
        
        # 返回评分最高的n_rec个物品
        return sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n_rec]
    
    def build_user_item_index(self):
        self.user_item_index = {}
        
        for user_id, items in self.user_items.items():
            # 按评分降序排列物品，并最多保留lastn个
            sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)[:self.lastn]
            self.user_item_index[user_id] = [item for item, _ in sorted_items]
    
    def build_item_item_index(self):
        """
        构建物品到物品的索引，返回每个物品最相似的K个物品
        """
        self.item_item_index = {}
        
        for item_id, similar_items in self.item_sim_sorted.items():
            self.item_item_index[item_id] = [item for _, item in similar_items]
    
    def get_similar_items(self, item_id):
        return self.item_item_index[item_id]
    
    def get_user_interacted_items(self, user_id):
        return self.user_item_index[user_id]
    
    def train(self):
        self.compute_item_similarity()
        
        # 构建索引
        self.build_user_item_index()
        self.build_item_item_index()
        
        print("111")
        return self

if __name__ == "__main__":
    item_cf = ItemCF(dataset_path="ml-100k/ratings.csv", top_k=20, lastn=50)
    if not item_cf.load_data():
        exit()
    
    item_cf.train()
    
    user_id = 1  
    recommendations = item_cf.recommend_for_user(user_id, n_rec=10)
    
    print(f"\n为用户 {user_id} 的推荐物品:")
    for item_id, score in recommendations:
        print(f"物品ID: {item_id}, 分数: {score:.4f}")
    
