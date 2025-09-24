import pandas as pd
import numpy as np
from collections import defaultdict
import math

# swing，itemcf的变体
class Swing:
    def __init__(self, 
                 dataset_path="ml-100k/ratings.csv", 
                 top_k=10,
                 lastn=30,
                 alpha=10):
        """
        取最相似的10个物品，保留用户最近交互的30个物品
        """
        self.dataset_path = dataset_path
        self.top_k = top_k
        self.lastn = lastn
        self.item_sim = defaultdict(dict)
        self.alpha = alpha
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
    

    def get_user_items(self):
        # 返回用户-物品的交互字典 {user: {item: rating}}
        self.user_items = defaultdict(dict)
        for _, row in self.ratings.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            rating = row.get("rating", 1.0)  # 如果没有rating就当1
            self.user_items[user_id][item_id] = rating
        return self.user_items


    # 获取共同用户，返回一个字典，key是物品，value是同时喜欢该物品的用户集合
    def get_co_users(self):
        self.item_users = defaultdict(list)       

        for _, row in self.ratings.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            self.item_users[item_id].append(user_id)
        # 返回的格式是 {item_id: [user_id1, user_id2, ...]}
        return self.item_users
    
    # 计算用户的overlap
    def get_overlap(self, users1, users2):
        # Overlap(u1,u2) = u1喜欢的物品和u2喜欢物品的交集
        return set(users1) & set(users2)
    
    def compute_item_similarity(self):
        # 两个物品的Swing相似度 = 1 / factor + overlap(u1,u2)
        # 不需要考虑rating
            
        self.get_user_items() # 返回如{user_id: {item_id: rating}}
        self.get_co_users()   # 返回如{item_id: [user_id1, user_id2, ...]}

        for u, items in self.user_items.items():
            print("处理用户:", u)
            items = list(items.keys())[-self.lastn:] # 最近lastn个物品
            # 两两物品配对
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    item_i, item_j = items[i], items[j]
                    # 获取共同用户集合
                    users_i = self.item_users[item_i]
                    users_j = self.item_users[item_j]
                    overlap_users = self.get_overlap(users_i, users_j)
                    overlap_users = list(overlap_users)

                    for idx1 in range(len(overlap_users)):
                        for idx2 in range(idx1 + 1, len(overlap_users)):
                            u1, u2 = overlap_users[idx1], overlap_users[idx2]
                            contrib = 1.0 / (self.alpha + len(self.get_overlap(self.user_items[u1], self.user_items[u2])))
                            self.item_sim[item_i][item_j] = self.item_sim[item_i].get(item_j, 0.0) + contrib 
                            self.item_sim[item_j][item_i] = self.item_sim[item_j].get(item_i, 0.0) + contrib

        # 排序
        self.item_sim_sorted = {
            i: sorted(sim.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
            for i, sim in self.item_sim.items()
        }
        return self.item_sim_sorted

    def recommend_for_user(self, user_id, n_rec=10):
        interacted_items = self.user_items[user_id]  # {item: rating}
        item_scores = defaultdict(float)

        for item_i, rating_i in interacted_items.items():
            # 用户对历史交互物品的兴趣 * 待计算物品和该物品的相似度
            if item_i not in self.item_sim:
                continue
            for item_j, sim_ij in self.item_sim[item_i].items():
                if item_j in interacted_items:  # 跳过用户已交互的物品
                    continue
                item_scores[item_j] += rating_i * sim_ij
        return sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n_rec]
    
    def build_user_item_index(self):
        self.user_item_index = {}
        
        for user_id, items in self.user_items.items():
            # 按评分降序排列 保留lastn个
            sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)[:self.lastn]
            self.user_item_index[user_id] = [item for item, _ in sorted_items]
    
    def build_item_item_index(self):
        """
        构建物品到物品的索引，返回每个物品最相似的K个物品
        """
        self.item_item_index = {}
        
        for item_id, similar_items in self.item_sim_sorted.items():
            self.item_item_index[item_id] = [item for item, _ in similar_items]
    
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
        return 0
    

if __name__ == "__main__":
    model = Swing(dataset_path="ml-100k/ratings.csv", top_k=10, lastn=20)
    if model.load_data():
        print("111")
        model.train()
        user_id = 1

        recommendations = model.recommend_for_user(user_id, n_rec=10)
        print(f"为用户 {user_id} 推荐的物品: {recommendations}")