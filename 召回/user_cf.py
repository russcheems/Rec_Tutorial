import pandas as pd
import numpy as np
from collections import defaultdict
import math


class UserCF:
    def __init__(self, 
                 dataset_path="ml-100k/ratings.csv", 
                 top_k=10,
                 lastn=15):
    
        # 同样维护两个索引
        # 用户 - 物品索引 保留用户最近交互的15个物品
        self.user_item_index = defaultdict(list)
        # 用户 - 用户索引 保留每个用户的top_k个相似用户
        self.user_user_index = defaultdict(list)

        self.dataset_path = dataset_path
        self.top_k = top_k
        self.lastn = lastn
        self.ratings = None
        # 交集索引 如果直接存成嵌套字典会很大


    def load_data(self):
        try:
            self.ratings = pd.read_csv(self.dataset_path)
            print(f"共{len(self.ratings)}条数据")
            return True
        except FileNotFoundError:
            # print(111)
            return False

    def get_user_items(self):
        # 构建用户 - 物品索引
        if self.ratings is not None:
            for user, group in self.ratings.groupby("userId"):
                self.user_item_index[user] = group["movieId"].tail(self.lastn).tolist()
        return self.user_item_index
    
    def get_user_users(self):
        # 构建用户 - 用户索引 即和用户最相似的top_k个用户 调用calculate_user_similarity()计算相似度

        for user1 in self.user_item_index.keys():
            similarities = []
            for user2 in self.user_item_index.keys():
                if user1 != user2:
                    sim = self.calculate_user_similarity(user1, user2)
                    similarities.append((user2, sim))
            # 取前k个相似用户
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:self.top_k]
            self.user_user_index[user1] = similarities

        return self.user_user_index

    def calculate_user_similarity(self, user1, user2):
        # sim = |N(u1) ∩ N(u2)| / sqrt(|N(u1)| * |N(u2)|)

        user1_items = set(self.user_item_index.get(user1, []))
        user2_items = set(self.user_item_index.get(user2, []))

        intersection = user1_items.intersection(user2_items)
        if not intersection:
            return 0

        return len(intersection) / math.sqrt(len(user1_items) * len(user2_items))
    


    def get_item_users(self):
        # 返回某个物品喜欢的用户数量
        item_users = defaultdict(set) #这里用set是为了去重
        for user, items in self.user_item_index.items():
            for item in items:
                item_users[item].add(user)
        # 返回格式是 {item_id: user_count}
        return {item: len(users) for item, users in item_users.items()}
    
    def recommend_for_user(self, user_id, n_rec=10):

        user_interacted_items = set(self.user_item_index[user_id])
        item_scores = defaultdict(float)
        similar_users = self.user_user_index[user_id]
        for sim_user_id, similarity in similar_users:
            # 相似用户交互的物品
            sim_user_items = self.user_item_index[sim_user_id]
            
            for item_id in sim_user_items:
                if item_id in user_interacted_items:
                    continue

                item_scores[item_id] += similarity

        return sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n_rec]
    
    def get_similar_users(self, user_id, n_users=10):

        if user_id not in self.user_user_index:
            print(f"用户ID {user_id} 不在数据集中")
            return []
        
        # 返回前n_users个相似用户
        similar_users = self.user_user_index[user_id]
        return similar_users[:min(n_users, len(similar_users))]
    
    def train(self):

        self.get_user_items()
        self.get_user_users()
        return 0


if __name__ == "__main__":
    model = UserCF(dataset_path="ml-100k/ratings.csv", top_k=10, lastn=15)
    if model.load_data():
        model.train()
        user_id = 1  
        recommendations = model.recommend_for_user(user_id, n_rec=10)
        
        print(f"\n为用户 {user_id} 的推荐物品:")
        for item_id, score in recommendations:
            print(f"物品ID: {item_id}, 分数: {score:.4f}")
    else:
        print("数据文件未找到，请检查路径")