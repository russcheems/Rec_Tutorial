import math
from collections import defaultdict
from operator import itemgetter
import pandas as pd

class UserCF:
    def __init__(self, trainData, similarity="cosine"):
        self._trainData = trainData
        self._similarity = similarity
        self._userSimMatrix = dict()

    def similarity(self):
        item_user = dict()
        for user, items in self._trainData.items():
            for item in items:
                item_user.setdefault(item, set())
                item_user[item].add(user)

        for item, users in item_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self._userSimMatrix.setdefault(u, defaultdict(int))
                    if self._similarity == "cosine":
                        self._userSimMatrix[u][v] += 1
                    elif self._similarity == "iif":
                        self._userSimMatrix[u][v] += 1.0 / math.log(1 + len(users))

        for u, related_user in self._userSimMatrix.items():
            for v, cuv in related_user.items():
                nu = len(self._trainData[u])
                nv = len(self._trainData[v])
                self._userSimMatrix[u][v] = cuv / math.sqrt(nu * nv)

    def recommend(self, user, N=10, K=5):
        recommends = dict()
        related_items = self._trainData[user]
        for v, sim in sorted(self._userSimMatrix[user].items(), key=itemgetter(1), reverse=True)[:K]:
            for item in self._trainData[v]:
                if item in related_items:
                    continue
                recommends.setdefault(item, 0.)
                recommends[item] += sim
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])


def load_data(path="ml-100k/u.data"):
    df = pd.read_csv(path, sep="\t", names=['user_id', 'item_id', 'rating', 'timestamp'])
    trainData = dict()
    for row in df.itertuples():
        if row.rating >= 4:  
            trainData.setdefault(row.user_id, set())
            trainData[row.user_id].add(row.item_id)
    return trainData


if __name__ == "__main__":
    trainData = load_data("ml-100k/u.data")
    model = UserCF(trainData, similarity="cosine")
    model.similarity()
    user = 1
    print("User:", user)
    print("Recommend:", model.recommend(user, N=10, K=5))