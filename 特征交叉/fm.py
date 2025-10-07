import numpy as np

class FactorizationMachine:

    def __init__(self, n_features, n_factors=10, learning_rate=0.01, epochs=10):
        # 初始化参数
        self.n_features = n_features
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.w0 = 0  # 全局偏置
        self.w = np.zeros(n_features)  # 一阶特征权重

        self.v = np.random.normal(scale=0.1, size=(n_features, n_factors))
        
    def fit(self, X, y):
 
        # 存储每个epoch的损失
        losses = []
        
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                
                # 计算预测值
                y_pred = self.predict_one(x)
                error = y_pred - y_true
                
                # 更新w0
                gradient_w0 = error
                self.w0 -= self.learning_rate * gradient_w0
                for j in range(self.n_features):
                    if x[j] != 0:  # 跳过零值特征，优化稀疏数据计算
                        gradient_wj = error * x[j]
                        self.w[j] -= self.learning_rate * gradient_wj
                
                for j in range(self.n_features):
                    if x[j] != 0:

                        for f in range(self.n_factors):
                            sum_vx = np.sum(self.v[:, f] * x)
                            gradient_vjf = error * (x[j] * sum_vx - self.v[j, f] * x[j]**2)
                            self.v[j, f] -= self.learning_rate * gradient_vjf

            y_pred = self.predict(X)
            loss = np.mean((y - y_pred) ** 2)
            losses.append(loss)
            
        return losses

    def predict_one(self, x):

        # 线性 w0 + Σ(wi*xi)
        linear_term = self.w0 + np.dot(self.w, x)
        
        interaction_term = 0
        sum_vx_square = np.zeros(self.n_factors)
        sum_vx2 = np.zeros(self.n_factors)
        
        for j in range(self.n_features):
            if x[j] != 0:  #
                vx = self.v[j] * x[j]
                sum_vx_square += vx
                sum_vx2 += vx**2

        interaction_term = 0.5 * np.sum(sum_vx_square**2 - sum_vx2)
        # 最终预测值 = 线性项 + 二阶交互项
        return linear_term + interaction_term
    
    def get_feature_interactions(self, i, j):
        return np.dot(self.v[i], self.v[j])

