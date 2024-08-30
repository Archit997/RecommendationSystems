import numpy as np 
from typing import List
from sklearn.cluster import KMeans
from typing import Any


class DataGen:
    def __init__(self, num_users: int, d: int, m: int, thetas: Any = None, user_dist: List[int] = None, epsilon: int = 0.01, normalize=True):
        self.num_users = num_users
        self.d = d
        self.m = m
        self.epsilon = epsilon
        self.thetas = thetas

        assert len(self.thetas) == self.num_users
        assert len(self.thetas[0]) == self.d

        if normalize:

            normalized_thetas = [np.array(theta) / np.linalg.norm(theta) for theta in thetas]

            normalized_thetas_matrix = np.array(normalized_thetas)
            
            self.thetas = np.array(normalized_thetas_matrix)

        if user_dist:
            self.dist = user_dist
        
        else:
            assert num_users % m == 0
            self.dist = [num_users // m for _ in range(m)]

        assert sum(self.dist) == self.num_users

    def gen_random_vec(self, d: int):

        random_vector = np.random.rand(d)  
        normalized_vector = random_vector / np.linalg.norm(random_vector)  

        return normalized_vector
    

    def generate_dist(self):
        print("thetas are")
        print(self.thetas)
        if self.thetas is not None:
            clusters = {i: self.thetas[i] for i in  range(self.m)}
        else:
            clusters = {i : self.gen_random_vec(self.d) for i in range(self.m)}

        users = np.zeros((self.num_users, self.d))
        thetas = np.zeros((self.num_users, self.d))

        user_ind = 0

        for i in range(self.m):
            median = clusters[i]

            for j in range(self.dist[i]):
                delta = np.random.rand(self.d)
                delta /= np.linalg.norm(delta)
                delta *= self.epsilon

                user_feat = delta + median
                user_feat /= np.linalg.norm(user_feat)

                users[user_ind] = user_feat
                thetas[user_ind] = median

                user_ind += 1

        save_path_users = f'data/synthetic/{self.num_users}users_{self.m}clusters.npy'
        save_path_thetas = f'data/synthetic/{self.num_users}users_{self.m}clusters_thetas.npy'
        print("saved thetas")
        print(self.thetas)
        np.save(save_path_users, users)
        np.save(save_path_thetas, self.thetas)

        return users, thetas
    
    def generate_random(self):
        random_matrix = np.random.rand(self.num_users, self.d)
        norms = np.linalg.norm(random_matrix, axis=1, keepdims=True)
        users = random_matrix / norms

        save_path_users = f'data/synthetic/{self.num_users}users_{self.m}clusters.npy'
        np.save(save_path_users, users)

        thetas = self.kmeans_thetas(users)

        save_path_thetas = f'data/synthetic/{self.num_users}users_{self.m}clusters_thetas.npy'
        np.save(save_path_thetas, thetas)

        return users
    
    def kmeans_thetas(self, user_features):
        U = user_features

        kmeans = KMeans(n_clusters=self.m).fit(U)

        thetas = np.zeros((self.num_users, self.d))

        for i in range(self.num_users):
            thetas[i] = kmeans.cluster_centers_[kmeans.labels_[i]]
        
        return thetas