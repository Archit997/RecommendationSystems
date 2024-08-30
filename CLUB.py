import networkx as nx
import numpy as np
from BASE import LinUCB_IND

class Cluster:
    def __init__(self, users, S, b, N):
        self.users = users
        self.S = S
        self.b = b
        self.N = N
        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)

class CLUB(LinUCB_IND):
    # random_init: use random initialization or not
    def __init__(self, nu, d, T, k, seed, seeds, gamma=1, edge_probability = 1):

        '''
        nu : number of users
        d : 
        T : number of rounds
        k : number of items recommended per turn
        G : graph structure created using networkx package
        clusters : dict of cluster index to cluster object
        cluster_inds : map of user to cluster index
        num_clusters : number of clusters
        '''

        super(CLUB, self).__init__(nu, d, T, gamma)
        self.nu = nu
        # self.alpha = 4 * np.sqrt(d) # parameter for cut edge
        self.k = k
        self.gamma = gamma
        self.seed = seed
        self.seeds = seeds
        self.G = nx.gnp_random_graph(nu, edge_probability, seed=self.seed)
        self.clusters = {0: Cluster(users=range(nu), S=np.eye(d), b=np.zeros(d), N=0)}
        self.cluster_inds = np.zeros(nu)
        self.num_clusters = np.zeros(T)

    def recommend(self, i, items, t):

        cluster = self.clusters[self.cluster_inds[i]]
        self.time_states[t].recommended_list = self._select_multi_item_ucb(cluster.S, cluster.Sinv, cluster.theta, items, cluster.N, t, self.k)
        return self.time_states[t].recommended_list

    def store_info(self, i, x, y, t, r, br):
        super(CLUB, self).store_info(i, x, y, t, r, br)

        c = self.cluster_inds[i]

        for item_vec in x:
            self.clusters[c].S += np.outer(item_vec, item_vec)
        self.clusters[c].b += y * x[-1] if len(x) else 0
        self.clusters[c].N += 1

        self.time_states[t].user_cluster_theta_estimate = self.clusters[c].theta
        self.clusters[c].Sinv, self.clusters[c].theta = self._update_inverse(self.clusters[c].S, self.clusters[c].b, self.clusters[c].Sinv, x, self.clusters[c].N)
        self.time_states[t].user_cluster_theta_update = self.clusters[c].theta

    def _if_split(self, theta, N1, N2):
        # alpha = 2 * np.sqrt(2 * self.d)
        alpha = 1
        def _factT(T):
            return np.sqrt((1 + np.log(1 + T)) / (1 + T))
        return np.linalg.norm(theta) >  alpha * (_factT(N1) + _factT(N2))
 
    def update(self, t):
        update_clusters = False
        for i in self.I:
            c = self.cluster_inds[i]

            A = [a for a in self.G.neighbors(i)]
            for j in A:
                if self.N[i] and self.N[j] and self._if_split(self.theta[i] - self.theta[j], self.N[i], self.N[j]):
                    self.G.remove_edge(i, j)

                    update_clusters = True

        if update_clusters:
            C = set()
            for i in self.I: # suppose there is only one user per round
                C = nx.node_connected_component(self.G, i)
                if len(C) < len(self.clusters[c].users):
                    remain_users = set(self.clusters[c].users)
                    self.clusters[c] = Cluster(list(C), S=sum([self.S[k]-np.eye(self.d) for k in C])+np.eye(self.d), b=sum([self.b[k] for k in C]), N=sum([self.N[k] for k in C]))

                    remain_users = remain_users - set(C)
                    c = max(self.clusters) + 1
                    while len(remain_users) > 0:

                        np.random.seed(self.seeds[t])
                        j = np.random.choice(list(remain_users))
                        C = nx.node_connected_component(self.G, j)

                        self.clusters[c] = Cluster(list(C), S=sum([self.S[k]-np.eye(self.d) for k in C])+np.eye(self.d), b=sum([self.b[k] for k in C]), N=sum([self.N[k] for k in C]))
                        for j in C:
                            self.cluster_inds[j] = c

                        c += 1
                        remain_users = remain_users - set(C)

            
        self.num_clusters[t] = len(self.clusters)