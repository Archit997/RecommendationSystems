import numpy as np
from typing import List
from tqdm import tqdm
from pydantic import BaseModel
from typing import Any

from ENVIRONMENT import Environment

class TimeState(BaseModel):
    user: int  = -1 # user at step t
    user_cluster_theta_estimate: Any = None # user theta feature estimate by agent
    user_cluster_theta_actual: Any = None # user theta feature actual stored in environ
    user_cluster_theta_update: Any  = None # updated user theta estimate by agent
    recommended_list: Any = None # recommended list DONE
    index: int = -1 # index of item picked by user
    prod_reward: int = 0 # expected reward of user with that list
    bin_reward: int = 0 # binomial reward of user with that list
    dis_prod_reward: int = 0 # discounted
    dis_bin_reward: int = 0 # discounted
    expected_rewards_actual: Any = None 
    expected_rewards_agent: Any = None
    rewards_radius_agent: Any = None


#TODO : instead of all the rewards written in init of Base just have an array of size T with each entry being TimeState object

class Base:
    # Base agent for online clustering of bandits
    def __init__(self, d, T, gamma=1):
        self.d = d
        self.T = T
        self.gamma = gamma

        #TODO : Fill this
        
        self.time_states: List[TimeState] = [TimeState() for _ in range(self.T)]

        self.best_rewards = np.zeros(self.T)

        self.cumulative_rewards = np.zeros(self.T)
        self.cumulative_bin_rewards = np.zeros(self.T)
        self.dis_cumul_reward = np.zeros(self.T)
        self.dis_cumul_bin_reward = np.zeros(self.T)

    def _beta(self, N, t):
        return np.sqrt(self.d * np.log(1 + N / self.d) + 4 * np.log(t) + np.log(2)) + 1

    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        return np.argmax(np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1))
    
    def _select_multi_item_ucb(self, S, Sinv, theta, items, N, t, k) -> List[int]:
        '''Return indices of top k items based on UCB'''

        expected_rewards = np.dot(items, theta)

        # assert np.all((expected_rewards >= 0) & (expected_rewards <= 1)) #TODO what to do with this

        radius_values = self._beta(N, t) * np.sqrt((np.matmul(items, Sinv) * items).sum(axis=1))
        ucb_values = expected_rewards + radius_values
        # ucb_values = np.clip(expected_rewards + radius_values, 0, 1)
        # if t<=5:
        #     print(ucb_values)
        sorted_indices = np.argsort(ucb_values)[::-1][:k]

        self.time_states[t].expected_rewards_agent = expected_rewards[sorted_indices]
        self.time_states[t].rewards_radius_agent = radius_values[sorted_indices]

        return sorted_indices


    def recommend(self, i, items, t):

        return

    def store_info(self, i, x, y, t, r, br):
        return

    def _update_inverse(self, S, b, Sinv, x, t):
        Sinv = np.linalg.inv(S)
        theta = np.matmul(Sinv, b)

        # assert np.linalg.norm(theta) <= 1 #TODO what to do with this math ??

        return Sinv, theta

    def update(self, t):
        return

    def run(self, envir: Environment):
        for t in tqdm(range(self.T)):

            self.I = envir.users[t]

            if len(self.I) > 1:
                raise ValueError("user??")
            
            for i in self.I:

                self.time_states[t].user = i
                
                items = envir.items

                items_list = self.recommend(i=i, items=items, t=t)

                x = [items[ind] for ind in items_list]
                y, r, br, K_t, self.time_states[t].user_cluster_theta_actual, self.time_states[t].expected_rewards_actual = envir.feedback(i=i, items_list=items_list, t=t)

                if K_t == -1:
                    assert y == 0 and r == 0


                self.time_states[t].index = len(x) if K_t == -1 else K_t

                x = x[ : K_t+1]
                self.store_info(i=i, x=x, y=y, t=t, r=r, br=br)

            self.update(t)

            if t==0:
                self.cumulative_bin_rewards[t] = self.time_states[t].bin_reward
                self.cumulative_rewards[t] = self.time_states[t].prod_reward
                self.dis_cumul_reward[t] = self.time_states[t].dis_prod_reward
                self.dis_cumul_bin_reward[t] = self.time_states[t].dis_bin_reward
                
            else:
                self.cumulative_bin_rewards[t] = self.time_states[t].bin_reward + self.cumulative_bin_rewards[t-1]
                self.cumulative_rewards[t] = self.time_states[t].prod_reward + self.cumulative_rewards[t-1]
                self.dis_cumul_reward[t] = self.time_states[t].dis_prod_reward + self.dis_cumul_reward[t-1]
                self.dis_cumul_bin_reward[t] = self.time_states[t].dis_bin_reward + self.dis_cumul_bin_reward[t-1]


class LinUCB(Base):
    def __init__(self, d, T):
        super(LinUCB, self).__init__(d, T)
        self.S = np.eye(d)
        self.b = np.zeros(d)
        self.Sinv = np.eye(d)
        self.theta = np.zeros(d)

    def recommend(self, i, items, t):
        return self._select_item_ucb(self.S, self.Sinv, self.theta, items, t, t)

    def store_info(self, i, x, y, t, r, br):
        self.rewards[t] += r
        self.best_rewards[t] += br

        for item_vec in x: 
            self.S += np.outer(item_vec, item_vec)
        self.b += y * x[-1]

        self.Sinv, self.theta = self._update_inverse(self.S, self.b, self.Sinv, x, t)

class LinUCB_Cluster(Base):
    def __init__(self, indexes, m, d, T):
        super(LinUCB_Cluster, self).__init__(d, T)
        self.indexes = indexes

        self.S = {i:np.eye(d) for i in range(m)}
        self.b = {i:np.zeros(d) for i in range(m)}
        self.Sinv = {i:np.eye(d) for i in range(m)}
        self.theta = {i:np.zeros(d) for i in range(m)}

        self.N = np.zeros(m)

    def recommend(self, i, items, t):
        j = self.indexes[i]
        return self._select_item_ucb(self.S[j], self.Sinv[j], self.theta[j], items, self.N[j], t)

    def store_info(self, i, x, y, t, r, br):
        self.rewards[t] += r
        self.best_rewards[t] += br

        j = self.indexes[i]
        self.S[j] += np.outer(x, x)
        self.b[j] += y * x
        self.N[j] += 1

        self.Sinv[j], self.theta[j] = self._update_inverse(self.S[j], self.b[j], self.Sinv[j], x, self.N[j])
        

class LinUCB_IND(Base):
    # each user is an independent LinUCB
    def __init__(self, nu, d, T, gamma):
        super(LinUCB_IND, self).__init__(d, T, gamma)
        self.S = {i:np.eye(d) for i in range(nu)}
        self.b = {i:np.zeros(d) for i in range(nu)}
        self.Sinv = {i:np.eye(d) for i in range(nu)}
        self.theta = {i:np.zeros(d) for i in range(nu)}

        self.N = np.zeros(nu)


    def store_info(self, i, x, y, t, r, br):
        self.time_states[t].prod_reward = r
        self.time_states[t].bin_reward = y
        self.time_states[t].dis_prod_reward = (self.gamma ** self.time_states[t].index) * r 
        self.time_states[t].dis_bin_reward = (self.gamma ** self.time_states[t].index) * y
        self.best_rewards[t] = br


        for item_vec in x: 
            self.S[i] += np.outer(item_vec, item_vec)

        self.b[i] += y * x[-1] if len(x) else 0
        self.N[i] += 1

        self.Sinv[i], self.theta[i] = self._update_inverse(self.S[i], self.b[i], self.Sinv[i], x, self.N[i])