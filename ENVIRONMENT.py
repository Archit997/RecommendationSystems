import numpy as np
from utlis import generate_items

def get_best_reward(items, theta):
	return np.max(np.dot(items, theta))

class Environment:
	# p: frequency vector of users
	def __init__(self, L, d, m, num_users, p, theta, T, seed, seeds, items=None):
       

		'''
		L : total number of items
		d : dimension of item feature vectors
		p : frequency vector of all the users (1 x num_users)
		items : L x d matrix of item feature vectors
		theta : -
		'''
		def generate_users(self):
			I = []
			for t in range(self.T):
				X = np.random.multinomial(1, self.p)
				I.append(np.nonzero(X)[0])
			return I
			
		self.L = L
		self.d = d
		self.p = p # probability distribution over users

		if items is not None:
			self.items = items
		else:
			self.items = generate_items(num_items = L, d = d, seed=seed)
		self.theta = theta
		self.T = T
		self.users = generate_users(self)
		self.seed = seed
		self.seeds = seeds


	def get_items(self):
		self.items = generate_items(num_items = self.L, d = self.d)
		return self.items

	def feedback(self, i, items_list, t):
		reward = 0
		product: int = 0
		expected_rewards = []
		K_t: int = -1

		for ik, ind in enumerate(items_list):
			x = self.items[ind, :]
			# print("thetas-")
			# print(self.theta[i])
			# print("x-")
			# print(x)
			r = np.dot(self.theta[i], x)
			# print("r-")
			# print(r)
			expected_rewards.append(round(float(r), 5))
			# print(r)
			np.random.seed(self.seeds[t])
			# print(r)

			y = np.random.binomial(1, r)
            
			# print(y)

			if y == 1 and K_t == -1:
				reward = 1
				product = r
				K_t = ik


		br = get_best_reward(self.items, self.theta[i])
		
		return reward, product, br, K_t, self.theta[i], expected_rewards