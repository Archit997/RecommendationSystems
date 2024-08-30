import numpy as np
import time
import random
import pickle
import matplotlib.pyplot as plt
from CLUB import CLUB
from gamma_club import GCLUB
from ENVIRONMENT import Environment

from utlis import generate_items, edge_probability

def main(num_stages, num_users, d, m, L, pj, k, gamma, num_rounds, items=None, filename=''):
    

    # set up theta vector
    def _get_theta(thetam, num_users, m):
        k = int(num_users / m)
        theta = {i:thetam[0] for i in range(k)}
        for j in range(1, m):
            theta.update({i:thetam[j] for i in range(k * j, k * (j + 1))})
        return theta

    if filename == '':
        thetam = generate_items(num_items=m, d=d)
        # thetam = np.concatenate((np.dot(np.concatenate((np.eye(m), np.zeros((m,d-m-1))), axis=1), ortho_group.rvs(d-1))/np.sqrt(2), np.ones((m,1))/np.sqrt(2)), axis=1)
        print(thetam, [[np.linalg.norm(thetam[i,:]-thetam[j,:]) for j in range(i)] for i in range(1,m)])
        theta = _get_theta(thetam, num_users, m)
        # print([np.linalg.norm(theta[0]-theta[i]) for i in range(num_users)])
    else:
        theta = np.load(filename)
        
    # set up frequency vector
    uniforms = ['uniform', 'half', 'arbitrary'] 
    def _get_half_frequency_vector(num_users, m):
        p0 = list(np.random.dirichlet(np.ones(m)))
        p = np.ones(num_users)
        k = int(num_users / m)
        for j in range(m):
            for i in range(k*j, k*(j+1)):
                p[i] = p0[j] / k
        p = list(p)
        return p
    prod_reward = []
    bin_reward = []
    gprod_reward = []
    gbin_reward = []
    time_states = []
    gtime_states = []
    for i in range(num_rounds):
        seed = int(time.time() * 100) % 399
        seeds = [int(time.time() * 100 + i + 1) % 399 for i in range(2 ** num_stages - 1)]
        
        print("Seed = %d" % seed)
        np.random.seed(seed)
        random.seed(seed)
        ps = [list(np.ones(num_users) / num_users), _get_half_frequency_vector(num_users=num_users,m=m), list(np.random.dirichlet(np.ones(num_users)))]
        
        for j in [pj]:
            p = ps[j]
            envir = Environment(L = L, d = d, m = m, num_users = num_users, p = p, theta = theta, T=2 ** num_stages - 1, items=items, seed=seed, seeds=seeds)
            # print(envir.items)


            club = CLUB(nu=num_users, d = d, gamma=gamma, T=2 ** num_stages - 1, k = k, edge_probability = edge_probability(num_users), seed=seed, seeds=seeds)
            start_time = time.time()
            club.run(envir)
            run_time = time.time() - start_time
            rounds = 2 ** num_stages - 1
            print(f"CLUB : {rounds} rounds {club.dis_cumul_reward[rounds-1]} product rewards {club.dis_cumul_bin_reward[rounds-1]} binomial rewards")
            print()
            print()
            prod_reward.append(club.cumulative_rewards)
            bin_reward.append(club.cumulative_bin_rewards)
            time_states.append(club.time_states)
            # print(club.time_states)
            # np.savez('club_' + uniforms[j] + '_nu' + str(num_users) + 'd' + str(d) + 'm' + str(m) + 'L' + str(L) + '_'+str(seed), seed, club.rewards, club.best_rewards, run_time, club.num_clusters)
            
            # path = f'results/{num_users}numusers_{d}d_{num_stages}stages_{L}items_CLUB.pkl'

            # with open(path, 'wb') as f:
            #     pickle.dump(club.time_states, f)
            
            club = GCLUB(nu=num_users, d = d, T=2 ** num_stages - 1, k=k, gamma=gamma, edge_probability = edge_probability(num_users), seed=seed, seeds=seeds)
            start_time = time.time()
            club.run(envir)
            run_time = time.time() - start_time
            print(f"Gamma - CLUB : {rounds} rounds {club.dis_cumul_reward[rounds-1]} product rewards {club.dis_cumul_bin_reward[rounds-1]} binomial rewards")
            print()
            print()
            gprod_reward.append(club.cumulative_rewards)
            gbin_reward.append(club.cumulative_bin_rewards)
            gtime_states.append(club.time_states)
            # print(club.time_states)
            # np.savez('gclub_' + uniforms[j] + '_nu' + str(num_users) + 'd' + str(d) + 'm' + str(m) + 'L' + str(L) + '_'+str(seed), seed, club.rewards, club.best_rewards, run_time, club.num_clusters)

            # path = f'results/{num_users}numusers_{d}d_{num_stages}stages_{L}items_GCLUB.pkl'

            # with open(path, 'wb') as f:
            #     pickle.dump(club.time_states, f)
    path = f'results/{num_users}numusers_{d}d_{num_stages}stages_{L}items_{num_rounds}_rounds_GCLUB.pkl'

    with open(path, 'wb') as f:
        pickle.dump(gtime_states, f)
    
    path = f'results/{num_users}numusers_{d}d_{num_stages}stages_{L}items_{num_rounds}_rounds_CLUB.pkl'

    with open(path, 'wb') as f:
        pickle.dump(time_states, f)

    return prod_reward, bin_reward, gprod_reward, gbin_reward

if __name__== "__main__":
    #main(num_stages = 20, num_users = 10000, d = 20, m = 10, L = 20, pj = 0)
    main(num_stages = 15, num_rounds=10, num_users = 1000, d = 20, m = 10, L = 1000, pj = 0, k=4, gamma = 0.7, filename='ml_1000user_d20_m10.npy')
    # total_time = list(range(a1.size))
    # plt.plot(total_time, a1, label='CLUB gamma = 0.7')
    
   
    # plt.plot(total_time, b1, label='GCLUB gamma = 0.7')
    
    
   

    # # Adding labels and title
    # plt.xlabel('Time')
    # plt.ylabel('Reward')
    # plt.title('Rewards Over Time varying gamma')

    # # Adding legend
    # plt.legend()

    # # Save the plot to a file
    # plt.savefig('rewards_plot1(varying gamma).png')  # You can specify any filename and format (e.g., .jpg, .pdf)

    # # Display the plot
    # plt.show()
    # plt.plot(total_time, a2, label='CLUB gamma = 0.7')
    
    
    # plt.plot(total_time, b2, label='GCLUB gamma = 0.7')
    
    
   


    # # Adding labels and title
    # plt.xlabel('Time')
    # plt.ylabel('Reward')
    # plt.title('Bin Rewards Over Time varying gamma')

    # # Adding legend
    # plt.legend()

    # # Save the plot to a file
    # plt.savefig('binrewards_plot1(varying gamma).png')  # You can specify any filename and format (e.g., .jpg, .pdf)

    # # Display the plot
    # plt.show()
    # main(num_stages = 20, num_users = 1000, d = 20, m = 10, L = 20, pj = 0, filename='yelp_1000user_d20.npy')
