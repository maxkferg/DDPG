import filter_env
from ddpg import *
import gc
gc.enable()
import matplotlib.pyplot as plt
from os.path import isdir
from os import mkdir
from sys import argv
from time import sleep

import inspect

TEST = False
ENV_NAME = 'MountainCarContinuous-v0'
#ENV_NAME = 'ObjectTransition-v0'
PATH = 'experiments/' + ENV_NAME + '-E5/'
EPISODES = 100000
TEST_NUM = 3

def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    #env = gym.make(ENV_NAME)
    
    agent = DDPG(env)
    #env.monitor.start('experiments/' + ENV_NAME,force=True)
    #env = gym.wrappers.Monitor(env, PATH, force=True)


    returns = []
    rewards = []

    fig_reward = plt.figure(1)
    fig_reward_tra = fig_reward.add_subplot(1, 1, 1)
    fig_avg_res, ax_avg_res = plt.subplots(2)

    init_state = []

    if len(argv) > 1:
        TEST = argv[1]
        init_state = [int(argv[2]), int(argv[3])]


    # if TEST: #for object transition
    #     agent.load_model(PATH)
    #     total_return = 0
    #     total_reward = 0
    #     for i in xrange(TEST_NUM):
    #         env.render()
    #         reward_tra = []                
    #         state = env.reset()
    #         #env.set_state(init_state)
    #         #state = env.get_state()
    #         info = {}
    #         info['state'] = init_state
    #         env.configure(info)

    #         reward_per_step = 0
    #         for j in xrange(env.spec.timestep_limit):
    #             env.render()
    #             action = agent.action(state) # direct action for test
    #             state, reward, done, _ = env.step(action)
    #             total_return += reward
    #             if done:
    #                 break
    #             reward_per_step += (reward - reward_per_step)/(j+1)
    #             sleep(0.2)
    #         total_reward += reward_per_step
    #     return

    for episode in xrange(EPISODES):
        state = env.reset()
        reward_tra = []
        print "episode:", episode
        # Train
        for step in xrange(env.spec.timestep_limit):
            #env.render()
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            #print('state={}, action={}, reward={}, next_state={}, done={}'.format(state, action, reward, next_state, done))
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            reward_tra.append(reward)
            if done:
                    break
    
        fig_reward_tra.clear()
        fig_reward_tra.plot(reward_tra)
        fig_reward.show()
        plt.pause(0.005)
    
        # Testing:
        #if episode % 1 == 0:
        if episode % 50 == 0 and episode >= 100:
            #env.render()            
            if not isdir(PATH):            
                mkdir(PATH)
            agent.save_model(PATH, episode)    
            total_return = 0
            total_reward = 0
            for i in xrange(TEST_NUM):
                #env.render()                
                state = env.reset()
                reward_per_step = 0
                for j in xrange(env.spec.timestep_limit):
                    env.render()
                    action = agent.action(state) # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_return += reward
                    if done:
                        break
                    reward_per_step += (reward - reward_per_step)/(j+1)
                total_reward += reward_per_step
        
            ave_return = total_return/TEST_NUM
            ave_reward = total_reward/TEST_NUM
            returns.append(ave_return)
            rewards.append(ave_reward)
        
            ax_avg_res[0].clear()
            ax_avg_res[1].clear()
            ax_avg_res[0].plot(returns)
            ax_avg_res[1].plot(rewards)
            fig_avg_res.show()
            plt.pause(0.005)

            print 'episode: ',episode,'Evaluation Average Return:',ave_return, '  Evaluation Average Reward: ', ave_reward

    fig_avg_res.savefig(PATH + "train_reward.png")
    #env.monitor.close()

if __name__ == '__main__':
    main()
