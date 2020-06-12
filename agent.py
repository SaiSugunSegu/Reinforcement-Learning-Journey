import gym
import numpy as np

gamma = 0.9
theta = 0.0005

def q_greedify_policy(env, V, pi, action, s, gamma):
    e = np.zeros(env.env.nA)
    for a in range(env.env.nA):
        q=0
        P = np.array(env.env.P[s][a])
        (x,y) = np.shape(P)
        for i in range(x):
            s_= int(P[i][1])
            p = P[i][0]
            r = P[i][2]
            q += p*(r+gamma*V[s_])
            e[a] = q
    m = np.argmax(e)
    action[s]=m
    pi[s][m] = 1

    return pi, action


def bellman_optimality_update(env, V, s, gamma):
    
    pi = np.zeros((env.env.nS, env.env.nA)) 
    
    e = np.zeros(env.env.nA)

    for a in range(env.env.nA):
        q=0
        P = np.array(env.env.P[s][a])
        (x,y) = np.shape(P)
        for i in range(x):
            s_= int(P[i][1])
            p = P[i][0]
            r = P[i][2]
            q += p*(r+gamma*V[s_])
            e[a] = q
    m = np.argmax(e)
    pi[s][m] = 1
    

    value = 0
    for a in range(env.env.nA):
        u = 0
        P = np.array(env.env.P[s][a])
        (x,y) = np.shape(P)
        for i in range(x):
            s_= int(P[i][1])
            p = P[i][0]
            r = P[i][2]
            
            u += p*(r+gamma*V[s_])
            
        value += pi[s,a] * u
  
    V[s]=value
    return V[s]


def value_iteration(env, gamma, theta):
    a = env.env.nA
    s = env.env.nS
    V = np.zeros(s)
    while True:
        delta = 0
        for s in range(env.env.nS):
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
            
    pi = np.zeros((env.env.nS, env.env.nA))   
    action = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        pi, action = q_greedify_policy(env, V, pi, action, s, gamma)
        
    return V, pi, action


V, pi, action = value_iteration(env, gamma, theta)
print(action)
print(np.reshape(action,(4,4)))


r=0
for i_episode in range(10):
    c = env.reset()
    for t in range(10000000):
        c, reward, done, info = env.step(action[c])
        if done:
            #print("Episode finished after {} timesteps".format(t+1))
            if reward == 0:
                print(r)
                r = 0
            else:
                r += reward
            break
print(r)
env.close()



