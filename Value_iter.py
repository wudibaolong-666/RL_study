import numpy as np
import time
from GridWorld import GridWorld


if __name__ == "__main__":
    world_length = 30
    gamma = 0.99
    env = GridWorld(world_length, 10)
    env.reset()

    #  init v
    v = np.zeros((world_length, world_length))
    v_past = np.zeros((world_length, world_length))
    #  init q
    q = np.zeros((world_length,world_length,5))
    #  init pi
    pi = np.zeros((world_length,world_length,5))

    for iter_num in range(100):
        v_past = v.copy()
        for i in range(world_length):
            for j in range(world_length):
                if i != world_length - 1 or j != world_length - 1:
                    for k in range(env.move.shape[0]):
                        env.set_state(np.array([i,j]))
                        next_state, reward, done= env.step(k)
                        q[i,j,k] = reward + gamma * v[next_state[0], next_state[1]]
                # policy update
                max_action = np.argmax(q[i,j,:])
                pi[i,j,:] = 0
                pi[i,j,max_action] = 1
                # value update
                v[i,j] = q[i,j,max_action]

        # print(v)
        # print(v_past)
        if np.sum((v-v_past)**2)<0.000001:
            print(iter_num)
            break

    env.reset()

    state = env.state
    while True:
        env.render()
        time.sleep(0.1)
        action = np.argmax(pi[state[0],state[1],:])
        state,reward,done = env.step(action)
        if done:
            env.render()
            env.set_state(state)
            time.sleep(0.1)
            env.close()
            break




