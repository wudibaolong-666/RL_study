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
    v_oldpi = np.zeros((world_length, world_length))
    #  init q
    q = np.zeros((world_length,world_length,5))
    #  init pi
    pi = np.zeros((world_length,world_length,5))
    pi[:, :, 0] = 1

    for iter_num in range(100):
        # PE
        v_oldpi = v.copy()
        while True:
            v_past = v.copy()
            for i in range(world_length):
                for j in range(world_length):
                    if i != world_length - 1 or j != world_length - 1:
                        new_v = 0
                        for k in range(env.move.shape[0]):
                            env.set_state(np.array([i, j]))
                            next_state, reward, done = env.step(k)
                            new_v += pi[i, j, k] * (reward + gamma * v[next_state[0], next_state[1]])
                        v[i, j] = new_v
            # if np.sum((v-v_past)**2) < 0.0001:
            #     break
            if np.allclose(v, v_past, atol=0.0001):
                break
        # PI
        for i in range(world_length):
            for j in range(world_length):
                if i != world_length - 1 or j != world_length - 1:
                    for k in range(env.move.shape[0]):
                        env.set_state(np.array([i, j]))
                        next_state, reward, done = env.step(k)
                        q[i,j,k] = reward + gamma * v[next_state[0],next_state[1]]
                    max_action = np.argmax(q[i, j, :])
                    pi[i, j, :] = 0
                    pi[i, j, max_action] = 1

        if  np.sum((v-v_oldpi)**2) < 0.0001:
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