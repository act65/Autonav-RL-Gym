import os
import json
import pandas as pd
import matplotlib.pyplot as plt

path = '/home/act65/catkin_ws/src/Autonav-RL-Gym/src/env/training_logs/'
logs = (os.path.join(path, fname) for fname in os.listdir(path))

def is_longer_than_N(fname, min=190, max=210):
    with open(fname, 'r') as f:
        log = f.read()
    l = len(log.split('\n'))
    return (min < l) and (l < max)

logs = filter(is_longer_than_N, logs)

# 'collision_count','ep_number','environment','goal_count', 'steps', 'reward_for_ep'

def parse_log(fname):
    data = []
    with open(fname, 'r') as f:
        for l in f:
            data.append(json.loads(l))
    return pd.DataFrame(data).values

for log in logs:
    x = parse_log(log)
    plt.plot(x[:, -1], label=x[0, 2])
plt.legend()
plt.show()
