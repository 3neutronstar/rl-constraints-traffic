# city-traffic_RL
City Optimization Reinforcement Learning based on https://github.com/3neutronstar/traffic-optimization_RL

## Decentralized DQN 
- Experiment
    1) Every 160s(depend on COMMON_PERIOD)
    2) Controls the each phase length that phases are in intersection system

- Agents
    1) Traffic Light Systems (Intersection)
    2) Have their own offset value
    3) Update itself asynchronously (according to offset value and COMMON_PERIOD value)

- State
    1) Queue Length(2 spaces per each inEdge, total 8 spaces) <br/>
    -> each number of vehicle is divided by max number of vehicles in an edge.(Normalize, TODO)
    2) Phase Length(If the number of phase is 4, spaces are composed of 4) <br/>
    -> (up,right,left,down) is divided by max period (Normalize)
    3) Searching method
        (1) Before phase ends, receive all the number of inflow vehicles

- Action (per each COMMON_PERIOD of intersection)
    1) Tuple of +,- of each phases (13)
    2) Length of phase time changes
    -> minimum value exists and maximum value exists

- Next State
    1) For agent, next state will be given after 160s.
    2) For environment, next state will be updated every 1s.

- Reward
    1) Max Pressure Control Theory (Reward = -pressure=-(inflow-outflow))

### Prerequisite
- python 3.7.9 above
- pytorch 1.7.1 above
- tensorboard 2.0.0 above

### How to use
check the condition state (throughput)
```shell script
    python ./Experiment/run.py simulate
``` 
Run in RL algorithm DQN (default device: cpu)
```shell script
    python ./Experiment/run.py train --gpu False
``` 
If you want to use other algorithm, use this code (ppo,super_dqn, ~~REINFORCE, a2c~~) 

```shell script
    python ./Experiment/run.py train --algorithm ppo
``` 
Check the RL performance that based on FRAP model [FRAP Paper]https://arxiv.org/abs/1905.04722
```shell script
    python ./Experiment/run.py train --model frap
``` 
Didn't check that it learns well. (Prototype)
- check the result
Tensorboard
```shell script
    tensorboard --logdir ./Experiment/training_data
``` 
Hyperparameter in json, model is in `./Experiment/training_data/[time you run]/model` directory.

- replay the model
```shell script
    python ./Experiment/run.py test --replay_name /replay_data in training_data dir/ --replay_epoch NUM
```

## Utils
gen_tllogic.py
```shell script
python /path/to/repo/util/gen_tllogic.py --file [xml]
```
graphcheck.py
```shell script
python /path/to/repo/util/gen_tllogic.py file_a file_b --type [edge or lane] --data speed
```
    - check the tensorboard
    `tensorboard --logdir tensorboard`
