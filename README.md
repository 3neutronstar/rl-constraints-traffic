# RL-constraints-traffic
Decentralized Deep Reinforcement Learning based Real-World Applicable Traffic Signal Optimization


## Decentralized DQN 

### Prerequisite
- python 3.7.9 above
- pytorch 1.7.1 above
- tensorboard 2.0.0 above

### How to use
check the condition state (throughput)
```shell script
    python run.py simulate --network [5x5grid, 5x5grid_v2, dunsan, dunsan_v2]
``` 
"Traffic data of Dunsan and Dunsan_v2 are classified by government of South Korea."

Run in RL algorithm DQN (default device: cpu)
```shell script
    python run.py train --network [5x5grid, 5x5grid_v2, dunsan, dunsan_v2]
``` 
"Traffic data of Dunsan and Dunsan_v2 are classified by government of South Korea."

- check the result
Tensorboard
```shell script
    tensorboard --logdir ./training_data
``` 
Hyperparameter in json, model is in `./training_data/[time you run]/model` directory.

- replay the model
```shell script
    python run.py test --replay_name /replay_data in training_data dir/ --replay_epoch NUM
```
### Performance
- Synthetic Data in 5x5grid(Straight Flow), 5x5grid_v2(Random Trips)

- Real World Data in Dunsan-dong, Daejeon, Korea
