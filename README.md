# Reinforcement Learning: Value Iteration & Q-Learning

**Course**: Fundamentals and Applications of Artificial Intelligence — Spring 1404  
**University**: Amirkabir University of Technology  

## Overview

Implementation of reinforcement learning algorithms tested on Gridworld, Crawler, and Pacman environments.

## Implemented Algorithms

| Component | Description |
|-----------|-------------|
| `ValueIterationAgent` | Batch value iteration for known MDPs |
| `AsynchronousValueIterationAgent` | Cyclic (single-state-per-step) value iteration |
| `PrioritizedSweepingValueIterationAgent` | Priority-queue-based value iteration |
| `QLearningAgent` | Q-learning with ε-greedy exploration |
| `ApproximateQAgent` | Q-learning with feature-based function approximation |
| `analysis.py` | Parameter tuning answers (BridgeGrid, DiscountGrid policies) |

## Modified Files

- `valueIterationAgents.py` — value iteration variants
- `qlearningAgents.py` — Q-learning and approximate Q-learning
- `analysis.py` — answers to policy/parameter analysis questions

## Running

```bash
# Gridworld (manual)
python gridworld.py -m

# Value iteration
python gridworld.py -a value -i 100 -k 10

# Async value iteration
python gridworld.py -a asynchvalue -i 1000 -k 10

# Prioritized sweeping
python gridworld.py -a priosweepvalue -i 1000

# Q-learning
python gridworld.py -a q -k 100 -e 0.3

# Crawler robot
python crawler.py

# Pacman Q-learning
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

# Approximate Q-learning
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid

# Run autograder
python autograder.py
python autograder.py -q q6
```

## Base Project

Based on the [UC Berkeley CS188 Reinforcement Learning Project](http://ai.berkeley.edu/reinforcement.html).
