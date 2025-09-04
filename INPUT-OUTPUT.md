# INPUT
```
    prototyp:
        obs, reward, terminated, truncated, _ = env.step(action)

   Zwracane przez env.step()
     observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
         An example is a numpy array containing the positions and velocities of the pole in CartPole.
     reward (SupportsFloat): The reward as a result of taking the action.
     terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
         which can be positive or negative. An example is reaching the goal state or moving into the lava from
         the Sutton and Barto Gridworld. If true, the user needs to call :meth:`reset`.
     truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
         Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
         Can be used to end the episode prematurely before a terminal state is reached.
         If true, the user needs to call :meth:`reset`.
```

# OUTPUT
```
 self.action = np.array([0.0, 0.0, 0.0])  # [steer(left/right), gas, brake]
```

