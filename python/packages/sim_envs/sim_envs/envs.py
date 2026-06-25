
from types import Callable
import gymnasium as gym

def _make_env(gym_id, seed, idx, render_mode=None) -> Callable:

    def thunk():
      
      print(f"Creating {idx} Environment for : {gym_id}, with render_mode: {render_mode}")
      env = gym.make(gym_id, render_mode=render_mode)
      # env = gym.wrappers.RecordEpisodeStatistics(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env

    return thunk

def make_envs(args) -> gym.vector.AsyncVectorEnv:
  
    return gym.vector.AsyncVectorEnv(
        [
            _make_env(
                args.gym_id,
                args.seed + i,
                i,
                args.render,
            )
            for i in range(args.num_envs)
        ]
    )