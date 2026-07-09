import numpy as np
import gymnasium as gym
from zero_copy_buffer import Transition, SimpleBuffer


class SimEnv:
    """
    One CartPole simulation worker.
    Runs episodes and writes PpoTransitions into the shared SimpleBuffer.
    Policy is random for phase 1 — the real actor will replace the
    action-selection line without changing anything else.
    """

    def __init__(self, buffer: SimpleBuffer, seed: int = 0):
        self.env    = gym.make("CartPole-v1")
        self.buffer = buffer
        self.seed   = seed
        self._total_written = 0

        # reset once so self.obs is always valid before collect() is called
        self._obs, _ = self.env.reset(seed=self.seed)

    def collect(self, n_steps: int) -> int:
        """
        Step the env n_steps times, writing each transition into the buffer.
        Resets automatically at episode end.
        Returns n_steps (always fully satisfied).
        """
        written = 0
        obs = self._obs

        while written < n_steps:
            # phase 1: random policy
            # phase 2: replace with actor.act(obs)
            action   = self.env.action_space.sample()
            log_prob = float(np.log(0.5))          # ln(uniform over 2 actions)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.write(
                Transition(
                    observation = tuple(obs.astype(np.float32)),
                    action      = float(action),
                    log_prob    = log_prob,
                    reward      = float(reward),
                    done        = float(done),
                )
            )

            written              += 1
            self._total_written  += 1
            obs = next_obs if not done else self._reset()

        self._obs = obs
        return written

    def _reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return obs

    def close(self) -> None:
        self.env.close()

    def __repr__(self) -> str:
        return (f"SimEnv(written={self._total_written}, "
                f"buf_len={self.buffer.len}/{self.buffer.capacity})")