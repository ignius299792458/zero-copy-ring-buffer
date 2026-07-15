import torch
import torch.nn as nn
import numpy as np
from zero_copy_buffer import RingBuffer


class TinyActor(nn.Module):
    """
    Minimal actor: obs(4) -> logits(2).
    Same architecture as ppo_impl — 2 hidden layers of 64, tanh activations,
    linear output (no softmax here; Categorical handles that).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, 4) -> logits: (B, 2)
        return self.net(obs)


class ModelReader:
    """
    Reads a batch of transitions from the Rust buffer and runs one
    actor forward pass. Phase 2 proves the data path works end-to-end.
    Phase 2 plugs the real PPO loss in here without changing the read path.
    """

    def __init__(
        self,
        buffer:     RingBuffer,
        batch_size: int = 64,
        device:     str = "cpu",
    ):
        self.buffer     = buffer
        self.batch_size = batch_size
        self.device     = torch.device(device)
        self.actor      = TinyActor().to(self.device)
        self._n_reads   = 0

    def read_and_forward(self) -> dict | None:
        """
        Pull batch_size transitions from the buffer.
        Returns a dict with tensors, or None if the buffer is not full enough.
        """
        if self.buffer.len < self.batch_size:
            return None

        batch = self.buffer.pop_batch(self.batch_size)

        # unpack Rust structs -> numpy -> tensor (one allocation)
        obs_np  = np.array([t.observation for t in batch], dtype=np.float32)
        act_np  = np.array([t.action      for t in batch], dtype=np.float32)
        rew_np  = np.array([t.reward      for t in batch], dtype=np.float32)
        done_np = np.array([t.done        for t in batch], dtype=np.float32)
        lp_np   = np.array([t.log_prob    for t in batch], dtype=np.float32)

        obs_t = torch.from_numpy(obs_np).to(self.device)   # (B, 4)

        with torch.no_grad():
            logits = self.actor(obs_t)                      # (B, 2)

        self._n_reads += 1
        return {
            "obs":      obs_t,
            "actions":  act_np,
            "rewards":  rew_np,
            "dones":    done_np,
            "log_probs": lp_np,
            "logits":   logits,
            "n":        len(batch),
        }

    def __repr__(self) -> str:
        return (f"ModelReader(reads={self._n_reads}, "
                f"batch={self.batch_size}, device={self.device})")