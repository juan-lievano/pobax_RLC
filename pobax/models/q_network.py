from flax import linen as nn
from jax import numpy as jnp
from jax._src.nn.initializers import orthogonal, constant

from pobax.models.network import SimpleNN, ScannedRNN
from pobax.models.embedding import CNN


class QNetwork(nn.Module):
    env_name: str
    action_dim: int
    hidden_size: int = 128
    memoryless: bool = False
    is_image: bool = False

    def setup(self):
        if self.is_image:
            self.embedding = CNN(hidden_size=self.hidden_size)
        elif not self.memoryless:
            self.embedding = nn.Sequential([
                nn.Dense(self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
                nn.relu
            ])
        else:
            self.embedding = SimpleNN(hidden_size=self.hidden_size)

        if not self.memoryless:
            self.memory = ScannedRNN(hidden_size=self.hidden_size)

        self.q_head = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

    def __call__(self, hidden, x):
        obs_dict, dones = x
        obs = obs_dict.obs
        action_mask = obs_dict.action_mask
        embedding = self.embedding(obs)
        if not self.memoryless:
            rnn_in = (embedding, dones)
            hidden, embedding = self.memory(hidden, rnn_in)

        q_vals = self.q_head(embedding)

        if action_mask is not None:
            q_vals = jnp.where(action_mask, q_vals, jnp.finfo(q_vals.dtype).min)

        return hidden, q_vals

    def forward_with_embedding(self, hidden, x):
        """Like __call__ but also returns the internal embedding (before Q-head).

        For RNN agents this is the GRU output; for memoryless agents it is the
        SimpleNN output.  Used by the probe pipeline to capture activations.

        Returns: (hidden, q_vals, embedding)  where embedding has shape [..., H].
        """
        obs_dict, dones = x
        obs = obs_dict.obs
        action_mask = obs_dict.action_mask
        embedding = self.embedding(obs)
        if not self.memoryless:
            rnn_in = (embedding, dones)
            hidden, embedding = self.memory(hidden, rnn_in)

        q_vals = self.q_head(embedding)

        if action_mask is not None:
            q_vals = jnp.where(action_mask, q_vals, jnp.finfo(q_vals.dtype).min)

        return hidden, q_vals, embedding
