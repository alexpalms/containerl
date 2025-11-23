"""Simple DeepRL Agent for Anylogic stock problem."""

import os

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from containerl import AllowedTypes, CRLAgent, create_agent_server


class Agent(CRLAgent):
    """Simple DeepRL Agent for Anylogic stock problem."""

    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "stock": spaces.Box(
                    low=0.0, high=10_000.0, shape=(1,), dtype=np.float32
                ),
                "order_rate": spaces.Box(
                    low=0.0, high=50.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        self.action_space = spaces.Box(0, 50, shape=(1,), dtype=np.float32)

        model_path = os.path.join(os.path.dirname(__file__), "model.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found at {model_path}")

        self.agent = PPO.load(model_path, device="cpu")

    def get_action(self, observation: dict[str, AllowedTypes]) -> AllowedTypes:
        """Return the optimal action as calculated by the deepRL model."""
        obs = {
            "stock": np.array(observation["stock"]) / 5000.0 - 1,
            "order_rate": np.array(observation["order_rate"]) / 25.0 - 1,
        }
        obs = np.concatenate([obs["stock"], obs["order_rate"]])
        prediction = self.agent.predict(obs, deterministic=True)
        act = np.array([(prediction[0][0] + 1) * 25], dtype=np.float32)
        return act


if __name__ == "__main__":
    agent = Agent()
    create_agent_server(agent)
