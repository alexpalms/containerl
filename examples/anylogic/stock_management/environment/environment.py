"""Custom Gym Environment for the Stock Management Game example model."""

import atexit
import logging
import os
import signal
from types import FrameType
from typing import Any, SupportsFloat, cast

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import psutil
from alpyne.data import SimStatus  # type: ignore
from alpyne.env import AlpyneEnv  # type: ignore
from alpyne.sim import AnyLogicSim  # type: ignore
from gymnasium import spaces
from gymnasium.wrappers.normalize import RunningMeanStd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

from containerl import create_environment_server

# Global list to track all subprocesses
_active_simulators: list[AnyLogicSim] = []


def cleanup_all_sims() -> None:
    """Cleanup function to kill all tracked processes."""
    while _active_simulators:
        sim = _active_simulators.pop()
        for pid in cast(list[int], sim._proc_pids):  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]
            try:
                process = psutil.Process(pid)
                for child in process.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
            except psutil.NoSuchProcess:
                pass


def signal_handler(
    signum: int,
    _frame: FrameType | None = None,
) -> Any:
    """Handle termination signals by cleaning up processes."""
    cleanup_all_sims()
    # Re-raise the signal to allow the Python runtime to handle it
    signal.signal(signum, signal.default_int_handler)
    os.kill(os.getpid(), signum)


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Register cleanup on normal exit
atexit.register(cleanup_all_sims)


class Environment(AlpyneEnv):  # type: ignore
    """
    Custom Gym Environment for the Stock Management Game example model.

    Observation:
        Type: Box(1)
        Name            Min     Max         Notes
        stock           0.0     10000.0
        last_stock      0.0     10000.0     Stock at last action
        demand          0.0     50.0        (Provided by the sim but not intended to be trained with)
        order_rate      0.0     50.0

    Actions:
        Type: Box(1)
        Name            Min     Max         Notes
        order_rate      0       50.0        per day

    Reward:
        1 if stock amount at 5000; falls off quartically

    Episode termination:
        If the stock amount falls beyond the configured limits
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("alpyne")
        self.logger.setLevel(logging.INFO)
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, "model.jar")
        if not os.path.exists(model_path):
            raise FileNotFoundError(rf"Missing file '{model_path}'.")

        sim = AnyLogicSim(
            model_path,
            config_defaults={
                "acquisition_lag_days": 10,
                "action_recurrence_days": 5,
                "stop_condition_limits": [0, 10000],
                "demand_volatility": 5,
            },
        )
        super().__init__(sim)

        # Track this simulation's processes
        _active_simulators.append(sim)

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
        self.render_mode = "rgb_array"  # Default render mode

        # Define target zone parameters
        self.target_zone_min: int = 4500
        self.target_zone_max: int = 5500

        self._last_obs: dict[str, np.ndarray] | None = None
        self._last_info: dict[str, Any] | None = None
        self._last_action: np.ndarray | None = None
        self._last_rew: SupportsFloat | None = None

        self.return_rms = RunningMeanStd(shape=())  # type: ignore
        self.returns = np.zeros(1)
        self.gamma = 0.99
        self.epsilon = 1e-8

        # Add history tracking for rendering
        self.history: dict[str, list[float | int]] = {
            "stock": [],
            "order_rate": [],
            "time": [],
        }
        self.current_time = 0

        # Initialize figure and axes for rendering
        self.fig: Figure | None = None
        self.ax: plt.Axes | None = None
        self.ax2: plt.Axes | None = None

    def _get_obs(self, status: SimStatus) -> dict[str, float]:
        return {
            "stock": np.clip(
                cast(float, status.observation["stock"]),
                cast(float, self.observation_space["stock"].low),  # type: ignore
                cast(float, self.observation_space["stock"].high),  # type: ignore
            ),
            "order_rate": np.clip(
                cast(float, status.observation["order_rate"]),
                cast(float, self.observation_space["order_rate"].low),  # type: ignore
                cast(float, self.observation_space["order_rate"].high),  # type: ignore
            ),
        }

    def _calc_reward(self, status: SimStatus) -> SupportsFloat:
        average_stock = 5000.0
        stock = cast(float, status.observation["stock"])
        coeff = -(((stock - average_stock) / (0.5 * average_stock)) ** 4)
        reward = max(
            -1.0,
            coeff + 1,
        )
        return reward

    def _normalize_reward(self, rew: SupportsFloat) -> np.ndarray:
        self.return_rms.update(self.returns)  # type: ignore
        normalization_factor = np.sqrt(
            cast(NDArray[np.floating[Any]], self.return_rms.var) + self.epsilon  # pyright: ignore[ reportUnknownMemberType]
        )
        rew_norm = cast(NDArray[np.floating[Any]], rew / normalization_factor)
        return rew_norm

    def _to_action(self, act: np.ndarray) -> dict[str, float]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return {"order_rate": act[0]}

    def _is_truncated(self, status: SimStatus) -> bool:
        return float(status.time) > 1000.0

    def _get_info(self, status: SimStatus) -> dict[str, int | float] | None:
        info = cast(dict[str, int | float] | None, super()._get_info(status))
        if info is not None:
            info["demand"] = cast(float, status.observation["demand"])
        return info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment."""
        obs, info = cast(
            tuple[dict[str, np.ndarray], dict[str, Any]],
            super().reset(seed=seed, options=options),
        )
        self._last_obs = obs
        self._last_info = info
        self._last_rew = 0

        # Reset history when environment resets
        self.history = {
            "stock": [],
            "order_rate": [],
            "time": [],
        }
        self.current_time = 0

        # Add initial observation to history
        self.history["stock"].append(float(obs["stock"][0]))
        self.history["order_rate"].append(float(obs["order_rate"][0]))
        self.history["time"].append(self.current_time)

        return obs, info

    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment."""
        obs, rew, term, trunc, info = cast(
            tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]],
            super().step(action),
        )
        rew = np.array([rew])
        self.returns = self.returns * self.gamma * (1 - term) + rew
        rew = self._normalize_reward(rew)
        rew = rew[0]
        self._last_action = action
        self._last_obs = obs
        self._last_info = info
        self._last_rew = rew

        # Update time and history
        self.current_time = info.get("time", self.current_time + 1)
        self.history["stock"].append(float(obs["stock"][0]))
        self.history["order_rate"].append(float(obs["order_rate"][0]))
        self.history["time"].append(self.current_time)

        if term or trunc:
            # Calculate percentage of time stock was outside target zone
            stock_values = self.history["stock"]
            total_steps = len(stock_values)
            outside_target = sum(
                1
                for s in stock_values
                if s < self.target_zone_min or s > self.target_zone_max
            )
            outside_percentage = (
                (outside_target / total_steps) * 100 if total_steps > 0 else 0
            )
            info["in_target_percentage"] = 100 - outside_percentage

            # Calculate standard deviation of order rate
            order_rate_values = self.history["order_rate"]
            if order_rate_values:
                order_rate_std = np.std(order_rate_values)
                info["order_rate_smoothness"] = (
                    float(1.0 / (order_rate_std + 1e-8)) * 100
                )

        return obs, rew, term, trunc, info

    def render(self) -> np.ndarray | None:
        """Render the environment."""
        if self.render_mode in ["human", "rgb_array"]:
            # Create figure and axis if they don't exist
            if self.fig is None or self.ax is None:
                plt.style.use(  # type: ignore
                    "dark_background"
                )  # Use dark background style
                self.fig, self.ax = plt.subplots(figsize=(10, 6))
                # Create a second y-axis sharing the same x-axis
                self.ax2 = self.ax.twinx()

                # Only use interactive mode for human rendering
                if self.render_mode == "human":
                    plt.ion()  # Turn on interactive mode

                # Set figure background to dark
                self.fig.patch.set_facecolor(  # type: ignore
                    "#1e1e1e"
                )  # Dark gray background
                self.ax.set_facecolor("#2d2d2d")  # Slightly lighter gray for plot area

            # Clear the current axes but keep the figure
            if self.ax2 is None:
                raise Exception("Axes not initialized properly.")
            self.ax.clear()
            self.ax2.clear()

            # Add shaded regions for "to be avoided" areas
            # Get the x-axis limits (will be updated after plotting)
            x_min = 0
            x_max = max(self.history["time"]) if self.history["time"] else 10

            # Add shaded region below 0 (danger zone)
            self.ax.add_patch(
                patches.Rectangle(
                    (x_min, -1000),  # (x, y)
                    x_max - x_min,  # width
                    1000,  # height
                    alpha=0.3,  # slightly higher transparency for dark mode
                    facecolor="#ff5555",  # brighter red for dark mode
                    edgecolor="none",
                    hatch="////",  # texture
                    zorder=0,  # ensure it's behind the plot
                )
            )

            # Add shaded region above 9000 (danger zone)
            self.ax.add_patch(
                patches.Rectangle(
                    (x_min, 7000),  # (x, y)
                    x_max - x_min,  # width
                    2000,  # height
                    alpha=0.3,  # transparency
                    facecolor="#ff5555",  # brighter red for dark mode
                    edgecolor="none",
                    hatch="////",  # texture
                    zorder=0,  # ensure it's behind the plot
                )
            )

            # Add target zone using the parameters
            self.ax.add_patch(
                patches.Rectangle(
                    (
                        x_min,
                        self.target_zone_min - 1000,
                    ),  # (x, y) (adjusted for the -1000 offset)
                    x_max - x_min,  # width
                    self.target_zone_max - self.target_zone_min,  # height
                    alpha=0.3,  # transparency
                    facecolor="#50fa7b",  # bright green for dark mode
                    edgecolor="none",
                    hatch="\\\\\\\\",  # texture (opposite direction)
                    zorder=0,  # ensure it's behind the plot
                )
            )

            # Create adjusted stock values (shifted down by 1000)
            adjusted_stock = [s - 1000 for s in self.history["stock"]]

            # Plot stock history on left axis (shifted down by 1000)
            (stock_line,) = self.ax.plot(  # pyright: ignore[reportUnknownMemberType]
                self.history["time"],
                adjusted_stock,
                "#8be9fd",
                linewidth=2,
                label="Stock",
            )

            # Plot current order rate on right axis
            (order_line,) = self.ax2.plot(  # pyright: ignore[reportUnknownMemberType]
                self.history["time"],
                self.history["order_rate"],
                "#ff79c6",
                linewidth=2,
                label="Order Rate",
            )

            # Update x-axis limits based on the data
            if self.history["time"]:
                self.ax.set_xlim(0, max(self.history["time"]))

            # Set labels and title
            self.ax.set_xlabel("Time", color="white", fontweight="bold")  # pyright: ignore[reportUnknownMemberType]
            self.ax.set_ylabel("Stock", color="#8be9fd", fontweight="bold")  # pyright: ignore[reportUnknownMemberType]
            self.ax2.set_ylabel(  # pyright: ignore[reportUnknownMemberType]
                "Order Rate",
                color="#ff79c6",
                fontweight="bold",
                rotation=270,
                labelpad=15,
            )
            self.ax2.yaxis.set_label_position("right")
            self.fig.suptitle(  # pyright: ignore[reportUnknownMemberType]
                "Stock Management Simulation",
                color="white",
                fontsize=14,
                fontweight="bold",
            )

            # Set colors for the axes
            self.ax.tick_params(axis="x", colors="white")  # pyright: ignore[reportUnknownMemberType]
            self.ax.tick_params(axis="y", labelcolor="#8be9fd")  # pyright: ignore[reportUnknownMemberType]
            self.ax2.tick_params(axis="y", labelcolor="#ff79c6")  # pyright: ignore[reportUnknownMemberType]

            # Set spines colors
            for spine in self.ax.spines.values():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                spine.set_color("#555555")  # pyright: ignore[reportUnknownMemberType]
            for spine in self.ax2.spines.values():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                spine.set_color("#555555")  # pyright: ignore[reportUnknownMemberType]

            # Calculate min/max values for display in annotations
            stock_min = min(self.history["stock"]) if self.history["stock"] else 0
            stock_max = max(self.history["stock"]) if self.history["stock"] else 10000
            adjusted_stock_min = stock_min - 1000
            adjusted_stock_max = stock_max - 1000
            order_min = (
                min(self.history["order_rate"]) if self.history["order_rate"] else 0
            )
            order_max = (
                max(self.history["order_rate"]) if self.history["order_rate"] else 50
            )

            # Set fixed y-axis limits
            self.ax.set_ylim(-1000, 9000)  # Fixed range for stock
            self.ax2.set_ylim(-1, 51)  # Fixed range for order rate

            # Add text annotations for min/max values (showing adjusted values)
            self.ax.text(  # pyright: ignore[reportUnknownMemberType]
                0.02,
                0.98,
                f"Stock Max: {adjusted_stock_max:.1f}",
                transform=self.ax.transAxes,
                verticalalignment="top",
                color="#8be9fd",
                fontweight="bold",
            )
            self.ax.text(  # pyright: ignore[reportUnknownMemberType]
                0.02,
                0.94,
                f"Stock Min: {adjusted_stock_min:.1f}",
                transform=self.ax.transAxes,
                verticalalignment="top",
                color="#8be9fd",
                fontweight="bold",
            )
            self.ax2.text(  # pyright: ignore[reportUnknownMemberType]
                0.98,
                0.98,
                f"Order Max: {order_max:.1f}",
                transform=self.ax2.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                color="#ff79c6",
                fontweight="bold",
            )
            self.ax2.text(  # pyright: ignore[reportUnknownMemberType]
                0.98,
                0.94,
                f"Order Min: {order_min:.1f}",
                transform=self.ax2.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                color="#ff79c6",
                fontweight="bold",
            )

            # Add danger zone labels
            self.ax.text(  # pyright: ignore[reportUnknownMemberType]
                0.98,
                0.06,
                "Danger Zone",
                transform=self.ax.transAxes,
                verticalalignment="bottom",
                horizontalalignment="right",
                color="#ff5555",
                fontweight="bold",
            )
            self.ax.text(  # pyright: ignore[reportUnknownMemberType]
                0.02,
                0.90,
                "Danger Zone",
                transform=self.ax.transAxes,
                verticalalignment="top",
                color="#ff5555",
                fontweight="bold",
            )

            # Add target zone label
            self.ax.text(  # pyright: ignore[reportUnknownMemberType]
                0.98,
                0.50,
                "Target Zone",
                transform=self.ax.transAxes,
                verticalalignment="center",
                horizontalalignment="right",
                color="#50fa7b",
                fontweight="bold",
            )

            # Add legend for both axes
            lines = [stock_line, order_line]
            labels = [line.get_label() for line in lines]
            legend = self.ax.legend(lines, labels, loc="upper center")  # pyright: ignore[reportUnknownMemberType]
            legend.get_frame().set_facecolor("#333333")  # pyright: ignore[reportUnknownMemberType]
            legend.get_frame().set_edgecolor("#555555")  # pyright: ignore[reportUnknownMemberType]
            for text in legend.get_texts():
                text.set_color("white")

            # Add grid for better readability (only on primary axis to avoid clutter)
            self.ax.grid(True, linestyle="--", alpha=0.3, color="#888888")  # pyright: ignore[reportUnknownMemberType]

            # Make x-axis show integer values only
            self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Add current values as text (showing adjusted stock value)
            current_stock = self.history["stock"][-1] if self.history["stock"] else 0
            adjusted_current_stock = current_stock - 1000
            current_order = (
                self.history["order_rate"][-1] if self.history["order_rate"] else 0
            )

            status_text = (
                f"Current Stock: {adjusted_current_stock:.1f}\n"
                f"Current Order Rate: {current_order:.1f}\n"
            )

            # Add text box with current values
            props = {
                "boxstyle": "round",
                "facecolor": "#333333",
                "edgecolor": "#555555",
                "alpha": 0.8,
            }
            self.ax.text(  # pyright: ignore[reportUnknownMemberType]
                0.02,
                0.02,
                status_text,
                transform=self.ax.transAxes,
                verticalalignment="bottom",
                color="white",
                bbox=props,
            )

            # Update the display
            self.fig.tight_layout()

            if self.render_mode == "human":
                plt.draw()
                plt.pause(  # pyright: ignore[reportUnknownMemberType]
                    0.01
                )  # Small pause to update the plot
                return None
            else:
                # Convert plot to RGB array
                canvas = FigureCanvas(self.fig)
                canvas.draw()

                # Get the RGB buffer from the figure
                width, height = self.fig.get_size_inches() * self.fig.get_dpi()
                width, height = int(width), int(height)

                # Get the image array from the canvas
                img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                img = img.reshape((height, width, 4))

                # Convert RGBA to RGB
                rgb_img = img[:, :, :3]

                return rgb_img
        else:
            # Default text rendering
            self.logger.info(
                f"{self._last_action} -> {self._last_obs} = {self._last_rew:.2f} ({self._last_info})"
            )
            return None

    def close(self) -> None:
        """Ensure cleanup when environment is closed."""
        # Close the matplotlib figure if it exists
        if hasattr(self, "fig") and self.fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.ax2 = None

        cleanup_all_sims()
        super().close()


if __name__ == "__main__":
    create_environment_server(Environment)
