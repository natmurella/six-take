import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


LOG_DIR = Path("logs/train")
CSV_PATH = LOG_DIR / "monitor.csv"

ROLLING_WINDOW = 500


def load_monitor_csv(path: Path) -> pd.DataFrame:
    with open(path, "r") as f:
        lines = f.readlines()

    # Skip comment lines (starting with '#')
    data_lines = [l for l in lines if not l.startswith("#")]

    from io import StringIO
    return pd.read_csv(StringIO("".join(data_lines)))


def plot_learning_curves(df: pd.DataFrame):
    episodes = np.arange(len(df))

    rewards = df["r"]
    win_rate = df["winner"] == 0
    score_diff = df["score_diff"]

    # Rolling stats
    reward_ma = rewards.rolling(ROLLING_WINDOW).mean()
    win_rate_ma = win_rate.rolling(ROLLING_WINDOW).mean()
    score_diff_ma = score_diff.rolling(ROLLING_WINDOW).mean()

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- Reward ---
    axes[0].plot(episodes, rewards, alpha=0.3, label="Episode reward")
    axes[0].plot(episodes, reward_ma, linewidth=2, label=f"Reward (MA {ROLLING_WINDOW})")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True)

    # --- Win rate ---
    axes[1].plot(episodes, win_rate_ma, linewidth=2)
    axes[1].set_ylabel("Win rate")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)

    # --- Score difference ---
    axes[2].plot(episodes, score_diff, alpha=0.3, label="Score diff")
    axes[2].plot(episodes, score_diff_ma, linewidth=2, label=f"Score diff (MA {ROLLING_WINDOW})")
    axes[2].set_ylabel("Score difference")
    axes[2].set_xlabel("Episodes")
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle("Training Performance")
    plt.tight_layout()
    # plt.show()
    plt.savefig(LOG_DIR / "learning_curves.png")


if __name__ == "__main__":
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    df = load_monitor_csv(CSV_PATH)
    plot_learning_curves(df)
