import pandas as pd
import matplotlib.pyplot as plt

# 1. Setup file paths
files = {
    "od": "data/od_reward.csv",
    "origin": "data/origin_reward..csv",
}

# 2. Load datasets
dfs = {key: pd.read_csv(path) for key, path in files.items()}

# 3. Plot parameters
scale = 1000.0  # Scale axes to thousands
window = 100    # Smoothing window size
c_od = "#0072B2"
c_origin = "#D55E00"

# 4. Create the plot
# figsize=(4.5, 3) keeps the single-plot aspect similar to the multi-plot figure
fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))

for key, color, label in [
    ("od", c_od, "OD Prices"),
    ("origin", c_origin, "Origin Prices"),
]:
    df = dfs[key].sort_values("Step")

    # Identify the reward column (ignoring __MIN/__MAX columns)
    reward_col = [c for c in df.columns if "total_reward" in c and "__" not in c][0]

    df_processed = df.copy()
    df_processed["reward_smooth"] = (
        df_processed[reward_col].rolling(window=window, min_periods=1).mean()
    )

    ax.plot(
        df_processed["Step"] / scale,
        df_processed["reward_smooth"] / scale,
        color=color,
        label=label,
        linewidth=1.5,
    )

# Styling
ax.set_ylabel("Total Reward ($10^3$)", fontsize=10)
ax.set_xlabel("Training Episodes ($10^3$)", fontsize=10)
ax.grid(True, linestyle=":", alpha=0.5)
ax.tick_params(direction="out", which="both", top=False, right=False)
ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

plt.tight_layout()

# Save the plot
plt.savefig("od_versus_origin_reward_curves.png", dpi=300, bbox_inches="tight")
plt.show()
