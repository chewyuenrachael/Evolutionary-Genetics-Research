import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import expon, norm

def simulate_ils_branch_lengths(n=1000, scale=1.0):
    return expon.rvs(scale=scale, size=n)

def simulate_introgression_branch_lengths(n=1000, scale=1.0, m=0.3):
    # n_intro fraction from a different distribution
    n_intro = int(n * m)
    n_ils = n - n_intro
    ils_lengths = expon.rvs(scale=scale, size=n_ils)
    # Simulate “introgression” portion from a small normal distribution
    intro_lengths = norm.rvs(loc=0.1, scale=0.05, size=n_intro)
    intro_lengths = np.clip(intro_lengths, 0, None)  # no negative lengths
    return np.concatenate([ils_lengths, intro_lengths])

# Generate data
np.random.seed(42)
ils_data = simulate_ils_branch_lengths(n=1000, scale=1.0)
intro_data = simulate_introgression_branch_lengths(n=1000, scale=1.0, m=0.3)

# 1) Determine a common x-range
combined_max_x = max(ils_data.max(), intro_data.max())

# 2) Determine a common y-range:
#    We'll do quick hist/density calculations to see the max density.
#    Alternatively, you could pick a fixed y-limit if you know it.
hist_ils = sns.histplot(ils_data, kde=True, stat="density")
max_y_ils = hist_ils.axes.get_ylim()[1]
plt.close()  # We don't actually show this plot

hist_intro = sns.histplot(intro_data, kde=True, stat="density")
max_y_intro = hist_intro.axes.get_ylim()[1]
plt.close()

combined_max_y = max(max_y_ils, max_y_intro)

# Now do the real plotting
plt.figure(figsize=(12, 5))

# Plot for pure ILS
plt.subplot(1, 2, 1)
sns.histplot(ils_data, kde=True, color="skyblue", stat="density")
plt.title("Unimodal Distribution (Pure ILS)")
plt.xlabel("Branch Length")
plt.ylabel("Density")
plt.xlim(0, combined_max_x)      # same x-limit
plt.ylim(0, combined_max_y)      # same y-limit

# Plot for ILS + Introgression
plt.subplot(1, 2, 2)
sns.histplot(intro_data, kde=True, color="salmon", stat="density")
plt.title("Bimodal Distribution (ILS + Introgression)")
plt.xlabel("Branch Length")
plt.ylabel("Density")
plt.xlim(0, combined_max_x)
plt.ylim(0, combined_max_y)

plt.tight_layout()
plt.savefig("intro_bimodality_same_x_and_y.png", dpi=300)
plt.show()
