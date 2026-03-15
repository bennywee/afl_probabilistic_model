import arviz as az
import numpy as np
import pandas as pd

# Extract posterior samples
mu_alpha_beta_posterior = idata.posterior["mu_alpha_beta"].values  # shape: (chains, draws, 3)
alpha_beta_home_posterior = idata.posterior["alpha_beta_home"].values  # shape: (chains, draws, home_games, 2)
alpha_beta_away_posterior = idata.posterior["alpha_beta_away"].values  # shape: (chains, draws, away_games, 2)

# Flatten chains and draws
mu_samples = mu_alpha_beta_posterior.reshape(-1, 3)  # (total_samples, 3)
home_samples = alpha_beta_home_posterior.reshape(-1, *alpha_beta_home_posterior.shape[2:])
away_samples = alpha_beta_away_posterior.reshape(-1, *alpha_beta_away_posterior.shape[2:])

print("=" * 70)
print("TOTAL EFFECT OF HOME_PERC AND AWAY_PERC (Including Intercept)")
print("=" * 70)

# HOME_PERC effect: (mu_alpha_beta[0] + alpha_beta_home[:, 0]) + (mu_alpha_beta[1] + alpha_beta_home[:, 1])
print("\nHOME PERCENTAGE EFFECT (by games played)")
print("-" * 70)

# Total intercept effect
intercept_home = mu_samples[:, 0, None] + home_samples[:, :, 0]  # (samples, games)
intercept_home_means = intercept_home.mean(axis=0)
intercept_home_stds = intercept_home.std(axis=0)

# Slope effect for home percentage
slope_home = mu_samples[:, 1, None] + home_samples[:, :, 1]  # (samples, games)
slope_home_means = slope_home.mean(axis=0)
slope_home_stds = slope_home.std(axis=0)

# Total combined effect
total_home_effect = intercept_home + slope_home
total_home_means = total_home_effect.mean(axis=0)
total_home_stds = total_home_effect.std(axis=0)

home_perc_df = pd.DataFrame({
    'Games_Played': idata.posterior["home_games_played"].values,


    'Intercept_Mean': intercept_home_means,
    'Intercept_Std': intercept_home_stds,
    'Slope_Mean': slope_home_means,
    'Slope_Std': slope_home_stds,
    'Total_Effect_Mean': total_home_means,
    'Total_Effect_Std': total_home_stds,
})
print(home_perc_df)

# AWAY_PERC effect: (mu_alpha_beta[0] + alpha_beta_away[:, 0]) + (mu_alpha_beta[2] + alpha_beta_away[:, 1])
print("\n" + "=" * 70)
print("AWAY PERCENTAGE EFFECT (by games played)")
print("-" * 70)

# Total intercept effect
intercept_away = mu_samples[:, 0, None] + away_samples[:, :, 0]  # (samples, games)
intercept_away_means = intercept_away.mean(axis=0)
intercept_away_stds = intercept_away.std(axis=0)

# Slope effect for away percentage
slope_away = mu_samples[:, 2, None] + away_samples[:, :, 1]  # (samples, games)
slope_away_means = slope_away.mean(axis=0)
slope_away_stds = slope_away.std(axis=0)

# Total combined effect
total_away_effect = intercept_away + slope_away
total_away_means = total_away_effect.mean(axis=0)
total_away_stds = total_away_effect.std(axis=0)

away_perc_df = pd.DataFrame({
    'Games_Played': idata.posterior["away_games_played"].values,
    'Intercept_Mean': intercept_away_means,
    'Intercept_Std': intercept_away_stds,
    'Slope_Mean': slope_away_means,
    'Slope_Std': slope_away_stds,
    'Total_Effect_Mean': total_away_means,
    'Total_Effect_Std': total_away_stds,
})
print(away_perc_df)

# Comparison plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Home percentage - Intercept component
axes[0, 0].scatter(home_perc_df['Games_Played'], home_perc_df['Intercept_Mean'], alpha=0.6)
axes[0, 0].fill_between(home_perc_df['Games_Played'],
                         home_perc_df['Intercept_Mean'] - home_perc_df['Intercept_Std'],
                         home_perc_df['Intercept_Mean'] + home_perc_df['Intercept_Std'], alpha=0.2)
axes[0, 0].set_title('Home - Intercept Component')
axes[0, 0].set_xlabel('Games Played')
axes[0, 0].set_ylabel('Effect')
axes[0, 0].grid(True, alpha=0.3)

# Home percentage - Slope component
axes[0, 1].scatter(home_perc_df['Games_Played'], home_perc_df['Slope_Mean'], alpha=0.6)
axes[0, 1].fill_between(home_perc_df['Games_Played'],
                         home_perc_df['Slope_Mean'] - home_perc_df['Slope_Std'],
                         home_perc_df['Slope_Mean'] + home_perc_df['Slope_Std'], alpha=0.2)
axes[0, 1].set_title('Home - Slope Component (Home Win %)')
axes[0, 1].set_xlabel('Games Played')
axes[0, 1].set_ylabel('Effect')
axes[0, 1].grid(True, alpha=0.3)

# Away percentage - Intercept component
axes[1, 0].scatter(away_perc_df['Games_Played'], away_perc_df['Intercept_Mean'], alpha=0.6, color='orange')
axes[1, 0].fill_between(away_perc_df['Games_Played'],
                         away_perc_df['Intercept_Mean'] - away_perc_df['Intercept_Std'],
                         away_perc_df['Intercept_Mean'] + away_perc_df['Intercept_Std'], alpha=0.2)
axes[1, 0].set_title('Away - Intercept Component')
axes[1, 0].set_xlabel('Games Played')
axes[1, 0].set_ylabel('Effect')
axes[1, 0].grid(True, alpha=0.3)

# Away percentage - Slope component
axes[1, 1].scatter(away_perc_df['Games_Played'], away_perc_df['Slope_Mean'], alpha=0.6, color='orange')
axes[1, 1].fill_between(away_perc_df['Games_Played'],
                         away_perc_df['Slope_Mean'] - away_perc_df['Slope_Std'],
                         away_perc_df['Slope_Mean'] + away_perc_df['Slope_Std'], alpha=0.2)
axes[1, 1].set_title('Away - Slope Component (Away Win %)')
axes[1, 1].set_xlabel('Games Played')
axes[1, 1].set_ylabel('Effect')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('percentage_effects_components.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY COMPARISON - TOTAL EFFECTS")
print("=" * 70)
summary_df = pd.DataFrame({
    'Effect': ['Home Percentage', 'Away Percentage'],
    'Total_Mean': [total_home_means.mean(), total_away_means.mean()],
    'Total_Std': [total_home_stds.mean(), total_away_stds.mean()],
    'Min': [total_home_means.min(), total_away_means.min()],
    'Max': [total_home_means.max(), total_away_means.max()],
    'Range': [total_home_means.max() - total_home_means.min(), total_away_means.max() - total_away_means.min()],
})
print(summary_df)


