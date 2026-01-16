"""
Test script for random_data_generator.py

Generates data and provides descriptive analysis including:
- Summary statistics
- Distribution plots for outcomes and treatment effects
- Region type analysis
- Treatment effect heterogeneity visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from random_data_generator import generate_random_comparison_data, get_data_summary


def plot_outcome_distributions(YF, YC, T, save_path=None):
    """Plot distributions of firm and user outcomes by treatment group."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Firm outcome distributions
    axes[0, 0].hist(
        YF[T == 0], bins=50, alpha=0.7, label="Control", color="blue", density=True
    )
    axes[0, 0].hist(
        YF[T == 1], bins=50, alpha=0.7, label="Treated", color="red", density=True
    )
    axes[0, 0].set_xlabel("Firm Outcome (YF)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Firm Outcome Distribution by Treatment Group")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # User outcome distributions
    axes[0, 1].hist(
        YC[T == 0], bins=50, alpha=0.7, label="Control", color="blue", density=True
    )
    axes[0, 1].hist(
        YC[T == 1], bins=50, alpha=0.7, label="Treated", color="red", density=True
    )
    axes[0, 1].set_xlabel("User Outcome (YC)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("User Outcome Distribution by Treatment Group")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Firm outcome box plots
    axes[1, 0].boxplot([YF[T == 0], YF[T == 1]], labels=["Control", "Treated"])
    axes[1, 0].set_ylabel("Firm Outcome (YF)")
    axes[1, 0].set_title("Firm Outcome Box Plot by Treatment Group")
    axes[1, 0].grid(True, alpha=0.3)

    # User outcome box plots
    axes[1, 1].boxplot([YC[T == 0], YC[T == 1]], labels=["Control", "Treated"])
    axes[1, 1].set_ylabel("User Outcome (YC)")
    axes[1, 1].set_title("User Outcome Box Plot by Treatment Group")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved outcome distributions plot to {save_path}")
    plt.show()


def plot_treatment_effects(tauF, tauC, region_type, save_path=None):
    """Plot distributions and scatter of treatment effects."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Treatment effect distributions
    axes[0, 0].hist(tauF, bins=50, alpha=0.7, color="blue", edgecolor="black")
    axes[0, 0].set_xlabel("Firm Treatment Effect (tauF)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Firm Treatment Effects")
    axes[0, 0].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero effect")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(tauC, bins=50, alpha=0.7, color="green", edgecolor="black")
    axes[0, 1].set_xlabel("User Treatment Effect (tauC)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of User Treatment Effects")
    axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero effect")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Scatter plot of treatment effects
    colors = {1: "green", 2: "red", 3: "blue", 4: "gray"}
    labels = {1: "Both +", 2: "Firm+ User-", 3: "Firm- User+", 4: "Both -"}
    for rt in [1, 2, 3, 4]:
        mask = region_type == rt
        axes[1, 0].scatter(
            tauF[mask], tauC[mask], alpha=0.5, s=10, c=colors[rt], label=labels[rt]
        )
    axes[1, 0].axhline(0, color="black", linestyle="-", linewidth=0.5)
    axes[1, 0].axvline(0, color="black", linestyle="-", linewidth=0.5)
    axes[1, 0].set_xlabel("Firm Treatment Effect (tauF)")
    axes[1, 0].set_ylabel("User Treatment Effect (tauC)")
    axes[1, 0].set_title("Treatment Effects Scatter Plot by Region Type")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Region type distribution
    unique, counts = np.unique(region_type, return_counts=True)
    region_labels = [labels[r] for r in unique]
    axes[1, 1].bar(
        region_labels,
        counts,
        color=[colors[r] for r in unique],
        edgecolor="black",
        alpha=0.7,
    )
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Region Type Distribution")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved treatment effects plot to {save_path}")
    plt.show()


def plot_heterogeneity_analysis(X, tauF, tauC, n_features_to_plot=3, save_path=None):
    """Plot treatment effect heterogeneity across features."""
    fig, axes = plt.subplots(
        2, n_features_to_plot, figsize=(5 * n_features_to_plot, 10)
    )

    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    for i in range(min(n_features_to_plot, X.shape[1])):
        # Firm effects by feature
        axes[0, i].scatter(X[:, i], tauF, alpha=0.3, s=5, color="blue")
        axes[0, i].set_xlabel(f"Feature {i}")
        axes[0, i].set_ylabel("Firm Treatment Effect (tauF)")
        axes[0, i].set_title(f"Firm Effect Heterogeneity: Feature {i}")
        axes[0, i].grid(True, alpha=0.3)

        # User effects by feature
        axes[1, i].scatter(X[:, i], tauC, alpha=0.3, s=5, color="green")
        axes[1, i].set_xlabel(f"Feature {i}")
        axes[1, i].set_ylabel("User Treatment Effect (tauC)")
        axes[1, i].set_title(f"User Effect Heterogeneity: Feature {i}")
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved heterogeneity analysis plot to {save_path}")
    plt.show()


def print_functional_form(functional_form):
    """Print the functional form (coefficients) used to generate the data."""
    print("=" * 80)
    print("FUNCTIONAL FORM (COEFFICIENTS)")
    print("=" * 80)

    print(f"\nBaseline Outcome Functions:")
    print(
        f"  Firm baseline (mu_F): Uses all {len(functional_form['baseline_coef_F'])} features"
    )
    print(f"    Coefficients: {functional_form['baseline_coef_F']}")
    print(
        f"  User baseline (mu_C): Uses all {len(functional_form['baseline_coef_C'])} features"
    )
    print(f"    Coefficients: {functional_form['baseline_coef_C']}")

    print(f"\nTreatment Effect Functions:")
    print(f"  Firm treatment effect (tau_F):")
    print(f"    Active features: {functional_form['n_active_features']}")
    print(f"    Selected feature indices: {functional_form['firm_selected_features']}")
    print(f"    Coefficients: {functional_form['effect_coef_F']}")
    print(f"    Feature-coefficient mapping:")
    for i, feat_idx in enumerate(functional_form["firm_selected_features"]):
        print(f"      X[{feat_idx}]: {functional_form['effect_coef_F'][i]:.6f}")

    print(f"  User treatment effect (tau_C):")
    print(f"    Active features: {len(functional_form['user_selected_features'])}")
    print(f"    Selected feature indices: {functional_form['user_selected_features']}")
    print(f"    Shared features: {functional_form['shared_features']}")
    print(
        f"    Overlap: {functional_form['overlap']:.2f} ({len(functional_form['shared_features'])} shared)"
    )
    print(f"    Coefficients: {functional_form['effect_coef_C']}")
    print(f"    Feature-coefficient mapping:")
    for i, feat_idx in enumerate(functional_form["user_selected_features"]):
        print(f"      X[{feat_idx}]: {functional_form['effect_coef_C'][i]:.6f}")

    print(f"\nParameters:")
    print(f"  Intensity: {functional_form['intensity']:.4f}")
    print(f"  Overlap: {functional_form['overlap']:.4f}")
    print(f"  Active features: {functional_form['n_active_features']}")

    print("=" * 80)


def print_descriptive_statistics(X, T, YF, YC, tauF, tauC, region_type):
    """Print comprehensive descriptive statistics."""
    print("=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)

    # Basic data info
    print(f"\nData Dimensions:")
    print(f"  Number of observations: {len(X)}")
    print(f"  Number of features: {X.shape[1]}")
    print(
        f"  Treatment rate: {T.mean():.4f} ({T.sum()} treated, {len(T) - T.sum()} control)"
    )

    # Feature statistics
    print(f"\nFeature Statistics (first 5 features):")
    for i in range(min(5, X.shape[1])):
        print(
            f"  Feature {i}: mean={X[:, i].mean():.4f}, std={X[:, i].std():.4f}, "
            f"min={X[:, i].min():.4f}, max={X[:, i].max():.4f}"
        )

    # Outcome statistics
    summary = get_data_summary(X, T, YF, YC)
    print(f"\nFirm Outcome (YF) Statistics:")
    print(f"  Overall mean: {YF.mean():.4f}, std: {YF.std():.4f}")
    print(f"  Treated mean: {summary['firm_treated_mean']:.4f}")
    print(f"  Control mean: {summary['firm_control_mean']:.4f}")
    print(f"  Average Treatment Effect (ATE): {summary['firm_ate']:.4f}")

    print(f"\nUser Outcome (YC) Statistics:")
    print(f"  Overall mean: {YC.mean():.4f}, std: {YC.std():.4f}")
    print(f"  Treated mean: {summary['user_treated_mean']:.4f}")
    print(f"  Control mean: {summary['user_control_mean']:.4f}")
    print(f"  Average Treatment Effect (ATE): {summary['user_ate']:.4f}")

    # Treatment effect statistics
    print(f"\nFirm Treatment Effect (tauF) Statistics:")
    print(f"  Mean: {tauF.mean():.4f}, std: {tauF.std():.4f}")
    print(f"  Min: {tauF.min():.4f}, Max: {tauF.max():.4f}")
    print(f"  Median: {np.median(tauF):.4f}")
    print(f"  Positive effects: {(tauF > 0).sum()} ({(tauF > 0).mean()*100:.2f}%)")
    print(f"  Negative effects: {(tauF <= 0).sum()} ({(tauF <= 0).mean()*100:.2f}%)")

    print(f"\nUser Treatment Effect (tauC) Statistics:")
    print(f"  Mean: {tauC.mean():.4f}, std: {tauC.std():.4f}")
    print(f"  Min: {tauC.min():.4f}, Max: {tauC.max():.4f}")
    print(f"  Median: {np.median(tauC):.4f}")
    print(f"  Positive effects: {(tauC > 0).sum()} ({(tauC > 0).mean()*100:.2f}%)")
    print(f"  Negative effects: {(tauC <= 0).sum()} ({(tauC <= 0).mean()*100:.2f}%)")

    # Region type statistics
    print(f"\nRegion Type Distribution:")
    region_labels = {
        1: "Both + (Win-Win)",
        2: "Firm+ User- (Trade-off)",
        3: "Firm- User+ (Trade-off)",
        4: "Both - (Lose-Lose)",
    }
    for rt in [1, 2, 3, 4]:
        count = (region_type == rt).sum()
        pct = 100 * count / len(region_type)
        print(f"  Region {rt} ({region_labels[rt]}): {count} observations ({pct:.2f}%)")

    # Correlation analysis
    print(f"\nCorrelations:")
    print(f"  Correlation between tauF and tauC: {np.corrcoef(tauF, tauC)[0, 1]:.4f}")
    print(f"  Correlation between YF and YC: {np.corrcoef(YF, YC)[0, 1]:.4f}")
    print(f"  Correlation between YF and T: {np.corrcoef(YF, T)[0, 1]:.4f}")
    print(f"  Correlation between YC and T: {np.corrcoef(YC, T)[0, 1]:.4f}")

    print("=" * 80)


def main():
    """Main function to run the test."""
    print("Generating random comparison data...")
    print("-" * 80)

    # Generate data
    X, T, YF, YC, tauF, tauC, region_type, functional_form = (
        generate_random_comparison_data(
            n_users=10000,
            n_features=10,
            n_active_features=5,
            overlap=0.5,
            intensity=1.0,
            effect_noise_std=0.0,
            firm_outcome_noise_std=1.0,
            user_outcome_noise_std=1.0,
            random_seed=42,
        )
    )

    print("Data generation complete!")
    print(f"Generated {len(X)} observations with {X.shape[1]} features")
    print("-" * 80)

    # Print functional form
    print_functional_form(functional_form)
    print()

    # Print descriptive statistics
    print_descriptive_statistics(X, T, YF, YC, tauF, tauC, region_type)

    # Create plots
    print("\nGenerating plots...")
    print("-" * 80)

    # Plot outcome distributions
    plot_outcome_distributions(YF, YC, T, save_path="data/outcome_distributions.png")

    # Plot treatment effects
    plot_treatment_effects(
        tauF, tauC, region_type, save_path="data/treatment_effects.png"
    )

    # Plot heterogeneity analysis
    plot_heterogeneity_analysis(
        X, tauF, tauC, n_features_to_plot=3, save_path="data/heterogeneity_analysis.png"
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    import os

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    main()
