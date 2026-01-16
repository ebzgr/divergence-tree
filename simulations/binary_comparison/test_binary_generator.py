"""
Test script for binary_data_generator.py

Generates data and provides descriptive analysis including:
- Summary statistics
- Distribution plots for outcomes and treatment effects
- Activation pattern analysis
- Region type analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from binary_data_generator import generate_binary_comparison_data, get_data_summary


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
    axes[1, 0].boxplot([YF[T == 0], YF[T == 1]], tick_labels=["Control", "Treated"])
    axes[1, 0].set_ylabel("Firm Outcome (YF)")
    axes[1, 0].set_title("Firm Outcome Box Plot by Treatment Group")
    axes[1, 0].grid(True, alpha=0.3)

    # User outcome box plots
    axes[1, 1].boxplot([YC[T == 0], YC[T == 1]], tick_labels=["Control", "Treated"])
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


def print_functional_form(functional_form):
    """Print the functional form (combinations and coefficients) used to generate the data."""
    print("=" * 80)
    print("FUNCTIONAL FORM (COMBINATIONS AND COEFFICIENTS)")
    print("=" * 80)

    print(f"\nCategorical Variables:")
    print(f"  Number of variables (k): {functional_form['k']}")
    print(f"  Categories per variable: {functional_form['n_categories']}")
    total_combinations = np.prod(functional_form["n_categories"])
    print(f"  Total possible combinations: {total_combinations}")

    print(f"\nBaseline Outcome Functions:")
    print(
        f"  Firm baseline (mu_F): Uses all {len(functional_form['baseline_coef_F'])} binary features"
    )
    print(
        f"  User baseline (mu_C): Uses all {len(functional_form['baseline_coef_C'])} binary features"
    )

    print(f"\nFirm Treatment Effect Activation:")
    print(f"  Number of activating combinations: {functional_form['m_firm']}")
    print(f"  Combinations and signs:")
    for i, (combo, sign) in enumerate(
        zip(functional_form["firm_combinations"], functional_form["firm_signs"])
    ):
        sign_str = "+" if sign > 0 else "-"
        print(
            f"    Combination {i+1}: {combo} -> sign={sign_str}, effect={sign * functional_form['intensity']:.2f}"
        )

    print(f"\nUser Treatment Effect Activation:")
    print(f"  Number of activating combinations: {functional_form['m_user']}")
    print(f"  Similarity: {functional_form['similarity']:.2f}")
    print(f"  Combinations and signs:")
    for i, (combo, sign) in enumerate(
        zip(functional_form["user_combinations"], functional_form["user_signs"])
    ):
        sign_str = "+" if sign > 0 else "-"
        print(
            f"    Combination {i+1}: {combo} -> sign={sign_str}, effect={sign * functional_form['intensity']:.2f}"
        )

    # Check shared combinations
    firm_set = set(functional_form["firm_combinations"])
    user_set = set(functional_form["user_combinations"])
    shared = firm_set & user_set
    print(f"\nShared Combinations: {len(shared)}")
    if len(shared) > 0:
        for combo in shared:
            firm_idx = functional_form["firm_combinations"].index(combo)
            user_idx = functional_form["user_combinations"].index(combo)
            firm_sign = functional_form["firm_signs"][firm_idx]
            user_sign = functional_form["user_signs"][user_idx]
            print(f"  {combo}: Firm sign={firm_sign}, User sign={user_sign}")

    print(f"\nParameters:")
    print(f"  Intensity: {functional_form['intensity']:.4f}")
    print(f"  Effect noise std: {functional_form['effect_noise_std']:.4f}")
    print(f"  Similarity: {functional_form['similarity']:.4f}")

    print("=" * 80)


def print_descriptive_statistics(
    X, T, YF, YC, tauF, tauC, region_type, functional_form
):
    """Print comprehensive descriptive statistics."""
    print("=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)

    # Basic data info
    print(f"\nData Dimensions:")
    print(f"  Number of observations: {len(X)}")
    print(f"  Number of binary features: {X.shape[1]}")
    print(f"  Number of categorical variables: {functional_form['k']}")
    print(
        f"  Treatment rate: {T.mean():.4f} ({T.sum()} treated, {len(T) - T.sum()} control)"
    )

    # Binary feature statistics
    print(f"\nBinary Feature Statistics:")
    print(f"  Feature sparsity: {(X == 0).mean():.4f} (proportion of zeros)")
    print(f"  Features per observation (should be k): {X.sum(axis=1).mean():.2f}")

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
    # Count activated vs inactive
    # Activated: close to +intensity, Inactive: close to -intensity (before noise)
    intensity = functional_form["intensity"]
    effect_noise_std = functional_form.get("effect_noise_std", 0.0)
    # Use a threshold that accounts for noise: within 2*std of target
    threshold = max(0.1, 2 * effect_noise_std)
    activated_F = np.abs(tauF - intensity) < threshold  # Close to +intensity
    inactive_F = np.abs(tauF + intensity) < threshold  # Close to -intensity
    print(
        f"  Activated (tauF ≈ +{intensity}): {activated_F.sum()} ({(activated_F).mean()*100:.2f}%)"
    )
    print(
        f"  Inactive (tauF ≈ -{intensity}): {inactive_F.sum()} ({(inactive_F).mean()*100:.2f}%)"
    )

    print(f"\nUser Treatment Effect (tauC) Statistics:")
    print(f"  Mean: {tauC.mean():.4f}, std: {tauC.std():.4f}")
    print(f"  Min: {tauC.min():.4f}, Max: {tauC.max():.4f}")
    print(f"  Median: {np.median(tauC):.4f}")
    print(f"  Positive effects: {(tauC > 0).sum()} ({(tauC > 0).mean()*100:.2f}%)")
    print(f"  Negative effects: {(tauC <= 0).sum()} ({(tauC <= 0).mean()*100:.2f}%)")
    # Count activated vs inactive
    # Activated: close to +intensity, Inactive: close to -intensity (before noise)
    intensity = functional_form["intensity"]
    effect_noise_std = functional_form.get("effect_noise_std", 0.0)
    # Use a threshold that accounts for noise: within 2*std of target
    threshold = max(0.1, 2 * effect_noise_std)
    activated_C = np.abs(tauC - intensity) < threshold  # Close to +intensity
    inactive_C = np.abs(tauC + intensity) < threshold  # Close to -intensity
    print(
        f"  Activated (tauC ≈ +{intensity}): {activated_C.sum()} ({(activated_C).mean()*100:.2f}%)"
    )
    print(
        f"  Inactive (tauC ≈ -{intensity}): {inactive_C.sum()} ({(inactive_C).mean()*100:.2f}%)"
    )

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
    print("Generating binary comparison data...")
    print("-" * 80)

    # Generate data
    # Note: generate_binary_comparison_data generates a single dataset
    # If you need train/test split, generate all data and split manually
    n_users_total = 55000  # Total users (train + test)
    X, T, YF, YC, tauF, tauC, region_type, functional_form = (
        generate_binary_comparison_data(
            n_users=n_users_total,
            k=2,
            n_categories=[10, 10],
            m_firm=10,
            m_user=10,
            similarity=0.5,
            intensity=1,
            effect_noise_std=0,
            firm_outcome_noise_std=10,
            user_outcome_noise_std=10,
            positive_ratio=0.75,
            random_seed=11,
        )
    )

    print("Data generation complete!")
    print(f"Generated {len(X)} observations with {X.shape[1]} binary features")
    print("-" * 80)

    # Print functional form
    print_functional_form(functional_form)
    print()

    # Print descriptive statistics
    print_descriptive_statistics(X, T, YF, YC, tauF, tauC, region_type, functional_form)

    # Create plots
    print("\nGenerating plots...")
    print("-" * 80)

    # Plot outcome distributions
    plot_outcome_distributions(YF, YC, T, save_path="data/outcome_distributions.png")

    # Plot treatment effects
    plot_treatment_effects(
        tauF, tauC, region_type, save_path="data/treatment_effects.png"
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    import os

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    main()
