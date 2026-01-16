import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Tuple
import pandas as pd


def make_group_table_from_arrays(
    T: np.ndarray,
    Y_firm: np.ndarray,
    Y_user: np.ndarray,
    Usefulness: np.ndarray,
    Novelty: np.ndarray,
    SunkCost: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Build a table grouped by High/Low (threshold) for Usefulness, Novelty, Sunk Cost and by Trial (Short/Long).
    - Y_firm is assumed binary (subscription).
    - Y_user is observed only if subscribed; NaNs are OK and are ignored in the mean.
    Returns a tidy pivoted DataFrame with columns:
       ('Avg_Firm_Outcome', 'Short'), ('Avg_Firm_Outcome', 'Long'),
       ('Avg_User_Outcome', 'Short'), ('Avg_User_Outcome', 'Long')
    """
    df = pd.DataFrame(
        {
            "Trial": np.where(np.asarray(T).astype(int) == 0, "Short", "Long"),
            "Y_firm": Y_firm.astype(float),
            "Y_user": Y_user,  # NaNs for non-subscribers are fine
            "Usefulness High": (Usefulness > threshold).astype(int),
            "Novelty High": (Novelty > threshold).astype(int),
            "Sunk Cost High": (SunkCost > threshold).astype(int),
        }
    )

    agg = (
        df.groupby(["Usefulness High", "Novelty High", "Sunk Cost High", "Trial"])
        .agg(Avg_Firm_Outcome=("Y_firm", "mean"), Avg_User_Outcome=("Y_user", "mean"))
        .reset_index()
    )

    table = agg.pivot_table(
        index=["Usefulness High", "Novelty High", "Sunk Cost High"],
        columns="Trial",
        values=["Avg_Firm_Outcome", "Avg_User_Outcome"],
    )

    # order columns consistently
    table = table.reindex(
        columns=pd.MultiIndex.from_product(
            [["Avg_Firm_Outcome", "Avg_User_Outcome"], ["Short", "Long"]]
        ),
        fill_value=np.nan,
    ).sort_index()

    return table


def _quantile_binned_xy(
    x: np.ndarray, y: np.ndarray, nbins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(x, qs)
    # avoid edge collisions
    eps = 1e-9
    edges[0] -= eps
    edges[-1] += eps
    bin_id = np.digitize(x, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.array(
        [
            np.nanmean(y[bin_id == i]) if np.any(bin_id == i) else np.nan
            for i in range(nbins)
        ]
    )
    return centers, means


def _plot_outcomes_vs_latent_on_ax(
    ax: plt.Axes,
    latent: np.ndarray,
    latent_label: str,
    T: np.ndarray,
    Y_firm: np.ndarray,
    Y_user: np.ndarray,
    nbins: int = 20,
) -> None:
    """
    Plot firm and user outcomes vs latent variable, separated by treatment.

    Args:
        ax: Matplotlib axis
        latent: Latent variable values
        latent_label: Label for the latent variable
        T: Treatment assignment (0=Short, 1=Long)
        Y_firm: Firm outcome (subscription)
        Y_user: User outcome (satisfaction)
        nbins: Number of bins for averaging
    """
    T = np.asarray(T).astype(int)
    mask0, mask1 = (T == 0), (T == 1)

    # Firm outcome (subscription rate)
    x0, yf0 = _quantile_binned_xy(latent[mask0], Y_firm[mask0], nbins)
    x1, yf1 = _quantile_binned_xy(latent[mask1], Y_firm[mask1], nbins)

    # User outcome (satisfaction - only for subscribers)
    x0u, yu0 = _quantile_binned_xy(latent[mask0], Y_user[mask0], nbins)
    x1u, yu1 = _quantile_binned_xy(latent[mask1], Y_user[mask1], nbins)

    ax.plot(x0, yf0, marker="o", linewidth=1, label="Subscription rate (Short)")
    ax.plot(x1, yf1, marker="o", linewidth=1, label="Subscription rate (Long)")
    ax.plot(x0u, yu0, marker="s", linewidth=1, label="Satisfaction (Short)")
    ax.plot(x1u, yu1, marker="s", linewidth=1, label="Satisfaction (Long)")

    ax.set_title(f"Outcomes vs {latent_label}")
    ax.set_xlabel(latent_label)
    ax.set_ylabel("Average outcome")
    ax.grid(True)
    ax.legend()


def _plot_pi_vs_latent_on_ax(
    ax: plt.Axes,
    latent: np.ndarray,
    latent_label: str,
    T: np.ndarray,
    pi: np.ndarray,
    nbins: int = 20,
) -> None:
    T = np.asarray(T).astype(int)
    mask0, mask1 = (T == 0), (T == 1)

    x0, p0 = _quantile_binned_xy(latent[mask0], pi[mask0], nbins)
    x1, p1 = _quantile_binned_xy(latent[mask1], pi[mask1], nbins)

    ax.plot(x0, p0, marker="o", linewidth=1, label="Subscription prob. (Short)")
    ax.plot(x1, p1, marker="o", linewidth=1, label="Subscription prob. (Long)")

    ax.set_title(f"Average subscription probability vs {latent_label}")
    ax.set_xlabel(latent_label)
    ax.set_ylabel("Average probability")
    ax.grid(True)
    ax.legend()


def plot_six_panels_outcomes_and_pi(
    T: np.ndarray,
    Y_firm: np.ndarray,
    Y_user: np.ndarray,
    pi: np.ndarray,
    L: Dict[str, np.ndarray],
    nbins: int = 20,
    figsize: Tuple[int, int] = (18, 10),
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create one figure with 6 subplots showing relationships between latent variables and outcomes:
      Row 1: Subscription probability vs Usefulness, Novelty, Sunk Cost
      Row 2: Firm outcomes (subscription rate) vs Usefulness, Novelty, Sunk Cost
      Row 3: User outcomes (satisfaction) vs Usefulness, Novelty, Sunk Cost

    Args:
        T: array-like, 0=Short, 1=Long
        Y_firm: array-like, firm outcome (subscription, 0/1)
        Y_user: array-like, user outcome (satisfaction; NaN for non-subscribers ok)
        pi: array-like, subscription probability
        L: dict with keys "U", "N", "S" (latent arrays in [0,1])
        nbins: number of quantile bins
        figsize: figure size
        show: whether to call plt.show()

    Returns:
        (fig, axes) where axes is a 3x3 ndarray of Axes.
    """
    U = np.asarray(L["U"])
    N = np.asarray(L["N"])
    S = np.asarray(L["S"])

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Row 1: Subscription probability vs latent variables
    _plot_pi_vs_latent_on_ax(axes[0, 0], U, "Usefulness", T, pi, nbins)
    _plot_pi_vs_latent_on_ax(axes[0, 1], N, "Novelty", T, pi, nbins)
    _plot_pi_vs_latent_on_ax(axes[0, 2], S, "Sunk Cost", T, pi, nbins)

    # Row 2: Firm outcomes (subscription rate) vs latent variables
    _plot_pi_vs_latent_on_ax(axes[1, 0], U, "Usefulness", T, Y_user, nbins)
    _plot_pi_vs_latent_on_ax(axes[1, 1], N, "Novelty", T, Y_user, nbins)
    _plot_pi_vs_latent_on_ax(axes[1, 2], S, "Sunk Cost", T, Y_user, nbins)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


def _binned_line(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    nbins: int = 20,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """
    Plot binned average of y vs x on the given axis using quantile bins.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(x, qs)
    # ensure strictly increasing to avoid edge collisions
    eps = 1e-9
    edges[0] -= eps
    edges[-1] += eps

    bin_id = np.digitize(x, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])

    means = np.empty(nbins, dtype=float)
    means.fill(np.nan)
    for i in range(nbins):
        mask = bin_id == i
        if np.any(mask):
            means[i] = np.nanmean(y[mask])

    ax.plot(centers, means, marker="o", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


# ---------- Figure A: previous 2×3 subplot ----------
# (Top: U/N/S vs their gated X features; Bottom: histograms of U/N/S)


def plot_latent_overview(
    X: np.ndarray,
    L: Dict[str, np.ndarray],
    gate_indices: Dict[str, Union[int, List[int]]],
    nbins: int = 20,
    figsize: tuple = (16, 8),
    show: bool = True,
    savepath: Optional[str] = None,
) -> plt.Figure:
    """
    Create a 2×3 figure:
      Row 1: Binned averages of each latent vs its gated X feature
      Row 2: Histograms of the three latents
    Args:
        X: (n, p) feature matrix
        L: dict with keys 'U','N','S' mapping to latent arrays in [0,1]
        gate_indices: e.g., {'U': 0, 'N': 1, 'S': 2}
        nbins: number of quantile bins for the binned curves
        figsize: figure size
        show: whether to plt.show()
        savepath: optional file path to save the figure
    Returns:
        The matplotlib Figure object.
    """
    U = np.asarray(L["U"])
    N = np.asarray(L["N"])
    S = np.asarray(L["S"])

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Top row: binned latent vs gate feature
    _binned_line(
        axes[0, 0],
        X[:, gate_indices["U"][0]],  # Use first (and only) index
        U,
        nbins,
        title="Usefulness vs gated X",
        xlabel="X[:, {}] (binned)".format(gate_indices["U"][0]),
        ylabel="avg U",
    )
    _binned_line(
        axes[0, 1],
        X[:, gate_indices["N"][0]],  # Use first (and only) index
        N,
        nbins,
        title="Novelty vs gated X",
        xlabel="X[:, {}] (binned)".format(gate_indices["N"][0]),
        ylabel="avg N",
    )
    _binned_line(
        axes[0, 2],
        X[:, gate_indices["S"][0]],  # Use first (and only) index
        S,
        nbins,
        title="Sunk Cost vs gated X",
        xlabel="X[:, {}] (binned)".format(gate_indices["S"][0]),
        ylabel="avg S",
    )

    # Bottom row: histograms
    axes[1, 0].hist(U, bins=30)
    axes[1, 0].set_title("Histogram of Usefulness")
    axes[1, 0].set_xlabel("U")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True)

    axes[1, 1].hist(N, bins=30)
    axes[1, 1].set_title("Histogram of Novelty")
    axes[1, 1].set_xlabel("N")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True)

    axes[1, 2].hist(S, bins=30)
    axes[1, 2].set_title("Histogram of Sunk Cost")
    axes[1, 2].set_xlabel("S")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].grid(True)

    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig


# ---------- Figure B: new 2×3 subplot ----------
# (Short: P_short vs N, P_short vs U, pi vs P_short; Long: P_long vs U, P_long vs S, pi vs P_long)


def plot_P_and_pi_overview(
    L: Dict[str, np.ndarray],
    T: np.ndarray,
    P_short: np.ndarray,
    P_long: np.ndarray,
    pi: np.ndarray,
    nbins: int = 20,
    figsize: tuple = (18, 9),
    show: bool = True,
    savepath: Optional[str] = None,
) -> plt.Figure:
    """
    Create a 2×3 figure for relationships among P variables and pi:
      Row 1 (Short / T=0): P_short vs N, P_short vs U, pi vs P_short
      Row 2 (Long  / T=1): P_long  vs U, P_long  vs S, pi vs P_long
    Args:
        L: dict with 'U','N','S' arrays
        T: treatment (0 short, 1 long)
        P_short, P_long: perceived usefulness scores
        pi: subscription probability
        nbins, figsize, show, savepath: plotting options
    """
    U = np.asarray(L["U"])
    N = np.asarray(L["N"])
    S = np.asarray(L["S"])
    T = np.asarray(T).astype(int)
    P_short = np.asarray(P_short)
    P_long = np.asarray(P_long)
    pi = np.asarray(pi)

    mask_short = T == 0
    mask_long = T == 1

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Row 1: Short
    _binned_line(
        axes[0, 0],
        N[mask_short],
        P_short[mask_short],
        nbins,
        title="Short (T=0): P_short vs Novelty",
        xlabel="N",
        ylabel="avg P_short",
    )
    _binned_line(
        axes[0, 1],
        U[mask_short],
        P_short[mask_short],
        nbins,
        title="Short (T=0): P_short vs Usefulness",
        xlabel="U",
        ylabel="avg P_short",
    )
    _binned_line(
        axes[0, 2],
        P_short[mask_short],
        pi[mask_short],
        nbins,
        title="Short (T=0): Subscription probability vs P_short",
        xlabel="P_short",
        ylabel="avg pi",
    )

    # Row 2: Long
    _binned_line(
        axes[1, 0],
        U[mask_long],
        P_long[mask_long],
        nbins,
        title="Long (T=1): P_long vs Usefulness",
        xlabel="U",
        ylabel="avg P_long",
    )
    _binned_line(
        axes[1, 1],
        S[mask_long],
        P_long[mask_long],
        nbins,
        title="Long (T=1): P_long vs Sunk cost",
        xlabel="S",
        ylabel="avg P_long",
    )
    _binned_line(
        axes[1, 2],
        P_long[mask_long],
        pi[mask_long],
        nbins,
        title="Long (T=1): Subscription probability vs P_long",
        xlabel="P_long",
        ylabel="avg pi",
    )

    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig


# ---------- (Optional) quick text summary ----------


def print_summary(YF: np.ndarray, T: np.ndarray, L: Dict[str, np.ndarray]) -> None:
    """
    Print simple summary stats: overall and by treatment subscription rates,
    and means/stds of U, N, S.
    """
    YF = np.asarray(YF)
    T = np.asarray(T).astype(int)
    U = np.asarray(L["U"])
    N = np.asarray(L["N"])
    S = np.asarray(L["S"])

    avg_conv = float(np.nanmean(YF))
    avg_conv_short = float(np.nanmean(YF[T == 0]))
    avg_conv_long = float(np.nanmean(YF[T == 1]))

    print(f"Average subscription rate (overall): {avg_conv:.3f}")
    print(f"Average subscription rate (Short, T=0): {avg_conv_short:.3f}")
    print(f"Average subscription rate (Long , T=1): {avg_conv_long:.3f}")
    print(
        f"Mean U={U.mean():.3f} (sd {U.std():.3f}), "
        f"N={N.mean():.3f} (sd {N.std():.3f}), "
        f"S={S.mean():.3f} (sd {S.std():.3f})"
    )
