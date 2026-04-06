# src/divtree/viz.py
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D


def plot_divergence_tree(
    tree,
    max_depth=None,
    round_digits=3,
    figsize=(12, 7),
    h_spacing=2.8,  # horizontal spacing between sibling subtrees
    v_spacing=2.0,  # vertical spacing between depths
    node_width=2.4,  # node box width
    node_height=1.0,  # node box height
    font_size=13,
    edge_color="#808080",
    internal_fc="#FFFFFF",
    internal_ec="#333333",
    leaf_fc="#F7F7F7",
    leaf_ec="#333333",
    diverge_posneg="#D9534F",  # tauF>0, tauC<0 (firm wins, consumer loses)
    diverge_negpos="#5BC0DE",  # tauF<0, tauC>0 (firm loses, consumer wins)
    align_pospos="#5CB85C",  # both positive
    align_negneg="#A0A0A0",  # both negative
    neutral="#DDDDDD",  # near zero / undefined
    edge_label_fontsize=11,
    show_legend=True,
    ax=None,
):
    """
    Pretty, spaced layout for DivergenceTree using matplotlib only.
    Colors leaves by alignment/divergence of (tauF, tauC).
    """

    root = getattr(tree, "root_", None)
    if root is None:
        raise ValueError("Tree is not fitted (tree.root_ is None).")

    # -------- helpers --------

    def _is_leaf(n):
        return (n.left is None) and (n.right is None)

    def _safe(x):
        return (
            (x is not None)
            and (isinstance(x, (int, float)))
            and math.isfinite(float(x))
        )

    def _sign(x, ztol=1e-9):
        if not _safe(x) or abs(float(x)) < ztol:
            return 0
        return 1 if x > 0 else -1

    def _leaf_color(tF, tC):
        sF, sC = _sign(tF), _sign(tC)
        if sF == 0 or sC == 0:
            return neutral
        if sF > 0 and sC > 0:
            return align_pospos
        if sF < 0 and sC < 0:
            return align_negneg
        if sF > 0 and sC < 0:
            return diverge_posneg
        if sF < 0 and sC > 0:
            return diverge_negpos
        return neutral

    def _r(x):
        if not _safe(x):
            return "nan"
        return f"{round(float(x), round_digits)}"

    # subtree leaf count for balanced x-positioning
    def _count_leaves(n, depth=0):
        if n is None:
            return 0
        if max_depth is not None and depth > max_depth:
            return 0
        if _is_leaf(n) or (max_depth is not None and depth == max_depth):
            return 1
        return _count_leaves(n.left, depth + 1) + _count_leaves(n.right, depth + 1)

    # Compute positions: x centered by subtree sizes, y by depth (depth 0 at top)
    pos = {}  # id(node) -> (x, y)

    def _layout(n, depth=0, x_left=0.0):
        if n is None:
            return 0.0
        if max_depth is not None and depth > max_depth:
            return 0.0

        # effective leaf counts (respecting max_depth)
        left_leaves = _count_leaves(n.left, depth + 1)
        right_leaves = _count_leaves(n.right, depth + 1)
        total_leaves = max(1, left_leaves + right_leaves)

        # center position of this subtree
        subtree_width = (total_leaves - 1) * h_spacing
        x_center = x_left + subtree_width / 2.0
        y = depth * v_spacing  # depth increases downward; we'll invert axis later
        pos[id(n)] = (x_center, y)

        # Lay out children
        x_cursor = x_left
        if n.left and (max_depth is None or depth + 1 <= max_depth):
            _layout(n.left, depth + 1, x_cursor)
            x_cursor += max(0, left_leaves - 1) * h_spacing
            if left_leaves == 0:
                x_cursor += h_spacing  # minimal gap if no leaves (depth cap)
        if n.right and (max_depth is None or depth + 1 <= max_depth):
            if left_leaves > 0:
                x_cursor += h_spacing
            _layout(n.right, depth + 1, x_cursor)

        return subtree_width

    _layout(root, 0, 0.0)

    # Collect nodes/edges to draw
    nodes = []
    edges = []

    def _walk(n):
        if n is None:
            return
        if max_depth is not None and n.depth > max_depth:
            return
        nodes.append(n)
        if n.left and (max_depth is None or n.left.depth <= max_depth):
            edges.append((n, n.left, True))
            _walk(n.left)
        if n.right and (max_depth is None or n.right.depth <= max_depth):
            edges.append((n, n.right, False))
            _walk(n.right)

    _walk(root)

    # normalize coordinates for pretty margins
    xs = [pos[id(n)][0] for n in nodes] or [0.0]
    ys = [pos[id(n)][1] for n in nodes] or [0.0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # figure/axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.axis("off")

    # draw edges
    for parent, child, is_left in edges:
        x0, y0 = pos[id(parent)]
        x1, y1 = pos[id(child)]
        ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=1.2, zorder=1)
        # edge label
        midx, midy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(
            midx,
            midy - 0.25,
            "≤" if is_left else ">",
            ha="center",
            va="center",
            fontsize=edge_label_fontsize,
            color=edge_color,
        )

    # draw nodes
    for n in nodes:
        x, y = pos[id(n)]
        is_leaf = (n.feature is None) or (n.left is None and n.right is None)

        # style
        if is_leaf:
            face = _leaf_color(n.tauF, n.tauC)
            edge = leaf_ec
        else:
            face = internal_fc
            edge = internal_ec

        rect = FancyBboxPatch(
            (x - node_width / 2, y - node_height / 2),
            node_width,
            node_height,
            boxstyle="round,pad=0.25,rounding_size=0.06",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
        ax.add_patch(rect)

        if is_leaf:
            txt = f"$\\tau_{{F}}={_r(n.tauF)}$\n$\\tau_{{C}}={_r(n.tauC)}$"
        else:
            thr = round(n.threshold, round_digits) if n.threshold is not None else "?"
            txt = (
                f"X[{n.feature}] ≤ {thr}\n"
                f"$\\tau_{{F}}={_r(n.tauF)}$\n"
                f"$\\tau_{{C}}={_r(n.tauC)}$"
            )

        ax.text(
            x,
            y,
            txt,
            ha="center",
            va="center",
            fontsize=font_size,
            color="#111111",
            zorder=3,
        )

    # bounds with ROOT AT TOP:
    ax.set_xlim(x_min - node_width, x_max + node_width)
    ax.set_ylim(y_min - node_height, y_max + node_height)  # normal order
    ax.invert_yaxis()  # flip so y=0 (root) is at the TOP

    if show_legend:
        legend_elems = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="$\\tau_F>0, \\tau_C>0$",
                markerfacecolor=align_pospos,
                markersize=12,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="$\\tau_F<0, \\tau_C<0$",
                markerfacecolor=align_negneg,
                markersize=12,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="$\\tau_F>0, \\tau_C<0$",
                markerfacecolor=diverge_posneg,
                markersize=12,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="$\\tau_F<0, \\tau_C>0$",
                markerfacecolor=diverge_negpos,
                markersize=12,
            ),
        ]
        ax.legend(handles=legend_elems, loc="upper left", frameon=False, fontsize=11)

    fig.tight_layout()
    return fig, ax
