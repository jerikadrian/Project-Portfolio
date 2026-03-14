import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display, Markdown

def display_title(s, pref="Figure", num=1, center=False):
    ctag = "center" if center else "p"
    s = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    s = f"{s}<br><br>"
    display(Markdown(s))

def _as_float(a):
    return np.asarray(a, dtype=float)

def _mask_valid(*arrays):
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        a = _as_float(a)
        m &= ~np.isnan(a)
    return m

def _jitter(x, scale=0.06, seed=0):
    rng = np.random.default_rng(seed)
    x = _as_float(x)
    return x + rng.normal(0, scale, size=len(x))

def _welch_ttest(y0, y1):
    y0 = _as_float(y0); y1 = _as_float(y1)
    y0 = y0[~np.isnan(y0)]
    y1 = y1[~np.isnan(y1)]
    t, p = stats.ttest_ind(y0, y1, equal_var=False)
    return t, p

def fig_height_glazing_regression_panelA(show=True):
    y = df["heating_load"].to_numpy(dtype=float)
    h = df["overall_height"].to_numpy(dtype=float)
    g = df["glazing_area"].to_numpy(dtype=float)

    m = _mask_valid(y, h, g)
    y = y[m]; h = h[m]; g = g[m]

    heights = np.array(sorted(np.unique(h)))
    if len(heights) != 2:
        raise ValueError(f"Expected exactly 2 unique heights, found: {heights}")

    h_low, h_high = heights[0], heights[1]
    idx_low = (h == h_low)
    idx_high = (h == h_high)

    lr_low = stats.linregress(g[idx_low], y[idx_low])
    lr_high = stats.linregress(g[idx_high], y[idx_high])

    slope_low, se_low = lr_low.slope, lr_low.stderr
    slope_high, se_high = lr_high.slope, lr_high.stderr

    p_line_low = lr_low.pvalue
    p_line_high = lr_high.pvalue

    Z = (slope_low - slope_high) / np.sqrt(se_low**2 + se_high**2)
    p_slope_diff = 2 * (1 - stats.norm.cdf(abs(Z)))  # two-sided

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    ax.scatter(g[idx_low], y[idx_low], alpha=0.55, label=f"h = {h_low:g}")
    ax.scatter(g[idx_high], y[idx_high], alpha=0.55, label=f"h = {h_high:g}")

    x_line = np.linspace(np.min(g), np.max(g), 200)
    ax.plot(x_line, lr_low.intercept + lr_low.slope * x_line, color="k", lw=2)
    ax.plot(x_line, lr_high.intercept + lr_high.slope * x_line, color="k", lw=2)

    ax.set_xlabel("Glazing Area")
    ax.set_ylabel("Heating Load")
    ax.legend()

    ax.text(
        0.02, 0.98,
        "Separate simple linear regressions by height\n"
        f"p (slope difference) = {p_slope_diff:.2e}\n"
        f"slope(h={h_low:g}) = {slope_low:.3f}  | p (h=3.5) = {p_line_low:.2e}\n"
        f"slope(h={h_high:g}) = {slope_high:.3f} | p (h=7.0) = {p_line_high:.2e}",
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(facecolor="0.90", alpha=0.85)
    )

    if show:
        plt.show()
    return fig

def fig_gld_zero_vs_nonzero_ttest(show=True, seed=0):
    y = df["heating_load"].to_numpy(dtype=float)
    gld = df["glazing_area_distribution"].to_numpy(dtype=float)

    m = _mask_valid(y, gld)
    y = y[m]; gld = gld[m]

    y0 = y[gld == 0]
    y1 = y[gld != 0]

    t_stat, p_val = _welch_ttest(y0, y1)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    groups = [y0[~np.isnan(y0)], y1[~np.isnan(y1)]]
    positions = [1, 2]

    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False
    )

    for box in bp["boxes"]:
        box.set_alpha(0.25)
        box.set_edgecolor("k")
    for key in ["whiskers", "caps", "medians"]:
        for line in bp[key]:
            line.set_color("k")
            if key == "medians":
                line.set_linewidth(2)

    for i, vals in enumerate(groups, start=1):
        xj = _jitter(np.ones(len(vals)) * positions[i - 1], scale=0.06, seed=seed + i)
        ax.scatter(xj, vals, alpha=0.22)

    ax.set_xlim(0.5, 2.5)
    ax.margins(x=0.02)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Glazing Area Distribution = 0", "Glazing Area Distribution ≠ 0"])
    ax.set_ylabel("Heating Load")

    ax.text(
        0.02, 0.98,
        f"Two-sample t-test\np = {p_val:.2e}",
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(facecolor="0.90", alpha=0.85)
    )

    if show:
        plt.show()
    return fig
    
def plot_hypothesis1(num=2):
    display_title(
        "Heating load vs glazing area by height (3.5 and 7.0)",
        pref="Figure", num=num, center=False
    )
    fig_height_glazing_regression_panelA(show=True)

def plot_hypothesis2(num=3):
    display_title(
        "Glazing area distribution (0 vs non-0) and heating load",
        pref="Figure", num=num, center=False
    )
    fig_gld_zero_vs_nonzero_ttest(show=True)
