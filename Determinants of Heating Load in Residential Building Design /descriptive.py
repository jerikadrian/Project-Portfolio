from IPython.display import display, Markdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def display_title(s, pref="Figure", num=1, center=False):
    ctag = "center" if center else "p"
    s = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref == "Figure":
        s = f"{s}<br><br>"
    else:
        s = f"<br><br>{s}"
    display(Markdown(s))
    
def _dispersion(series: pd.Series):
    x = pd.Series(series).dropna()
    stdev = x.std(ddof=0)  
    xmin = x.min()
    xmax = x.max()
    rng = xmax - xmin
    q25 = x.quantile(0.25)
    q75 = x.quantile(0.75)
    iqr = q75 - q25
    return stdev, xmin, xmax, rng, q25, q75, iqr

def display_dispersion_table(num=1):
    display_title("Dispersion summary statistics", pref="Table", num=num, center=False)

    out = df.apply(_dispersion, axis=0)
    labels = ["st.dev.", "min", "max", "range", "25th", "75th", "IQR"]

    if isinstance(out, pd.Series):
        tab = pd.DataFrame(out.values.tolist(), index=df.columns, columns=labels).T
    else:
        tab = out.copy()
        tab.index = labels

    round_dict = {
        "relative_compactness": 4, "surface_area": 1, "wall_area": 1, "roof_area": 1,
        "overall_height": 2, "orientation": 1, "glazing_area": 2,
        "glazing_area_distribution": 1, "heating_load": 2
    }
    display(tab.round(round_dict))

COLORS = {
    "relative_compactness": "b",
    "surface_area": "r",
    "wall_area": "g",
    "roof_area": "orange",
    "overall_height": "y",
    "orientation": "magenta",
    "glazing_area": "purple",
    "glazing_area_distribution": "teal",
}

def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y)
    return np.corrcoef(x[m], y[m])[0, 1]

def plot_regression_line(ax, x, y, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y)
    x = x[m]
    y = y[m]
    a, b = np.polyfit(x, y, 1)
    x0, x1 = np.min(x), np.max(x)
    ax.plot([x0, x1], [a * x0 + b, a * x1 + b], **kwargs)

def _jitter(vals, scale=0.06, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    return vals + rng.normal(0, scale, size=len(vals))

def plot_descriptive(num=1, jitter_scale=0.06, point_alpha=0.35, a_hspace=0.12):
    
    display_title("Descriptive plots for building design features", pref="Figure", num=num, center=False)

    y = df["heating_load"].to_numpy()

    cont_specs = [
        ("Relative Compactness", "relative_compactness"),
        ("Surface Area", "surface_area"),
        ("Wall Area", "wall_area"),
        ("Roof Area", "roof_area"),
        ("Overall Height", "overall_height"),
        ("Glazing Area", "glazing_area"),
    ]

    cat_specs = [
        ("Orientation", "orientation"),
        ("Glazing Area Distribution", "glazing_area_distribution"),
    ]

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.25)

    gs_A = outer[0].subgridspec(2, 3, hspace=0.05, wspace=0.05)
    axs_A = np.array([[fig.add_subplot(gs_A[r, c]) for c in range(3)] for r in range(2)])

    gs_B = outer[1].subgridspec(1, 2, wspace=0.05)
    axs_B = [fig.add_subplot(gs_B[0, 0]), fig.add_subplot(gs_B[0, 1])]

    for i, (label, colname) in enumerate(cont_specs):
        r = i // 3
        c = i % 3
        ax = axs_A[r, c]

        x = df[colname].to_numpy()
        color = COLORS[colname]

        ax.scatter(x, y, alpha=point_alpha, color=color)
        plot_regression_line(ax, x, y, color="k", lw=2)

        rr = pearson_r(x, y)
        ax.text(
            0.05, 0.90, f"Pearson r = {rr:.3f}",
            transform=ax.transAxes,
            bbox=dict(color="0.85", alpha=0.7)
        )
        ax.set_xlabel(label)

    axs_A[0, 0].set_ylabel("Heating Load")
    axs_A[1, 0].set_ylabel("Heating Load")

    for rr in range(2):
        for cc in range(1, 3):
            axs_A[rr, cc].set_yticklabels([])

    axs_A[0, 0].set_title("(A) Continuous building design features", loc="left")

    for ax, (label, colname), seed in zip(axs_B, cat_specs, [101, 202]):
        g = df[colname].to_numpy(dtype=float)

        m = ~np.isnan(g) & ~np.isnan(y)
        g0 = g[m]
        y0 = y[m]

        cats = np.sort(np.unique(g0))
        grouped = [y0[g0 == c] for c in cats]

        tick_labels = [str(int(c)) if float(c).is_integer() else str(c) for c in cats]

        bp = ax.boxplot(
            grouped,
            tick_labels=tick_labels,
            patch_artist=True
        )

        box_color = COLORS[colname]
        for box in bp["boxes"]:
            box.set_facecolor(box_color)
            box.set_alpha(0.20)
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color("k")

        for i_cat, vals in enumerate(grouped, start=1):
            xj = _jitter(np.full(len(vals), i_cat), scale=jitter_scale, seed=seed)
            ax.scatter(xj, vals, alpha=0.25, color=box_color)

        means = [np.mean(vals) for vals in grouped]
        ax.plot(range(1, len(means) + 1), means, marker="o", lw=2, color="k")

        ax.set_xlabel(f"{label}")
        axs_B[0].set_ylabel("Heating Load")
        axs_B[1].set_ylabel("")

    axs_B[0].set_title("(B) Categorical building design features", loc="left")

    plt.show()
