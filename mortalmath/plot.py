import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from .config import MODEL_COLORS, SYSTEM_PROMPTS

METRICS = [
    ('completion_tokens', 'Completion Tokens', True),
    ('prioritized_safety', 'Refusal Rate (no \\boxed{})', False),
    ('keyword_ratio', 'Safety/Math Keyword Ratio', False),
    ('is_correct', 'MATH Answer Correctness', False),
]


def plot_results(df, x_axis='urgency_level', out_path=None):
    df = df.copy()
    xticklabels = None

    if x_axis == 'system_prompt':
        mapping = {p: i for i, p in enumerate(SYSTEM_PROMPTS)}
        df[x_axis] = df[x_axis].map(mapping)
        xticklabels = [f"#{i}" for i in range(len(SYSTEM_PROMPTS))]

    models = sorted(df['model'].unique())
    default_palette = sns.color_palette("tab10", len(models))
    palette = {m: MODEL_COLORS.get(m, default_palette[i]) for i, m in enumerate(models)}

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    handles, labels = None, None

    for i, (ax, (col, col_name, is_log)) in enumerate(zip(axes.flatten(), METRICS)):
        sns.lineplot(data=df, x=x_axis, y=col, hue='model', style='model',
                     markers=True, dashes=False, errorbar=('ci', 80), ax=ax, palette=palette)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend():
            ax.get_legend().remove()

        if col == 'completion_tokens':
            max_x = df[x_axis].max()
            last = df[df[x_axis] == max_x].groupby('model')[['completion_tokens', 'latency']].mean()
            for model in models:
                if model in last.index:
                    ax.text(x=max_x, y=last.loc[model, 'completion_tokens'],
                            s=f"  {last.loc[model, 'latency']:.1f}s", color=palette[model],
                            va='center', ha='left', fontweight='bold', fontsize=9)

        ax.set_xlabel(x_axis.replace('_', ' ').title() if i > 1 else "", fontweight="bold")
        ax.set_ylabel(col_name, fontweight='bold', fontsize=11)

        if xticklabels and i > 1:
            ax.set_xticks(range(len(xticklabels)))
            ax.set_xticklabels(xticklabels)
        elif x_axis == 'urgency_level':
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        if is_log:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1.0, 2.0, 5.0], numticks=10))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.grid(True, which="minor", alpha=0.3)
        elif col == 'keyword_ratio':
            ax.set_ylim(bottom=-0.05)
        else:
            ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=min(len(labels), 4), frameon=True, edgecolor="grey", fontsize=10)
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
