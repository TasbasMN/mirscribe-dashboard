import pandas as pd
import matplotlib as mpl


df = pd.read_csv("data/tripletss.csv")
full = pd.read_csv("data/real.csv")
full['is_cancer_promoting'] = ((full['gene_is_onc'] & (full['stats_log2fc'] > 0)) |
                                       (full['gene_is_tsupp'] & (full['stats_log2fc'] < 0)))





def plot_signature_gene_heatmap(df, top_n_genes=30, top_n_sigs=20, all_signatures=None):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap, PowerNorm

    # --- preprocess ---
    data = (
        df.groupby(['gene_name', 'mutation_signature'])
        .agg({
            'total_hits': 'sum',
            'stats_log2fc_mean': 'mean',
            'gene_type': 'first',
            'mirna_family': pd.Series.nunique
        })
        .reset_index()
    )

    # use full list of signatures if provided
    if all_signatures is None:
        all_sigs = sorted(df["mutation_signature"].unique())
    else:
        all_sigs = all_signatures

    # build full matrix and include all signatures
    full_matrix = data.pivot(index='gene_name', columns='mutation_signature', values='total_hits').fillna(0)
    full_matrix = full_matrix.reindex(columns=all_sigs, fill_value=0)

    # select top genes by total hits
    top_genes = full_matrix.sum(axis=1).nlargest(top_n_genes).index
    heatmap_df = full_matrix.loc[top_genes]

    # get gene role
    gene_type_map = (
        data.drop_duplicates('gene_name')
        .set_index('gene_name')['gene_type']
        .reindex(heatmap_df.index)
        .fillna('Unannotated')
    )

    role_priority = {
        'Oncogene': 0,
        'DualRole': 1,
        'TumorSuppressor': 2,
        'Unannotated': 3
    }
    gene_sort_key = gene_type_map.map(role_priority)
    sorted_genes = gene_sort_key.sort_values().index
    heatmap_df = heatmap_df.loc[sorted_genes]

    gene_fc = (
        data[data['gene_name'].isin(sorted_genes)]
        .groupby('gene_name')['stats_log2fc_mean']
        .mean()
        .reindex(sorted_genes)
    )

    gene_type = gene_type_map.loc[sorted_genes]

    role_colors = {
        'Oncogene': '#d95f02',
        'TumorSuppressor': '#1b9e77',
        'DualRole': '#7570b3',
        'Unannotated': '#999999'
    }
    gene_colors = gene_type.map(role_colors).fillna('#999999')
    fc_colors = ['green' if v > 0 else 'red' for v in gene_fc.values]

    # --- highlight logic violations ---
    highlight_mask = (
        ((gene_type == "Oncogene") & (gene_fc > 0)) |
        ((gene_type == "TumorSuppressor") & (gene_fc < 0))
    )

    highlight_genes = set(gene_fc[highlight_mask].index)

    # --- compute signature net effect ---
    sig_data = data[data['mutation_signature'].isin(heatmap_df.columns)]
    sig_data['direction'] = np.where(sig_data['stats_log2fc_mean'] > 0, 'up', 'down')

    sig_counts = (
        sig_data
        .groupby(['mutation_signature', 'direction'])
        .size()
        .unstack(fill_value=0)
        .reindex(heatmap_df.columns, fill_value=0)
    )

    sig_counts['net'] = sig_counts.get('up', 0) - sig_counts.get('down', 0)
    sig_counts['color'] = ['green' if x > 0 else 'red' for x in sig_counts['net']]
    sig_net = sig_counts['net']

    # --- layout ---
    fig = plt.figure(figsize=(17, 10))
    gs = GridSpec(
        2, 4,
        width_ratios=[1.0, 6, 0.15, 0.6],
        height_ratios=[0.5, 5],
        hspace=0.15,
        wspace=0.4
    )

    # --- top-left: legend ---
    ax_legend = fig.add_subplot(gs[0, 0])
    ax_legend.axis('off')
    legend_patches = [Patch(color=color, label=role) for role, color in role_colors.items()]
    ax_legend.legend(
        handles=legend_patches,
        title='Gene Role',
        loc='upper left',
        bbox_to_anchor=(-0.1, 1.0),
        frameon=False,
        handlelength=1.0,
        handleheight=0.8,
        borderpad=0.3,
        labelspacing=0.4,
        title_fontsize='small',
        fontsize='small'
    )

    # --- top: signature net effect ---
    ax_top = fig.add_subplot(gs[0, 1])
    ymax = np.ceil(np.abs(sig_net).max() * 1.1)
    yticks = np.linspace(-ymax, ymax, 5)
    ax_top.bar(
        x=np.arange(len(sig_net)),
        height=sig_net.values,
        color=sig_counts['color'],
        edgecolor='black',
        linewidth=0.3
    )
    ax_top.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_top.set_ylim(-ymax, ymax)
    ax_top.set_yticks(yticks)
    ax_top.set_yticklabels([f"{y:.0f}" for y in yticks])
    ax_top.set_xticks([])
    ax_top.set_xlim(-0.5, len(sig_net) - 0.5)
    ax_top.set_ylabel('Net Hits\n(+Up / -Down)')

    # --- left: gene log2FC ---
    ax_left = fig.add_subplot(gs[1, 0])
    xmax = np.ceil(np.abs(gene_fc).max() * 1.1)
    xticks = np.linspace(-xmax, xmax, 5)
    ax_left.barh(
        y=np.arange(len(gene_fc)),
        width=gene_fc.values,
        color=fc_colors,
        edgecolor='black',
        linewidth=0.3
    )
    ax_left.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax_left.set_xlim(-xmax, xmax)
    ax_left.set_xticks(xticks)
    ax_left.set_xticklabels([f"{x:.1f}" for x in xticks])
    ax_left.set_yticks([])
    ax_left.set_ylim(-0.5, len(gene_fc) - 0.5)
    ax_left.set_xlabel('log2FC')
    ax_left.invert_yaxis()

    # --- center: heatmap with perceptual scaling and cell labels ---
    ax_heat = fig.add_subplot(gs[1, 1])
    greens = plt.cm.Greens(np.linspace(0.05, 1, 256))
    custom_colors = np.vstack([[1, 1, 1, 1], greens])
    custom_cmap = ListedColormap(custom_colors)
    norm = PowerNorm(gamma=0.4, vmin=0, vmax=heatmap_df.values.max())

    annot_strings = heatmap_df.copy()
    annot_strings = annot_strings.applymap(lambda x: f"{int(x)}" if x != 0 else "")

    heatmap = sns.heatmap(
        heatmap_df,
        cmap=custom_cmap,
        norm=norm,
        annot=annot_strings,
        fmt='',
        annot_kws={"size": 7},
        ax=ax_heat,
        cbar=False,
        linewidths=0.5,
        linecolor='#eeeeee'
    )

    ax_heat.set_xlabel('Mutation Signature')
    ax_heat.set_ylabel('')
    ax_heat.set_title('Mutation Density Across Genes and Signatures')
    plt.sca(ax_heat)
    plt.xticks(rotation=60, ha='right')
    plt.yticks(rotation=0)

    for ticklabel, gene in zip(ax_heat.get_yticklabels(), sorted_genes):
        ticklabel.set_color(gene_colors[gene])
        if gene in highlight_genes:
            ticklabel.set_backgroundcolor("#fde0dd")
            ticklabel.set_weight("bold")

    # --- colorbar ---
    ax_cbar = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(
        heatmap.collections[0],
        cax=ax_cbar,
        orientation='vertical',
        shrink=0.85
    )
    cbar.set_label('Total Hits')

    plt.tight_layout()
    # plt.savefig("signature_gene_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()


all_sigs = [
    "SBS5", "SBS13", "SBS3", "SBS8", "SBS2", "SBS1", "SBS39",
    "SBS44", "SBS18", "SBS26", "SBS10d", "SBS21"
]

plot_signature_gene_heatmap(df, top_n_genes=40, all_signatures=all_sigs)
