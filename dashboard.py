import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, PowerNorm

# load data
@st.cache_data
def load_data():
    return pd.read_csv("data/triplets.csv")

df = load_data()

st.title("Triplet Search Dashboard")

# Initialize session state for filters if not already present
if 'filters' not in st.session_state:
    st.session_state.filters = {
        'mutation_signature': "All",
        'mirna_family': "All",
        'gene_type': "All",
        'is_cancer_promoting': "All",
        'triplet_behavior': "All",
        'gene_name': "All"
    }

# Initialize show_network if not present
if 'show_network' not in st.session_state:
    st.session_state.show_network = False

# Initialize show_heatmap if not present
if 'show_heatmap' not in st.session_state:
    st.session_state.show_heatmap = False

# Initialize reset flags if not present
for filter_name in list(st.session_state.filters.keys()) + ['all_filters']:
    reset_key = f"reset_{filter_name}"
    if reset_key not in st.session_state:
        st.session_state[reset_key] = False

# Handle resets before rendering widgets
if st.session_state.reset_all_filters:
    for key in st.session_state.filters:
        st.session_state.filters[key] = "All"
    st.session_state.reset_all_filters = False

for filter_name in st.session_state.filters.keys():
    reset_key = f"reset_{filter_name}"
    if st.session_state[reset_key]:
        st.session_state.filters[filter_name] = "All"
        st.session_state[reset_key] = False

# Sidebar for filters
st.sidebar.header("Filters")

# Function to extract actual value from option with count
def extract_value(option):
    if option == "All":
        return option
    return option.split(" (")[0]

# Function to add counts to dropdown options sorted by count (descending)
def add_counts_to_options(df, column_name, current_filters=None):
    # Apply current filters except for the one we're building options for
    if current_filters:
        filtered_df = df.copy()
        for filter_name, filter_value in current_filters.items():
            if filter_value != "All" and filter_name != column_name:
                if filter_name == 'is_cancer_promoting':
                    filtered_df = filtered_df[filtered_df[filter_name].astype(str) == filter_value]
                else:
                    filtered_df = filtered_df[filtered_df[filter_name] == filter_value]
    else:
        filtered_df = df
    
    if column_name == "is_cancer_promoting":
        # Special case for boolean column
        counts = filtered_df[column_name].astype(str).value_counts()
        options = ["All"] + [f"{val} ({counts[val]})" for val in counts.index.tolist()]
        return options
    
    counts = filtered_df[column_name].value_counts()
    # Sort by count descending
    options = ["All"] + [f"{val} ({counts[val]})" for val in counts.index.tolist()]
    return options

# Update filters in session state
def update_filters(filter_name, value):
    # Extract actual value without count
    actual_value = extract_value(value) if isinstance(value, str) else value
    st.session_state.filters[filter_name] = actual_value

# Get options with counts for dropdowns based on current filters
mutation_signature_options = add_counts_to_options(df, "mutation_signature", st.session_state.filters)
mirna_family_options = add_counts_to_options(df, "mirna_family", st.session_state.filters)
gene_type_options = add_counts_to_options(df, "gene_type", st.session_state.filters)
cancer_promoting_options = add_counts_to_options(df, "is_cancer_promoting", st.session_state.filters)
triplet_behavior_options = add_counts_to_options(df, "triplet_behavior", st.session_state.filters)
gene_name_options = add_counts_to_options(df, "gene_name", st.session_state.filters)

# Find the index of the selected option in each dropdown
def find_index(options, value):
    for i, option in enumerate(options):
        if option == "All" and value == "All":
            return 0
        if extract_value(option) == value:
            return i
    return 0  # Default to "All" if not found

# Create a dropdown with an inline reset button and search
def filter_with_search_reset(label, options, session_key, filter_name, index):
    # Create a container for the entire component
    container = st.sidebar.container()
    
    # Add the label with a reset button
    col1, col2 = container.columns([5, 1])
    col1.markdown(f"**{label}**")
    reset_button = col2.button("↻", key=f"reset_button_{filter_name}", help="Reset this filter")
    
    # Add a search box for filtering options
    search_term = container.text_input(
        "Search", 
        key=f"search_{filter_name}",
        placeholder=f"Search {label.split(' ')[0].lower()}...",
        label_visibility="collapsed"
    )
    
    # Filter options based on search term
    if search_term:
        filtered_options = [opt for opt in options if search_term.lower() in opt.lower()]
        # Always keep "All" option
        if "All" not in filtered_options and "All" in options:
            filtered_options = ["All"] + filtered_options
        display_options = filtered_options if filtered_options else options
    else:
        display_options = options
    
    # Add the dropdown below (full width)
    selected = container.selectbox(
        "",  # Empty label since we already showed it above
        display_options,
        index=min(index, len(display_options)-1) if display_options else 0,
        key=session_key,
        on_change=lambda: update_filters(filter_name, st.session_state[session_key]),
        label_visibility="collapsed"  # Hide the label completely
    )
    
    # Handle reset button click
    if reset_button:
        st.session_state[f"reset_{filter_name}"] = True
        st.rerun()
        
    return selected

# Dropdown filters with counts, search and reset buttons
selected_signature = filter_with_search_reset(
    f"Mutation Signature ({len(mutation_signature_options)-1})",
    mutation_signature_options,
    "mutation_signature_select",
    "mutation_signature",
    find_index(mutation_signature_options, st.session_state.filters['mutation_signature'])
)

selected_mirna = filter_with_search_reset(
    f"miRNA Family ({len(mirna_family_options)-1})",
    mirna_family_options,
    "mirna_family_select",
    "mirna_family",
    find_index(mirna_family_options, st.session_state.filters['mirna_family'])
)

selected_gene_type = filter_with_search_reset(
    f"Gene Type ({len(gene_type_options)-1})",
    gene_type_options,
    "gene_type_select",
    "gene_type",
    find_index(gene_type_options, st.session_state.filters['gene_type'])
)

selected_cancer_promoting = filter_with_search_reset(
    "Cancer Promoting (2)",
    cancer_promoting_options,
    "is_cancer_promoting_select",
    "is_cancer_promoting",
    find_index(cancer_promoting_options, st.session_state.filters['is_cancer_promoting'])
)

selected_behavior = filter_with_search_reset(
    f"Triplet Behavior ({len(triplet_behavior_options)-1})",
    triplet_behavior_options,
    "triplet_behavior_select",
    "triplet_behavior",
    find_index(triplet_behavior_options, st.session_state.filters['triplet_behavior'])
)

# Gene name dropdown with search
selected_gene = filter_with_search_reset(
    f"Gene Name ({len(gene_name_options)-1})",
    gene_name_options,
    "gene_name_select",
    "gene_name",
    find_index(gene_name_options, st.session_state.filters['gene_name'])
)

# Visualization buttons
st.sidebar.markdown("---")
st.sidebar.header("Visualizations")
vis_col1, vis_col2 = st.sidebar.columns(2)

with vis_col1:
    if st.button("Network Graph", type="secondary", use_container_width=True):
        st.session_state.show_network = not st.session_state.show_network

with vis_col2:
    if st.button("Heatmap", type="secondary", use_container_width=True):
        st.session_state.show_heatmap = not st.session_state.show_heatmap

# Reset all filters button
st.sidebar.markdown("---")
if st.sidebar.button("Reset All Filters", type="primary"):
    st.session_state.reset_all_filters = True
    st.rerun()

# Get actual filter values for filtering
actual_signature = extract_value(selected_signature)
actual_mirna = extract_value(selected_mirna)
actual_gene_type = extract_value(selected_gene_type)
actual_cancer_promoting = extract_value(selected_cancer_promoting)
actual_behavior = extract_value(selected_behavior)
actual_gene = extract_value(selected_gene)

# Apply all filters to get the final dataframe
filtered_df = df.copy()

# Apply dropdown filters
if actual_signature != "All":
    filtered_df = filtered_df[filtered_df["mutation_signature"] == actual_signature]

if actual_mirna != "All":
    filtered_df = filtered_df[filtered_df["mirna_family"] == actual_mirna]

if actual_gene_type != "All":
    filtered_df = filtered_df[filtered_df["gene_type"] == actual_gene_type]

if actual_cancer_promoting != "All":
    filtered_df = filtered_df[filtered_df["is_cancer_promoting"].astype(str) == actual_cancer_promoting]

if actual_behavior != "All":
    filtered_df = filtered_df[filtered_df["triplet_behavior"] == actual_behavior]

if actual_gene != "All":
    filtered_df = filtered_df[filtered_df["gene_name"] == actual_gene]

# Main content area - Results
st.header("Results")

if filtered_df.empty:
    st.warning("No matching results.")
else:
    st.success(f"{len(filtered_df)} results found.")
    st.dataframe(filtered_df)
    
    # Heatmap visualization
    if st.session_state.show_heatmap:
        st.header("Signature-Gene Heatmap")
        
        with st.expander("Heatmap Options", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                top_n_genes = st.slider("Number of Top Genes", min_value=10, max_value=50, value=30)
            with col2:
                # Get the unique signatures to select from
                available_sigs = sorted(filtered_df["mutation_signature"].unique().tolist())
                all_sigs = st.multiselect(
                    "Mutation Signatures to Include",
                    options=available_sigs,
                    default=available_sigs[:min(12, len(available_sigs))]
                )
                if not all_sigs:  # If none selected, use all available
                    all_sigs = available_sigs
                    
        # Function to plot heatmap
        def plot_signature_gene_heatmap(df, top_n_genes=30, all_signatures=None):
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
            if all_signatures is None or len(all_signatures) == 0:
                all_sigs = sorted(df["mutation_signature"].unique())
            else:
                all_sigs = all_signatures

            # build full matrix and include all signatures
            full_matrix = data.pivot(index='gene_name', columns='mutation_signature', values='total_hits').fillna(0)
            # Ensure all selected signatures are included (might not all be in the filtered data)
            full_matrix = full_matrix.reindex(columns=all_sigs, fill_value=0)

            # select top genes by total hits
            top_genes = full_matrix.sum(axis=1).nlargest(top_n_genes).index
            if len(top_genes) == 0:
                st.warning("No genes found with these filters.")
                return None
                
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
            fig = plt.figure(figsize=(14, 10))
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
            if len(sig_net) > 0:
                ymax = max(1, np.ceil(np.abs(sig_net).max() * 1.1)) if len(sig_net) > 0 else 1
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
            if len(gene_fc) > 0:
                xmax = max(1, np.ceil(np.abs(gene_fc).max() * 1.1)) if len(gene_fc) > 0 else 1
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
            
            vmax = heatmap_df.values.max() if heatmap_df.values.max() > 0 else 1
            norm = PowerNorm(gamma=0.4, vmin=0, vmax=vmax)

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
            return fig
        
        # Check if we have enough data
        if len(filtered_df) > 0 and len(filtered_df['gene_name'].unique()) > 0:
            fig = plot_signature_gene_heatmap(filtered_df, top_n_genes=top_n_genes, all_signatures=all_sigs)
            if fig:
                st.pyplot(fig)
        else:
            st.warning("Not enough data for heatmap visualization.")
        
        # Explanation of the heatmap
        st.markdown("""
        ### Heatmap Explanation
        
        **Center Heatmap**: Shows the distribution of hits across genes (rows) and mutation signatures (columns).
        - Numbers in cells show the actual hit count
        - Color intensity represents hit density (logarithmic scale)
        
        **Gene Labels**:
        - **Oncogenes** (orange)
        - **Tumor Suppressors** (green)
        - **Dual Role** (purple)
        - **Unannotated** (gray)
        - **Bold highlighted** genes follow expected behavior (oncogenes upregulated, tumor suppressors downregulated)
        
        **Left Bar Chart**: Shows log2 fold change for each gene
        - **Green**: Upregulated (positive log2FC)
        - **Red**: Downregulated (negative log2FC)
        
        **Top Bar Chart**: Shows net effect of each signature
        - Positive (green): More upregulating hits
        - Negative (red): More downregulating hits
        """)
    
    # Interactive Network visualization
    if st.session_state.show_network:
        st.header("Interactive Network Visualization")
        
        # Limit to top entities if there are too many
        max_items = 100
        if len(filtered_df) > max_items:
            st.warning(f"Showing network for top {max_items} items only for better visualization")
            filtered_df_network = filtered_df.head(max_items)
        else:
            filtered_df_network = filtered_df
        
        # Network visualization options
        col1, col2 = st.columns(2)
        with col1:
            show_labels = st.checkbox("Show All Labels", value=False)
        with col2:
            node_size_factor = st.slider("Node Size", min_value=5, max_value=20, value=10)
        
        # Create a Pyvis network
        net = Network(notebook=True, height="600px", width="100%", bgcolor="#222222", font_color="white")
        
        # Set physics layout options
        net.barnes_hut(spring_length=150, spring_strength=0.01, damping=0.09, central_gravity=0.3)
        
        # Create network graph first with networkx to get node sizes based on degree
        G = nx.Graph()
        
        # Add nodes for each category
        signatures = set(filtered_df_network["mutation_signature"])
        mirnas = set(filtered_df_network["mirna_family"])
        genes = set(filtered_df_network["gene_name"])
        
        # Add nodes with different node types (for coloring)
        for sig in signatures:
            G.add_node(sig, group="signature")
        
        for mirna in mirnas:
            G.add_node(mirna, group="mirna")
            
        for gene in genes:
            G.add_node(gene, group="gene")
            
        # Add edges between connected entities
        for _, row in filtered_df_network.iterrows():
            G.add_edge(row["mutation_signature"], row["mirna_family"])
            G.add_edge(row["mirna_family"], row["gene_name"])
        
        # Calculate degrees for sizing
        degrees = dict(G.degree())
        
        # Add nodes to the Pyvis network
        group_colors = {
            "signature": "#4287f5",  # blue
            "mirna": "#f5a142",     # orange
            "gene": "#42f56f"       # green
        }
        
        group_titles = {
            "signature": "Mutation Signature",
            "mirna": "miRNA Family",
            "gene": "Gene"
        }
        
        # Add nodes to Pyvis network
        for node, attrs in G.nodes(data=True):
            group = attrs['group']
            title = f"{group_titles[group]}: {node}<br>Connections: {degrees[node]}"
            # Size nodes based on their degree
            size = 10 + (degrees[node] * node_size_factor)
            
            # Show labels based on checkbox or if node has many connections
            label = node if show_labels or degrees[node] > 2 else ""
            
            net.add_node(node, 
                        label=label, 
                        title=title, 
                        color=group_colors[group], 
                        size=size,
                        group=group)
        
        # Add edges
        for source, target in G.edges():
            net.add_edge(source, target, title=f"{source} -> {target}")
        
        # Generate the interactive visualization
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
            net.save_graph(temp_file.name)
            # Display in Streamlit
            with open(temp_file.name, 'r', encoding='utf-8') as f:
                html_string = f.read()
            
            # Inject some CSS to fix the background color to match Streamlit
            html_string = html_string.replace('<style type="text/css">', 
                                             '<style type="text/css">\nbody {background-color: transparent !important;}\n')
            
            st.components.v1.html(html_string, height=600)
            os.unlink(temp_file.name)  # Delete the temp file
        
        # Network stats
        st.subheader("Network Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes", len(G.nodes))
            st.metric("Total Relationships", len(G.edges))
        
        with col2:
            # Find most connected entities by category
            if signatures:
                most_connected_sig = max(signatures, key=lambda x: G.degree(x))
                st.metric("Top Mutation Signature", most_connected_sig, 
                         f"{G.degree(most_connected_sig)} connections")
            
        with col3:
            if mirnas:
                most_connected_mirna = max(mirnas, key=lambda x: G.degree(x))
                st.metric("Top miRNA Family", most_connected_mirna,
                         f"{G.degree(most_connected_mirna)} connections")
            
            if genes:
                most_connected_gene = max(genes, key=lambda x: G.degree(x))
                st.metric("Top Gene", most_connected_gene, 
                         f"{G.degree(most_connected_gene)} connections")
        
        st.info("""
        **Interactive Network Features**:
        - **Zoom**: Scroll to zoom in/out
        - **Pan**: Click and drag to move around the network
        - **Select**: Click on nodes to highlight connections
        - **Hover**: Mouse over nodes for detailed information
        - **Move Nodes**: Drag nodes to rearrange the network
        """)

    # Show visualization charts in columns
    st.header("Data Visualization")
    col1, col2 = st.columns(2)
    with col1:
        # plot 1: mutation signature counts
        st.subheader("Mutation Signature Distribution")
        sig_counts = filtered_df["mutation_signature"].value_counts()
        fig, ax = plt.subplots()
        sig_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Mutation Signature")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # plot 2: miRNA families
        st.subheader("miRNA Family Hits")
        mirna_counts = filtered_df["mirna_family"].value_counts().head(10)  # Top 10
        fig2, ax2 = plt.subplots()
        mirna_counts.plot(kind="bar", ax=ax2, color="purple")
        ax2.set_xlabel("miRNA Family")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
    with col2:
        # plot 3: gene roles
        if "gene_type" in filtered_df.columns:
            st.subheader("Gene Type Composition")
            type_counts = filtered_df["gene_type"].value_counts()
            fig3, ax3 = plt.subplots()
            type_counts.plot(kind="pie", ax=ax3, autopct='%1.1f%%')
            st.pyplot(fig3)