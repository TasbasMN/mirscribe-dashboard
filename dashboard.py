import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import tempfile
import os

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
        'triplet_behavior': "All"
    }

# Initialize show_network if not present
if 'show_network' not in st.session_state:
    st.session_state.show_network = False

# Initialize reset flags if not present
for filter_name in list(st.session_state.filters.keys()) + ['gene_query', 'all_filters']:
    reset_key = f"reset_{filter_name}"
    if reset_key not in st.session_state:
        st.session_state[reset_key] = False

# Handle resets before rendering widgets
if st.session_state.reset_all_filters:
    for key in st.session_state.filters:
        st.session_state.filters[key] = "All"
    if 'gene_query_input' in st.session_state:
        st.session_state.gene_query_input = ""
    st.session_state.reset_all_filters = False

for filter_name in st.session_state.filters.keys():
    reset_key = f"reset_{filter_name}"
    if st.session_state[reset_key]:
        st.session_state.filters[filter_name] = "All"
        st.session_state[reset_key] = False

if st.session_state.reset_gene_query:
    if 'gene_query_input' in st.session_state:
        st.session_state.gene_query_input = ""
    st.session_state.reset_gene_query = False

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
                elif filter_name == 'gene_query':
                    if 'gene_query_input' in st.session_state and st.session_state.gene_query_input:
                        filtered_df = filtered_df[filtered_df["gene_name"].str.lower().str.contains(st.session_state.gene_query_input.lower())]
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

# Find the index of the selected option in each dropdown
def find_index(options, value):
    for i, option in enumerate(options):
        if option == "All" and value == "All":
            return 0
        if extract_value(option) == value:
            return i
    return 0  # Default to "All" if not found

# Create a dropdown with an inline reset button
def filter_with_reset_inline(label, options, session_key, filter_name, index):
    # Create a container for the entire component
    container = st.sidebar.container()
    
    # Add the label with a reset button
    col1, col2 = container.columns([5, 1])
    col1.markdown(f"**{label}**")
    reset_button = col2.button("↻", key=f"reset_button_{filter_name}", help="Reset this filter")
    
    # Add the dropdown below (full width)
    selected = container.selectbox(
        "",  # Empty label since we already showed it above
        options,
        index=index,
        key=session_key,
        on_change=lambda: update_filters(filter_name, st.session_state[session_key]),
        label_visibility="collapsed"  # Hide the label completely
    )
    
    # Handle reset button click
    if reset_button:
        st.session_state[f"reset_{filter_name}"] = True
        st.rerun()
        
    return selected

# Dropdown filters with counts and reset buttons
selected_signature = filter_with_reset_inline(
    f"Mutation Signature ({len(mutation_signature_options)-1})",
    mutation_signature_options,
    "mutation_signature_select",
    "mutation_signature",
    find_index(mutation_signature_options, st.session_state.filters['mutation_signature'])
)

selected_mirna = filter_with_reset_inline(
    f"miRNA Family ({len(mirna_family_options)-1})",
    mirna_family_options,
    "mirna_family_select",
    "mirna_family",
    find_index(mirna_family_options, st.session_state.filters['mirna_family'])
)

selected_gene_type = filter_with_reset_inline(
    f"Gene Type ({len(gene_type_options)-1})",
    gene_type_options,
    "gene_type_select",
    "gene_type",
    find_index(gene_type_options, st.session_state.filters['gene_type'])
)

selected_cancer_promoting = filter_with_reset_inline(
    "Cancer Promoting (2)",
    cancer_promoting_options,
    "is_cancer_promoting_select",
    "is_cancer_promoting",
    find_index(cancer_promoting_options, st.session_state.filters['is_cancer_promoting'])
)

selected_behavior = filter_with_reset_inline(
    f"Triplet Behavior ({len(triplet_behavior_options)-1})",
    triplet_behavior_options,
    "triplet_behavior_select",
    "triplet_behavior",
    find_index(triplet_behavior_options, st.session_state.filters['triplet_behavior'])
)

# Text search with reset button
st.sidebar.header("Text Search")

# Create a container for the search
search_container = st.sidebar.container()
search_col1, search_col2 = search_container.columns([5, 1])
search_col1.markdown("**Search by Gene Name**")
reset_search_button = search_col2.button("↻", key="reset_button_gene_query", help="Clear search")

gene_query = search_container.text_input(
    "",  # Empty label since we already showed it above
    key="gene_query_input",
    on_change=lambda: update_filters('gene_query', st.session_state.gene_query_input),
    label_visibility="collapsed"  # Hide the label completely
)

# Handle search reset button click
if reset_search_button:
    st.session_state.reset_gene_query = True
    st.rerun()

# Network plot button
st.sidebar.markdown("---")
if st.sidebar.button("Plot Interactive Network", type="secondary"):
    st.session_state.show_network = not st.session_state.show_network

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

# Apply text search
if gene_query:
    filtered_df = filtered_df[filtered_df["gene_name"].str.lower().str.contains(gene_query.lower())]

# Main content area - Results
st.header("Results")

if filtered_df.empty:
    st.warning("No matching results.")
else:
    st.success(f"{len(filtered_df)} results found.")
    st.dataframe(filtered_df)
    
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