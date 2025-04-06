import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# load data
@st.cache_data
def load_data():
    return pd.read_csv("data/triplets.csv")

df = load_data()

st.title("Triplet Search Dashboard")

# search inputs
gene_query = st.text_input("Search by Gene Name")
mirna_query = st.text_input("Search by miRNA Family")
signature_query = st.text_input("Search by Mutation Signature")

# apply filters
filtered_df = df.copy()

if gene_query:
    filtered_df = filtered_df[filtered_df["gene_name"].str.lower() == gene_query.lower()]

if mirna_query:
    filtered_df = filtered_df[filtered_df["mirna_family"].str.lower() == mirna_query.lower()]

if signature_query:
    filtered_df = filtered_df[filtered_df["mutation_signature"].str.lower() == signature_query.lower()]

# results
if gene_query or mirna_query or signature_query:
    if filtered_df.empty:
        st.warning("No matching results.")
    else:
        st.success(f"{len(filtered_df)} results found.")
        st.dataframe(filtered_df)

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
        mirna_counts = filtered_df["mirna_family"].value_counts()
        fig2, ax2 = plt.subplots()
        mirna_counts.plot(kind="bar", ax=ax2, color="purple")
        ax2.set_xlabel("miRNA Family")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

        # plot 3: gene roles
        if "gene_role" in filtered_df.columns:
            st.subheader("Gene Role Composition")
            role_counts = filtered_df["gene_role"].value_counts()
            fig3, ax3 = plt.subplots()
            role_counts.plot(kind="bar", ax=ax3, color="orange")
            ax3.set_xlabel("Gene Role")
            ax3.set_ylabel("Count")
            st.pyplot(fig3)
