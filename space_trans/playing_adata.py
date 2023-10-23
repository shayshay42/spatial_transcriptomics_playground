import numpy as np

import scanpy as sc
import squidpy as sq

sc.logging.print_header()
print(f"squidpy=={sq.__version__}")

# load the pre-processed dataset
adata = sq.datasets.seqfish()

sq.pl.spatial_scatter(
    adata, color="celltype_mapped_refined", shape=None, figsize=(10, 10)
)

sq.gr.spatial_neighbors(adata, coord_type="generic")
# sq.gr.nhood_enrichment(adata, cluster_key="celltype_mapped_refined")
sq.pl.nhood_enrichment(adata, cluster_key="celltype_mapped_refined", method="ward")

sq.pl.spatial_scatter(
    adata,
    color="celltype_mapped_refined",
    groups=[
        "Endothelium",
        "Haematoendothelial progenitors",
        "Allantois",
        "Lateral plate mesoderm",
        "Intermediate mesoderm",
        "Presomitic mesoderm",
    ],
    shape=None,
    size=2,
)

sq.gr.co_occurrence(adata, cluster_key="celltype_mapped_refined")
sq.pl.co_occurrence(
    adata,
    cluster_key="celltype_mapped_refined",
    clusters="Lateral plate mesoderm",
    figsize=(10, 5),
)

sq.gr.ligrec(
    adata,
    n_perms=100,
    cluster_key="celltype_mapped_refined",
)
sq.pl.ligrec(
    adata,
    cluster_key="celltype_mapped_refined",
    source_groups="Lateral plate mesoderm",
    target_groups=["Intermediate mesoderm", "Allantois"],
    means_range=(0.3, np.inf),
    alpha=1e-4,
    swap_axes=True,
)

dir(adata)

sq.gr.spatial_neighbors(adata)

adata.obsp["spatial_connectivities"]

sq.gr.spatial_neighbors(adata, n_neighs=100, coord_type="generic")
_, idx = adata.obsp["spatial_connectivities"][420, :].nonzero()
idx = np.append(idx, 420)
sq.pl.spatial_scatter(
    adata[idx, :],
    shape=None,
    connectivity_key="spatial_connectivities",
    size=100,
)

arr = adata.obsp["spatial_distances"][420, :].toarray()

print(max(arr[0]))

adj = adata.obsp["spatial_connectivities"][0].toarray()

heatmap(adj)


import torch
from torch_geometric.data import Data
import scipy.sparse as sp

# Convert adjacency matrix to edge index format expected by PyTorch Geometric
# Your adjacency matrix is probably stored in adata.obsp['spatial_connectivities']
def adjacency_to_edge_index(adjacency_matrix):
    coo = adjacency_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    return torch.tensor(indices, dtype=torch.long), torch.tensor(values, dtype=torch.float)

edge_index, edge_weight = adjacency_to_edge_index(adata.obsp['spatial_connectivities'])
x = torch.tensor(adata.X, dtype=torch.float)  # Node features (e.g., gene expressions)

# Create a PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)



from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class SimpleGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SimpleGCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

# Example usage:
conv = SimpleGCNConv(adata.n_vars, 64)  # Adjust input and output dimensions based on your data.
x = adata.X  # Node features
edge_index = data.edge_index  # Edge index

# Perform one round of message passing
new_node_features = conv(x, edge_index)
