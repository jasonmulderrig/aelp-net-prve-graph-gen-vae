import torch
import torch.nn as nn
from torch_geometric.nn.models.autoencoder import InnerProductDecoder
from torch_geometric.utils import batched_negative_sampling

EPS = 1e-15
n_0 = 10
batch_size = 2
n = n_0 * batch_size
d = 5
z = torch.randn((n, d))
inner_product_decoder = InnerProductDecoder()

batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
pos_edge_index = torch.tensor([[0, 1, 1, 2, 0, 13, 0, 14],
                                [1, 0, 2, 1, 13, 0, 14, 0]], dtype=torch.long)
neg_edge_index = batched_negative_sampling(pos_edge_index, batch, z.size(0))
print(neg_edge_index)
pos_edge_probs = inner_product_decoder(z, pos_edge_index)
neg_edge_probs = inner_product_decoder(z, neg_edge_index)
print(pos_edge_probs)
print(pos_edge_probs.numel())
print(neg_edge_probs)
print(neg_edge_probs.numel())
pos_edge_probs_pos_loss = -torch.log(pos_edge_probs + EPS).mean()
neg_edge_probs_pos_loss = -torch.log(1-pos_edge_probs + EPS).mean()
print(pos_edge_probs_pos_loss)
print(neg_edge_probs_pos_loss)

print("\n")

empty_pos_edge_index = torch.empty((2, 0), dtype=torch.long) # If empty, then eventually return pos_loss = 0!!!
empty_neg_edge_index = batched_negative_sampling(empty_pos_edge_index, batch, z.size(0)) # If empty, have num_neg_samples = z.size(0)
print(empty_neg_edge_index)
# empty_pos_edge_probs = inner_product_decoder(z, empty_pos_edge_index)
# empty_neg_edge_probs = inner_product_decoder(z, empty_neg_edge_index)
# print(empty_pos_edge_probs)
# print(empty_pos_edge_probs.numel())
# print(empty_neg_edge_probs)
# print(empty_neg_edge_probs.numel())
# empty_pos_edge_probs_pos_loss = -torch.log(empty_pos_edge_probs + EPS).mean()
# empty_neg_edge_probs_pos_loss = -torch.log(1-empty_neg_edge_probs + EPS).mean()
# print(empty_pos_edge_probs_pos_loss)
# print(empty_neg_edge_probs_pos_loss)