import torch
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# num_nodes = 10
# x = torch.randn(num_nodes, 5)
# x_0 = torch.randn(num_nodes, 5)
# x_1 = torch.randn(num_nodes, 5)
# x_2 = torch.randn(num_nodes, 5)

# # triple edge is included in this example!!!
# edge_index = torch.tensor([
#     [0, 1, 0, 2, 0, 1, 0, 2, 0, 3, 0, 4, 0, 1],
#     [1, 0, 2, 0, 1, 0, 2, 0, 3, 0, 4, 0, 1, 0]
# ])
# edge_index_0 = torch.tensor([
#     [0, 1, 0, 2, 0, 3, 0, 4],
#     [1, 0, 2, 0, 3, 0, 4, 0]
# ])
# edge_index_1 = torch.tensor([
#     [0, 1, 0, 2],
#     [1, 0, 2, 0]
# ])
# edge_index_2 = torch.tensor([
#     [0, 1],
#     [1, 0]
# ])
# edge_attr = torch.tensor([
#     [2, 10],
#     [2, 10],
#     [3, 6],
#     [3, 6],
#     [4, 6],
#     [4, 6],
#     [5, 9],
#     [5, 9],
#     [7, 3],
#     [7, 3],
#     [8, 2],
#     [8, 2],
#     [1, 8],
#     [1, 8]
# ])
# edge_attr_0 = torch.tensor([
#     [2, 10],
#     [2, 10],
#     [3, 6],
#     [3, 6],
#     [7, 3],
#     [7, 3],
#     [8, 2],
#     [8, 2]
# ])
# edge_attr_1 = torch.tensor([
#     [3, 6],
#     [3, 6],
#     [4, 6],
#     [4, 6]
# ])
# edge_attr_2 = torch.tensor([
#     [1, 8],
#     [1, 8]
# ])
# edge_type = torch.tensor([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1])
# edge_type_0 = torch.tensor([1, 1, 2, 2, 1, 1, 2, 2])
# edge_type_1 = torch.tensor([1, 1, 2, 2])
# edge_type_2 = torch.tensor([1, 1])

# data = Data(
#     num_nodes=num_nodes, x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type,
#     x_0=x_0, edge_index_0=edge_index_0, edge_attr_0=edge_attr_0, edge_type_0=edge_type_0,
#     x_1=x_1, edge_index_1=edge_index_1, edge_attr_1=edge_attr_1, edge_type_1=edge_type_1,
#     x_2=x_2, edge_index_2=edge_index_2, edge_attr_2=edge_attr_2, edge_type_2=edge_type_2)
# print(data.is_undirected())
# print("\n")

# data_list = [data, data]
# loader = DataLoader(data_list, batch_size=len(data_list))
# batch = next(iter(loader))

# print(batch)
# print("\n")
# print(batch.x)
# print(batch.x_0)
# print(batch.x_1)
# print(batch.x_2)
# print("\n")
# print(batch.edge_index)
# print(batch.edge_index_0)
# print(batch.edge_index_1)
# print(batch.edge_index_2)
# print("\n")
# print(batch.edge_attr)
# print(batch.edge_attr_0)
# print(batch.edge_attr_1)
# print(batch.edge_attr_2)
# print("\n")
# print(batch.edge_type)
# print(batch.edge_type_0)
# print(batch.edge_type_1)
# print(batch.edge_type_2)
# print("\n")
# print(batch.batch)













# num_nodes = 10
# x = torch.randn(num_nodes, 5)
# x_0 = torch.randn(num_nodes, 5)
# x_1 = torch.randn(num_nodes, 5)
# x_2 = torch.randn(num_nodes, 5)

# # triple edge is not included in this example!!!
# edge_index = torch.tensor([
#     [0, 1, 0, 2, 0, 1, 0, 2, 0, 3, 0, 4],
#     [1, 0, 2, 0, 1, 0, 2, 0, 3, 0, 4, 0]
# ])
# edge_index_0 = torch.tensor([
#     [0, 1, 0, 2, 0, 3, 0, 4],
#     [1, 0, 2, 0, 3, 0, 4, 0]
# ])
# edge_index_1 = torch.tensor([
#     [0, 1, 0, 2],
#     [1, 0, 2, 0]
# ])
# edge_index_2 = torch.tensor([
#     [-1, -1],
#     [-1, -1]
# ])
# edge_attr = torch.tensor([
#     [2, 10],
#     [2, 10],
#     [3, 6],
#     [3, 6],
#     [4, 6],
#     [4, 6],
#     [5, 9],
#     [5, 9],
#     [7, 3],
#     [7, 3],
#     [8, 2],
#     [8, 2]
# ])
# edge_attr_0 = torch.tensor([
#     [2, 10],
#     [2, 10],
#     [3, 6],
#     [3, 6],
#     [7, 3],
#     [7, 3],
#     [8, 2],
#     [8, 2]
# ])
# edge_attr_1 = torch.tensor([
#     [3, 6],
#     [3, 6],
#     [4, 6],
#     [4, 6]
# ])
# edge_attr_2 = torch.tensor([
#     [-1, -1],
#     [-1, -1]
# ])
# edge_type = torch.tensor([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2])
# edge_type_0 = torch.tensor([1, 1, 2, 2, 1, 1, 2, 2])
# edge_type_1 = torch.tensor([1, 1, 2, 2])
# edge_type_2 = torch.tensor([-1, -1])

# data = Data(
#     num_nodes=num_nodes, x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type,
#     x_0=x_0, edge_index_0=edge_index_0, edge_attr_0=edge_attr_0, edge_type_0=edge_type_0,
#     x_1=x_1, edge_index_1=edge_index_1, edge_attr_1=edge_attr_1, edge_type_1=edge_type_1,
#     x_2=x_2, edge_index_2=edge_index_2, edge_attr_2=edge_attr_2, edge_type_2=edge_type_2)
# print(data.is_undirected())
# print("\n")

# data_list = [data, data]
# loader = DataLoader(data_list, batch_size=len(data_list))
# batch = next(iter(loader))

# print(batch)
# print("\n")
# print(batch.x)
# print(batch.x_0)
# print(batch.x_1)
# print(batch.x_2)
# print("\n")
# print(batch.edge_index)
# print(batch.edge_index_0)
# print(batch.edge_index_1)
# print(batch.edge_index_2)
# print("\n")
# # print(batch.edge_attr)
# # print(batch.edge_attr_0)
# # print(batch.edge_attr_1)
# # print(batch.edge_attr_2)
# # print("\n")
# # print(batch.edge_type)
# # print(batch.edge_type_0)
# # print(batch.edge_type_1)
# # print(batch.edge_type_2)
# # print("\n")
# print(batch.batch)
# print("\n")
# # print(batch.num_nodes)
# # print(batch.batch_size)
# # print(batch.num_nodes // batch.batch_size)
# # cum_num_nodes = torch.cumsum(torch.cat([torch.tensor([0]), torch.bincount(batch.batch)[:-1]]), dim=0)
# # print(cum_num_nodes)
# # edge_offsets = cum_num_nodes[batch.batch[batch.edge_index[0]]]
# # print(edge_offsets)
# # edge_offsets = cum_num_nodes[batch.batch[batch.edge_index_0[0]]]
# # print(edge_offsets)
# # edge_offsets = cum_num_nodes[batch.batch[batch.edge_index_1[0]]]
# # print(edge_offsets)
# # edge_offsets = cum_num_nodes[batch.batch[batch.edge_index_2[0]]]
# # print(edge_offsets)
# print(batch.batch[batch.edge_index[0]])
# print(batch.batch[batch.edge_index_0[0]])
# print(batch.batch[batch.edge_index_1[0]])
# print(batch.batch[batch.edge_index_2[0]])














num_nodes = 10
x = torch.randn(num_nodes, 5)
x_0 = torch.randn(num_nodes, 5)
x_1 = torch.randn(num_nodes, 5)
x_2 = torch.randn(num_nodes, 5)

# triple edge is not included in this example!!!
edge_index = torch.tensor([
    [0, 1, 0, 2, 0, 1, 0, 2, 0, 3, 0, 4],
    [1, 0, 2, 0, 1, 0, 2, 0, 3, 0, 4, 0]
])
edge_index_0 = torch.tensor([
    [0, 1, 0, 2, 0, 3, 0, 4],
    [1, 0, 2, 0, 3, 0, 4, 0]
])
edge_index_1 = torch.tensor([
    [0, 1, 0, 2],
    [1, 0, 2, 0]
])
edge_index_2 = torch.empty((2, 0), dtype=torch.long)
edge_attr = torch.tensor([
    [2, 10],
    [2, 10],
    [3, 6],
    [3, 6],
    [4, 6],
    [4, 6],
    [5, 9],
    [5, 9],
    [7, 3],
    [7, 3],
    [8, 2],
    [8, 2]
])
edge_attr_0 = torch.tensor([
    [2, 10],
    [2, 10],
    [3, 6],
    [3, 6],
    [7, 3],
    [7, 3],
    [8, 2],
    [8, 2]
])
edge_attr_1 = torch.tensor([
    [3, 6],
    [3, 6],
    [4, 6],
    [4, 6]
])
edge_attr_2 = torch.empty((0, 2), dtype=torch.long)

edge_type = torch.tensor([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2])
edge_type_0 = torch.tensor([1, 1, 2, 2, 1, 1, 2, 2])
edge_type_1 = torch.tensor([1, 1, 2, 2])
edge_type_2 = torch.empty((0, 1), dtype=torch.long)

data = Data(
    num_nodes=num_nodes, x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type,
    x_0=x_0, edge_index_0=edge_index_0, edge_attr_0=edge_attr_0, edge_type_0=edge_type_0,
    x_1=x_1, edge_index_1=edge_index_1, edge_attr_1=edge_attr_1, edge_type_1=edge_type_1,
    x_2=x_2, edge_index_2=edge_index_2, edge_attr_2=edge_attr_2, edge_type_2=edge_type_2)
print(data.is_undirected())
print("\n")
print(data.validate())
print("\n")
print(data.__inc__("edge_index", None))
print(data.__inc__("edge_index_0", None))
print(data.__inc__("edge_attr_0", None))
print(data.__inc__("edge_type_0", None))
print("\n")
print(data.__cat_dim__("edge_index", None))
print(data.__cat_dim__("edge_index_0", None))
print(data.__cat_dim__("edge_attr_0", None))
print(data.__cat_dim__("edge_type_0", None))
print("\n")
print(edge_index.shape, edge_index.shape[data.__cat_dim__("edge_index", None)])
print(edge_index_0.shape, edge_index_0.shape[data.__cat_dim__("edge_index_0", None)])
print(edge_attr_0.shape, edge_attr_0.shape[data.__cat_dim__("edge_attr_0", None)])
print(edge_type_0.shape, edge_type_0.shape[data.__cat_dim__("edge_type_0", None)])

# data_list = [data, data]
# loader = DataLoader(data_list, batch_size=len(data_list))
# batch = next(iter(loader))

# print(batch)
# print("\n")
# print(batch.x)
# print(batch.x_0)
# print(batch.x_1)
# print(batch.x_2)
# print("\n")
# print(batch.edge_index)
# print(batch.edge_index_0)
# print(batch.edge_index_1)
# print(batch.edge_index_2)
# print("\n")
# # print(batch.edge_attr)
# # print(batch.edge_attr_0)
# # print(batch.edge_attr_1)
# # print(batch.edge_attr_2)
# # print("\n")
# # print(batch.edge_type)
# # print(batch.edge_type_0)
# # print(batch.edge_type_1)
# # print(batch.edge_type_2)
# # print("\n")
# print(batch.batch)
# print("\n")
# # print(batch.num_nodes)
# # print(batch.batch_size)
# # print(batch.num_nodes // batch.batch_size)
# # cum_num_nodes = torch.cumsum(torch.cat([torch.tensor([0]), torch.bincount(batch.batch)[:-1]]), dim=0)
# # print(cum_num_nodes)
# # edge_offsets = cum_num_nodes[batch.batch[batch.edge_index[0]]]
# # print(edge_offsets)
# # edge_offsets = cum_num_nodes[batch.batch[batch.edge_index_0[0]]]
# # print(edge_offsets)
# # edge_offsets = cum_num_nodes[batch.batch[batch.edge_index_1[0]]]
# # print(edge_offsets)
# # edge_offsets = cum_num_nodes[batch.batch[batch.edge_index_2[0]]]
# # print(edge_offsets)
# print(batch.batch[batch.edge_index[0]])
# print(batch.batch[batch.edge_index_0[0]])
# print(batch.batch[batch.edge_index_1[0]])
# print(batch.batch[batch.edge_index_2[0]])