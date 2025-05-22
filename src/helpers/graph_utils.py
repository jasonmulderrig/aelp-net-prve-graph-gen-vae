import numpy as np
import networkx as nx

def lexsorted_edges(
        edges: list[tuple[int, int]] | np.ndarray,
        return_indcs: bool=False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Lexicographically sorted edges.

    This function takes a list (or an np.ndarray) of (A, B) nodes
    specifying edges, converts this to an np.ndarray, and
    lexicographically sorts the edge entries.

    Args:
        edges (list[tuple[int, int]] | np.ndarray): Edges.
        return_indcs (bool): Flag indicating if the indices of the lexicographic edge sort should be returned, or not. Default is False.
    
    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]: Lexicographically
        sorted edges, or the lexicographically sorted edges and the
        indices of the lexicographic edge sort.
    
    """
    edges = np.sort(np.asarray(edges, dtype=int), axis=1)
    lexsort_indcs = np.lexsort((edges[:, 1], edges[:, 0]))
    if return_indcs: return edges[lexsort_indcs], lexsort_indcs
    else: return edges[lexsort_indcs]

# change this to provide back counts of unique lexsorted edges, if
# requested? back impose this to rest of code, and do this for
# irreg_net_prve repo as well.
def unique_lexsorted_edges(edges: list[tuple[int, int]]) -> np.ndarray:
    """Unique lexicographically sorted edges.

    This function takes a list (or an np.ndarray) of (A, B) nodes
    specifying edges, converts this to an np.ndarray, lexicographically
    sorts the edge entries, and extracts the resulting unique edges.

    Args:
        edges (list[tuple[int, int]] | np.ndarray): Edges.
    
    Returns:
        np.ndarray: Unique lexicographically sorted edges.
    
    """
    return np.unique(lexsorted_edges(edges), axis=0)

def add_nodes_from_numpy_array(
        graph: nx.Graph | nx.MultiGraph,
        nodes: np.ndarray) -> nx.Graph | nx.MultiGraph:
    """Add node numbers from a np.ndarray array to an undirected
    NetworkX graph.

    This function adds node numbers from a np.ndarray array to an
    undirected NetworkX graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        nodes (np.ndarray): Node numbers.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph.
    
    """
    graph.add_nodes_from(nodes.tolist())
    return graph

def add_edges_from_numpy_array(
        graph: nx.Graph | nx.MultiGraph,
        edges: np.ndarray) -> nx.Graph | nx.MultiGraph:
    """Add edges from a two-dimensional np.ndarray to an undirected
    NetworkX graph.

    This function adds edges from a two-dimensional np.ndarray to an
    undirected NetworkX graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        edges (np.ndarray): Edges.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph.
    
    """
    graph.add_edges_from(list(tuple(edge) for edge in edges.tolist()))
    return graph

def add_nodes_and_node_attributes_from_numpy_arrays(
        graph: nx.Graph | nx.MultiGraph,
        nodes: np.ndarray,
        attr_names: list[str],
        *attr_vals: tuple[np.ndarray]) -> nx.Graph | nx.MultiGraph:
    """Add node numbers and node attributes from np.ndarray arrays to an
    undirected NetworkX graph.

    This function adds node numbers and associated node attributes from
    np.ndarray arrays to an undirected NetworkX graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        nodes (np.ndarray): Node numbers.
        attr_names (list[str]): Node attribute names.
        *attr_vals (tuple[np.ndarray]): Node attribute values.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph.
    
    """
    # Ensure that the number of node attributes matches the number of
    # provided node attribute arrays
    if len(attr_names) != len(attr_vals):
        error_str = (
            "The number of node attribute names must match the number "
            + "of node attribute value arrays."
        )
        raise ValueError(error_str)
    # Ensure that each node value attribute array has an attribute
    # specified for each node
    if any(np.shape(attr)[0] != np.shape(nodes)[0] for attr in attr_vals):
        error_str = (
            "Each node attribute value array must have the same number "
            + "of entries as the number of nodes."
        )
        raise ValueError(error_str)
    
    # Convert np.ndarrays to lists
    nodes = nodes.tolist()
    attr_vals = tuple(attr.tolist() for attr in attr_vals)

    # Add nodes and node attributes to graph
    graph.add_nodes_from(
        (node, {attr_names[attr_indx]: attr_vals[attr_indx][node_indx] for attr_indx in range(len(attr_names))})
        for node_indx, node in enumerate(nodes)
    )
    return graph

def add_edges_and_edge_attributes_from_numpy_arrays(
        graph: nx.Graph | nx.MultiGraph,
        edges: np.ndarray,
        attr_names: list[str],
        *attr_vals: tuple[np.ndarray]) -> nx.Graph | nx.MultiGraph:
    """Add edges and edge attributes from np.ndarray arrays to an
    undirected NetworkX graph.

    This function adds edges and associated edge attributes from
    np.ndarray arrays to an undirected NetworkX graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        edges (np.ndarray): Edges.
        attr_names (list[str]): Edge attribute names.
        *attr_vals (tuple[np.ndarray]): Edge attribute values.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph.
    
    """
    # Ensure that the number of edge attributes matches the number of
    # provided edge attribute arrays
    if len(attr_names) != len(attr_vals):
        error_str = (
            "The number of edge attribute names must match the number "
            + "of edge attribute value arrays."
        )
        raise ValueError(error_str)
    # Ensure that each edge value attribute array has an attribute
    # specified for each edge
    if any(np.shape(attr)[0] != np.shape(edges)[0] for attr in attr_vals):
        error_str = (
            "Each edge attribute value array must have the same number "
            + "of entries as the number of edges."
        )
        raise ValueError(error_str)
    
    # Convert np.ndarrays to lists
    edges = list(tuple(edge) for edge in edges.tolist())
    attr_vals = tuple(attr.tolist() for attr in attr_vals)

    # Add edges and edge attributes to graph
    graph.add_edges_from(
        (edge[0], edge[1], {attr_names[attr_indx]: attr_vals[attr_indx][edge_indx] for attr_indx in range(len(attr_names))})
        for edge_indx, edge in enumerate(edges)
    )
    return graph

def extract_nodes_to_numpy_array(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Extract nodes from an undirected NetworkX graph to a np.ndarray
    array.

    This function extract nodes from an undirected NetworkX graph to a
    np.ndarray array.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        np.ndarray: Node numbers.
    
    """
    return np.asarray(list(graph.nodes()), dtype=int)

def extract_edges_to_numpy_array(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Extract edges from an undirected NetworkX graph to a np.ndarray
    array.

    This function extract edges from an undirected NetworkX graph to a
    np.ndarray array.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        np.ndarray: Edges.
    
    """
    return np.asarray(list(graph.edges()), dtype=int)

def extract_node_attribute_to_numpy_array(
        graph: nx.Graph | nx.MultiGraph,
        attr_name: str,
        dtype_int: bool=False) -> np.ndarray:
    """Extract node attribute from an undirected NetworkX graph.

    This function extracts a node attribute from an undirected NetworkX
    graph to a np.ndarray array.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        attr_name (str): Node attribute name.
        dtype_int (bool): Boolean indicating if the node attribute is of type int. Default value is False.
    
    Returns:
        np.ndarray: Node attribute.
    
    """
    if dtype_int:
        return np.asarray(
            list(dict(nx.get_node_attributes(graph, attr_name)).values()),
            dtype=int)
    else:
        return np.asarray(
            list(dict(nx.get_node_attributes(graph, attr_name)).values()))

def extract_edge_attribute_to_numpy_array(
        graph: nx.Graph | nx.MultiGraph,
        attr_name: str,
        dtype_int: bool=False) -> np.ndarray:
    """Extract edge attribute from an undirected NetworkX graph.

    This function extracts an edge attribute from an undirected NetworkX
    graph to a np.ndarray array.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        attr_name (str): Edge attribute name.
        dtype_int (bool): Boolean indicating if the edge attribute is of type int. Default value is False.
    
    Returns:
        np.ndarray: Edge attribute.
    
    """
    if dtype_int:
        return np.asarray(
            list(dict(nx.get_edge_attributes(graph, attr_name)).values()),
            dtype=int)
    else:
        return np.asarray(
            list(dict(nx.get_edge_attributes(graph, attr_name)).values()))

def largest_connected_component(
        graph: nx.Graph | nx.MultiGraph) -> nx.Graph | nx.MultiGraph:
    """Isolate and return the largest/maximum connected component.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph that is
        the largest/maximum connected component of the input graph.

    """
    return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()

def edge_id(graph: nx.Graph | nx.MultiGraph) -> tuple[np.ndarray, np.ndarray]:
    """Edge identification.
    
    This function identifies and counts the edges in a graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Edges and number of times each
        edge occurs in the graph.
    
    """
    # Gather edges and edges counts
    graph_edges = np.asarray(list(graph.edges()), dtype=int)
    if graph.is_multigraph():
        graph_edges, graph_edges_counts = np.unique(
            np.sort(graph_edges, axis=1), return_counts=True, axis=0)
    else:
        graph_edges_counts = np.ones(np.shape(graph_edges)[0], dtype=int)
    
    return graph_edges, graph_edges_counts

def yasuda_morita_procedure(
        A: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[int, int]], list[int]]:
    """Yasuda-Morita procedure.
    
    This function applies the Yasuda-Morita procedure to yield the
    elastically-effective network satisfying the Scanlan-Case criteria.

    Args:
        A (np.ndarray): Adjacency matrix (with no multiedges).
    
    Returns:
        tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[int, int]], list[int]]:
        Adjacency matrix of the elastically-effective network (with no
        multiedges), list of nodes involved with removed bridge centers,
        list of nodes involved with removed dangling edges, and list of
        nodes involved with removed self-loops.
    """
    # Gather nodes
    n = np.shape(A)[0]
    nodes = np.arange(n, dtype=int)

    # Initialize lists
    bridge_center_node_list = []
    dangling_node_list = []
    loop_list = []

    # Yasuda-Morita procedure
    while True:
        # Initialize trackers
        bridge_center_node_elim = False
        dangling_node_elim = False
        loop_elim = False
        
        # Bridge center node elimination
        for center_node in range(n):
            center_node = int(center_node)
            # Edges excluding self-loops
            A_row = np.delete(A[center_node, :], center_node, axis=0)
            A_row_nodes = np.delete(nodes, center_node, axis=0)
            # Check if node is a bridge center node
            if np.sum(A_row) == 2:
                bridge_center_node_elim = True
                # Check if the bridge bridges the same two nodes, and
                # thus is actually a bridging loop
                if np.size(np.where(A_row == 2)[0]) > 0:
                    root_node = int(A_row_nodes[np.where(A_row == 2)[0][0]])
                    # Eliminate the bridge center node from the network
                    A[root_node, center_node] = 0
                    A[center_node, root_node] = 0
                    # Add loop to root node
                    A[root_node, root_node] += 2
                    # Add to bridge center node list
                    bridge_center_node_list.append(
                        (root_node, center_node, root_node))
                # Otherwise, the bridge bridges two distinct nodes
                else:
                    # Identify all nodes involved in the bridge
                    bridge_nodes = A_row_nodes[np.where(A_row == 1)[0]]
                    left_node = int(bridge_nodes[0])
                    right_node = int(bridge_nodes[1])
                    # Eliminate the bridge center node from the network
                    A[left_node, center_node] = 0
                    A[center_node, right_node] = 0
                    A[right_node, center_node] = 0
                    A[center_node, left_node] = 0
                    # Ensure bridge remains intact in the network
                    A[left_node, right_node] += 1
                    A[right_node, left_node] += 1
                    # Add to bridge center node list
                    bridge_center_node_list.append(
                        (left_node, center_node, right_node))
                break
        if bridge_center_node_elim == True:
            continue
        else:
            
            # Dangling node elimination
            for dangling_node in range(n):
                dangling_node = int(dangling_node)
                # Edges excluding self-loops
                A_row = np.delete(A[dangling_node, :], dangling_node, axis=0)
                A_row_nodes = np.delete(nodes, dangling_node, axis=0)
                # Check if node is a dangling node
                if np.sum(A_row) == 1:
                    dangling_node_elim = True
                    # Identify the root node for the dangling node
                    root_node = int(A_row_nodes[np.where(A_row == 1)[0][0]])
                    # Eliminate the dangling node from the network
                    A[root_node, dangling_node] = 0
                    A[dangling_node, root_node] = 0
                    # Add to dangling node list
                    dangling_node_list.append((root_node, dangling_node))
                    break
            if dangling_node_elim == True:
                continue
            else:
                
                # Loop elimination
                for node in range(n):
                    node = int(node)
                    if A[node, node] >= 2:
                        loop_elim = True
                        # Eliminate the loop from the network
                        A[node, node] -= 2
                        # Add to loop list
                        loop_list.append(node)
                        break
                if loop_elim == True:
                    continue
                else: break # Yasuda-Morita procedure has finished
    
    return A, bridge_center_node_list, dangling_node_list, loop_list

def surviving_bridge_restoration(
        A: np.ndarray,
        bridge_center_node_list: list[tuple[int, int, int]]) -> np.ndarray:
    """Surviving bridge restoration.

    This function adds back bridges that were once removed to yield the
    most fundamental elastically-effective network but yet still exist
    between surviving nodes in end-linked networks.

    Args:
        A (np.ndarray): Adjacency matrix (with no multiedges).
        bridge_center_node_list (list[tuple[int, int, int]]): List of nodes involved with bridge centers that were removed to yield the most fundamental elastically-effective network.
    
    Returns:
        np.ndarray: Adjacency matrix of the elastically-effective
        network (with no multiedges) containing bridge centers in
        between surviving nodes.
    
    """
    # Add back bridging centers between surviving nodes in reverse
    # elimination order
    bridge_center_node_list.reverse()

    for bridge_nodes in bridge_center_node_list:
        left_node = int(bridge_nodes[0])
        center_node = int(bridge_nodes[1])
        right_node = int(bridge_nodes[2])

        # Ignore bridging loops
        if left_node == right_node: pass
        else:
            # Add back bridging center node between surviving nodes
            if np.sum(A[left_node, :]) > 0 and np.sum(A[right_node, :]) > 0:
                A[left_node, right_node] -= 1
                A[right_node, left_node] -= 1
                A[left_node, center_node] += 1
                A[center_node, right_node] += 1
                A[right_node, center_node] += 1
                A[center_node, left_node] += 1
            # Ignore bridges that bridge eliminated nodes
            else: pass
    
    return A

def multiedge_restoration(
        graph: nx.Graph | nx.MultiGraph,
        graph_edges: np.ndarray,
        graph_edges_counts: np.ndarray) -> nx.Graph | nx.MultiGraph:
    """Multiedge restoration.
    
    This function restores multiedges in a graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        np.ndarray: Edges.
        np.ndarray: Number of (multi)edges involved for each edge.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph.
    
    """
    # If the graph is of type nx.MultiGraph, then add back redundant
    # edges to all multiedges.
    if graph.is_multigraph():
        # Address multiedges by adding back redundant edges
        if np.any(graph_edges_counts > 1):
            # Extract multiedges
            multiedges = np.where(graph_edges_counts > 1)[0]
            for multiedge in np.nditer(multiedges):
                multiedge = int(multiedge)
                # Number of edges in the multiedge
                edge_num = graph_edges_counts[multiedge]
                # Multiedge nodes
                node_0 = int(graph_edges[multiedge, 0])
                node_1 = int(graph_edges[multiedge, 1])
                # Add back redundant edges
                if graph.has_edge(node_0, node_1):
                    graph.add_edges_from(
                        list((node_0, node_1) for _ in range(edge_num-1)))
    
    return graph

def elastically_effective_graph(
        graph: nx.Graph | nx.MultiGraph) -> nx.Graph | nx.MultiGraph:
    """Elastically-effective graph.

    This function returns the portion of a given graph that corresponds
    to the elastically-effective network in the graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph
        corresponding to the elastically-effective network of the given
        graph.
    
    """
    # Gather edges and edge counts
    graph_edges, graph_edges_counts = edge_id(graph)

    # Acquire adjacency matrix with no multiedges
    if graph.is_multigraph():
        A = nx.to_numpy_array(nx.Graph(graph), dtype=int)
    else:
        A = nx.to_numpy_array(graph, dtype=int)
    
    # Apply the Yasuda-Morita procedure to return the
    # elastically-effective network that satisfies the Scanlan-Case
    # criteria.
    A, _, _, _ = yasuda_morita_procedure(A)

    # Acquire NetworkX graph
    if graph.is_multigraph():
        graph = nx.from_numpy_array(A, create_using=nx.MultiGraph)
    else:
        graph = nx.from_numpy_array(A)

    # Restore multiedges
    graph = multiedge_restoration(graph, graph_edges, graph_edges_counts)

    # As a hard fail-safe, isolate and return the largest/maximum
    # connected component
    return largest_connected_component(graph)

def elastically_effective_end_linked_graph(
        graph: nx.Graph | nx.MultiGraph) -> nx.Graph | nx.MultiGraph:
    """Elastically-effective end-linked graph.

    This function returns the portion of a given graph that corresponds
    to the elastically-effective end-linked network in the graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph
        corresponding to the elastically-effective end-linked network of
        the given graph.
    
    """
    # Multiedge identification
    graph_edges, graph_edges_counts = edge_id(graph)

    # Acquire adjacency matrix with no multiedges
    if graph.is_multigraph():
        A = nx.to_numpy_array(nx.Graph(graph), dtype=int)
    else:
        A = nx.to_numpy_array(graph, dtype=int)
    
    # Apply the Yasuda-Morita procedure to return the
    # elastically-effective network that satisfies the Scanlan-Case
    # criteria.
    A, bridge_center_node_list, _, _ = yasuda_morita_procedure(A)

    # Add back bridging centers between surviving nodes in reverse
    # elimination order
    A = surviving_bridge_restoration(A, bridge_center_node_list)

    # Acquire NetworkX graph
    if graph.is_multigraph():
        graph = nx.from_numpy_array(A, create_using=nx.MultiGraph)
    else:
        graph = nx.from_numpy_array(A)

    # Restore multiedges
    graph = multiedge_restoration(graph, graph_edges, graph_edges_counts)

    # As a hard fail-safe, isolate and return the largest/maximum
    # connected component
    return largest_connected_component(graph)

def conn_edges_attr_to_edge_index_attr_arr(
        conn_edges_attr: np.ndarray) -> np.ndarray:
    """Edge index attributes array converted from an edge attributes
    array.

    This function generates an edge index attributes array converted
    from an edge attributes array. The edge index attributes array is
    formatted such that it properly complies with the PyTorch Geometric
    Data edge_index argument
    (torch_geometric.data.edge_index).

    Args:
        conn_edges_attr (np.ndarray): Edge attributes from the graph capturing the periodic connections between the core nodes.
    
    Returns:
        np.ndarray: Corresponding edge index attributes array.
    
    """
    m = np.shape(conn_edges_attr)[0]
    return (
        np.concatenate(
            (conn_edges_attr, conn_edges_attr))[np.arange(2*m).reshape(2, m).T.flatten()]
    )

def edge_index_attr_arr_to_conn_edges_attr(
        edge_index_attr_arr: np.ndarray) -> np.ndarray:
    """Edge attributes array converted from an edge index attributes
    array.

    This function generates an edge attributes array converted from an
    edge index attributes array. The edge index attributes array is
    formatted such that it properly complies with the PyTorch Geometric
    Data edge_index argument
    (torch_geometric.data.edge_index).

    Args:
        edge_index_attr_arr (np.ndarray): Edge index attributes array from the graph capturing the periodic connections between the core nodes.
    
    Returns:
        np.ndarray: Corresponding edge attributes array.
    
    """
    return edge_index_attr_arr[::2]

def conn_edges_to_edge_index_arr(conn_edges: np.ndarray) -> np.ndarray:
    """Edge index array converted from an edges array.

    This function generates an edge index array converted from an edges
    array. The edge index array is formatted such that it properly
    complies with the PyTorch Geometric Data edge_index argument
    (torch_geometric.data.edge_index).

    Args:
        conn_edges (np.ndarray): Edges from the graph capturing the periodic connections between the core nodes.
    
    Returns:
        np.ndarray: Corresponding edge index array.
    
    """
    m = np.shape(conn_edges)[0]
    return (
        np.transpose(
            np.vstack(
                (conn_edges, conn_edges[:, ::-1]))[np.arange(2*m).reshape(2, m).T.flatten()])
    )

def edge_index_arr_to_conn_edges(edge_index_arr: np.ndarray) -> np.ndarray:
    """Edges array converted from an edge index array.
    
    This function generates an edges array converted from an edge index
    array. The edge index array is formatted such that it properly
    complies with the PyTorch Geometric Data edge_index argument
    (torch_geometric.data.edge_index).

    Args:
        edge_index_arr (np.ndarray): Edge index array from the graph capturing the periodic connections between the core nodes.
    
    Returns:
        np.ndarray: Corresponding edges array.
    
    """
    return np.transpose(edge_index_arr)[::2]

def conn_edges_to_conn_sparse_A_arr(
        conn_edges: np.ndarray,
        conn_edges_type: np.ndarray,
        n: int,
        k_max: int) -> np.ndarray | None:
    """Structured sparse adjacency array converted from edges and edges
    type arrays.

    This function generates a structured sparse adjacency array
    converted from edges and edges type arrays. The structured sparse
    adjacency array is an n-by-k_max np.ndarray. In this array, the
    nodes are assumed to be numbered from 1 to n, and each node could
    possibly have a maximal degree of k_max. A positive, non-zero entry
    "val" in the ith row corresponds to the core edge (i+1, val). A
    negative, non-zero entry "-1*val" in the ith row corresponds to the
    periodic boundary edge (i+1, val). Note that here rows are
    zero-indexed, and referred to as such. To ensure adjacency array
    symmetry (in an adjacency matrix sense), the edge (val, i+1) is also
    appropriately stored in the structured sparse adjacency array. All
    zero entries corresponds the absence of an edge. This function
    generates a structured sparse adjacency array from two np.ndarrays:
    one representing the edges from the graph capturing the periodic
    connections between the core nodes, where nodes are zero-indexed,
    i.e,. are numbered from 0 to n-1; and the other representing the
    type label for each edge (core edges are of type 1, and periodic
    boundary edges are of type 2.).

    Args:
        conn_edges (np.ndarray): Edges from the graph capturing the periodic connections between the core nodes.
        conn_edges_type (np.ndarray): Type label for the edges from the graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 2.
        n (int): Number of nodes.
        k_max (int): Maximal degree of any given node.
    
    Returns:
        np.ndarray | None: Corresponding structured sparse adjacency
        array.
    
    """
    # Confirm that the nodes in the provided network are within the
    # specified number of nodes
    conn_edges_n = np.max(conn_edges)
    if conn_edges_n > n-1:
        error_str = (
            "The largest node number in the provided edge arrays is "
            + "less than the specified number of nodes."
        )
        raise ValueError(error_str)
    
    # Confirm that the provided network does not violate the maximal
    # node degree
    m = np.shape(conn_edges)[0]
    k = np.zeros(n, dtype=int)
    for edge in range(m):
        k[int(conn_edges[edge, 0])] += 1
        k[int(conn_edges[edge, 1])] += 1
    if np.max(k) > k_max:
        error_str = (
            "The maximal degree in the provided network is greater "
            + "than the specified maximal degree."
        )
        raise ValueError(error_str)
    
    # Initialize the structured sparse adjacency np.ndarray
    conn_sparse_A_arr = np.zeros((n, k_max), dtype=int)

    # Symmetrically enter each edge in the structured sparse adjacency
    # array
    for edge in range(m):
        # Node numbers
        core_node_0 = int(conn_edges[edge, 0])
        core_node_1 = int(conn_edges[edge, 1])
        # Edge type
        edge_type = conn_edges_type[edge]

        k_indx = np.where(conn_sparse_A_arr[core_node_0, :]==0)[0][0]
        conn_sparse_A_arr[core_node_0, k_indx] = (
            core_node_1 + 1 if edge_type else -1 * (core_node_1+1)
        )

        k_indx = np.where(conn_sparse_A_arr[core_node_1, :]==0)[0][0]
        conn_sparse_A_arr[core_node_1, k_indx] = (
            core_node_0 + 1 if edge_type else -1 * (core_node_0+1)
        )
    
    # Sort the values of each row of the structured sparse adjacency
    # array
    return np.sort(conn_sparse_A_arr, axis=1)

def conn_sparse_A_arr_to_conn_edges(
        conn_sparse_A_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edges and edges type arrays converted from a structured sparse
    adjacency array.

    This function generates the edges and edges type arrays converted
    from a structured sparse adjacency array. This function generates
    the edges array which represents the edges from the graph capturing
    the periodic connections between the core nodes, where nodes are
    zero-indexed, i.e,. are numbered from 0 to n-1. This function also
    generates the edges type array representing the type label for each
    edge (core edges are of type 1, and periodic boundary edges are of
    type 2.). These arrays are generated from the structured sparse
    adjacency array, an n-by-k_max np.ndarray. In this array, the nodes
    are assumed to be numbered from 1 to n, and each node could possibly
    have a maximal degree of k_max. A positive, non-zero entry "val" in
    the ith row corresponds to the core edge (i+1, val). A negative,
    non-zero entry "-1*val" in the ith row corresponds to the periodic
    boundary edge (i+1, val). Note that here rows are zero-indexed, and
    referred to as such. To ensure adjacency array symmetry (in an
    adjacency matrix sense), the edge (val, i+1) is also appropriately
    stored in the structured sparse adjacency array. All zero entries
    corresponds the absence of an edge.

    Args:
        conn_sparse_A_arr (np.ndarray): Structured sparse adjacency array representing the core and periodic boundary edges in the graph that captures the periodic connections between the core nodes.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Corresponding edges and edges
        type label arrays.
    
    """
    # Initialize lists for edges and edges type
    conn_edges = []
    conn_edges_type = []

    # Shape of the structured sparse adjacency array
    n, k_max = np.shape(conn_sparse_A_arr)

    # Interrogate each entry in the structured sparse adjacency array
    for core_node_0 in range(n):
        for k_indx in range(k_max):
            val = int(conn_sparse_A_arr[core_node_0, k_indx])

            # Continue if there is no edge
            if val == 0: continue
            else:
                # Identify the edge as either a core or a periodic
                # boundary edge, add the edge to the edges and edges
                # type lists, remove the edge, and remove its symmetric
                # equivalent in the structured sparse adjacency array
                conn_edges_type.append(1 if val > 0 else 0)
                core_node_1 = val - 1 if val > 0 else -1 * val - 1
                conn_edges.append((core_node_0, core_node_1))
                conn_sparse_A_arr[core_node_0, k_indx] = 0
                sym_val = core_node_0 + 1 if val > 0 else -1 * (core_node_0+1)
                sym_k_indx = (
                    np.where(conn_sparse_A_arr[core_node_1, :]==sym_val)[0][0]
                )
                conn_sparse_A_arr[core_node_1, sym_k_indx] = 0
    
    # Lexicographically sort the edges and edge types
    conn_edges, lexsort_indcs = lexsorted_edges(conn_edges, return_indcs=True)
    conn_edges_type = np.asarray(conn_edges_type, dtype=int)
    conn_edges_type = conn_edges_type[lexsort_indcs]

    return conn_edges, conn_edges_type