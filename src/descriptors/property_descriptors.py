import numpy as np
import networkx as nx
from src.descriptors.general_topological_descriptors import (
    gamma_func,
    gamma_inv_func,
    prop_ee_m_func,
    prop_eeel_m_func
)
from src.descriptors.nodal_degree_topological_descriptors import k_func
from src.helpers.graph_utils import (
    elastically_effective_graph,
    elastically_effective_end_linked_graph
)

def ee_glbl_mean_gamma_func(
        ee_conn_edges: np.ndarray,
        ee_conn_edges_type: np.ndarray,
        l_cntr_ee_conn_edges: np.ndarray,
        coords: np.ndarray,
        L: np.ndarray) -> float:
    """Elastically-effective global mean chain/edge stretch.

    This function calculates the mean chain/edge stretch from a supplied
    elastically-effective graph.

    Args:
        ee_conn_edges (np.ndarray): Edges from the elastically-effective graph capturing the periodic connections between the core nodes.
        ee_conn_edges_type (np.ndarray): Type label for the edges from the elastically-effective graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 2.
        l_cntr_ee_conn_edges (np.ndarray): Contour length of the edges from the elastically-effective graph capturing the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (np.ndarray): Tessellation scaling (i.e., simulation box side lengths).
    
    Returns:
        float: Elastically-effective global mean chain/edge stretch.
    
    """
    return (
        np.mean(
            gamma_func(
                ee_conn_edges, ee_conn_edges_type, l_cntr_ee_conn_edges, 
                coords, L))
    )

def eeel_glbl_mean_gamma_func(
        eeel_conn_edges: np.ndarray,
        eeel_conn_edges_type: np.ndarray,
        l_cntr_eeel_conn_edges: np.ndarray,
        coords: np.ndarray,
        L: np.ndarray) -> float:
    """Elastically-effective end-linked global mean chain/edge stretch.

    This function calculates the mean chain/edge stretch from a supplied
    elastically-effective end-linked graph.

    Args:
        eeel_conn_edges (np.ndarray): Edges from the elastically-effective end-linked graph capturing the periodic connections between the core nodes.
        eeel_conn_edges_type (np.ndarray): Type label for the edges from the elastically-effective end-linked graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 2.
        l_cntr_eeel_conn_edges (np.ndarray): Contour length of the edges from the elastically-effective end-linked graph capturing the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (np.ndarray): Tessellation scaling (i.e., simulation box side lengths).
    
    Returns:
        float: Elastically-effective end-linked global mean chain/edge
        stretch.
    
    """
    return (
        np.mean(
            gamma_func(
                eeel_conn_edges, eeel_conn_edges_type, l_cntr_eeel_conn_edges, 
                coords, L))
    )

def ee_glbl_mean_gamma_inv_func(
        ee_conn_edges: np.ndarray,
        ee_conn_edges_type: np.ndarray,
        l_cntr_ee_conn_edges: np.ndarray,
        coords: np.ndarray,
        L: np.ndarray) -> float:
    """Elastically-effective global mean inverse chain/edge stretch.

    This function calculates the mean inverse chain/edge stretch from a
    supplied elastically-effective graph.

    Args:
        ee_conn_edges (np.ndarray): Edges from the elastically-effective graph capturing the periodic connections between the core nodes.
        ee_conn_edges_type (np.ndarray): Type label for the edges from the elastically-effective graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 2.
        l_cntr_ee_conn_edges (np.ndarray): Contour length of the edges from the elastically-effective graph capturing the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (np.ndarray): Tessellation scaling (i.e., simulation box side lengths).
    
    Returns:
        float: Elastically-effective global mean inverse chain/edge
        stretch.
    
    """
    return (
        np.mean(
            gamma_inv_func(
                ee_conn_edges, ee_conn_edges_type, l_cntr_ee_conn_edges, 
                coords, L))
    )

def eeel_glbl_mean_gamma_inv_func(
        eeel_conn_edges: np.ndarray,
        eeel_conn_edges_type: np.ndarray,
        l_cntr_eeel_conn_edges: np.ndarray,
        coords: np.ndarray,
        L: np.ndarray) -> float:
    """Elastically-effective end-linked global mean inverse chain/edge
    stretch.

    This function calculates the mean inverse chain/edge stretch from a
    supplied elastically-effective end-linked graph.

    Args:
        eeel_conn_edges (np.ndarray): Edges from the elastically-effective end-linked graph capturing the periodic connections between the core nodes.
        eeel_conn_edges_type (np.ndarray): Type label for the edges from the elastically-effective end-linked graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 2.
        l_cntr_eeel_conn_edges (np.ndarray): Contour length of the edges from the elastically-effective end-linked graph capturing the periodic connections between the core nodes.
        coords (np.ndarray): Coordinates of the core nodes.
        L (np.ndarray): Tessellation scaling (i.e., simulation box side lengths).
    
    Returns:
        float: Elastically-effective end-linked global mean inverse
        chain/edge stretch.
    
    """
    return (
        np.mean(
            gamma_inv_func(
                eeel_conn_edges, eeel_conn_edges_type, l_cntr_eeel_conn_edges, 
                coords, L))
    )

def dobrynin_kappa_func(k_mean: float, prop_m: float) -> float:
    """Quality factor.

    This function calculates the quality factor (as defined in Dobrynin
    et al., Macromolecules, 2023) of a polymer network topology.

    Args:
        k_mean (float): Average elastically-effective or elastically-effective end-linked cross-link degree.
        prop_m (float): Proportion of chains that are elastically-effective or elastically-effective end-linked.
    
    Returns:
        float: Quality factor.
    
    """
    C_loop = 0.4 * k_mean / (k_mean-2.)
    return (1.-2./k_mean) * C_loop * prop_m

def ee_dobrynin_kappa_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Elastically-effective quality factor.

    This function calculates the quality factor (as defined in Dobrynin
    et al., Macromolecules, 2023) of a given graph with respect to its
    elastically-effective portion.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        float: Quality factor.
    
    """
    return (
        dobrynin_kappa_func(
            np.mean(k_func(elastically_effective_graph(graph))),
            prop_ee_m_func(graph))
    )

def eeel_dobrynin_kappa_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Elastically-effective end-linked quality factor.

    This function calculates the quality factor (as defined in Dobrynin
    et al., Macromolecules, 2023) of a given graph with respect to its
    elastically-effective end-linked portion.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        float: Quality factor.
    
    """
    return (
        dobrynin_kappa_func(
            np.mean(k_func(elastically_effective_end_linked_graph(graph))),
            prop_eeel_m_func(graph))
    )