from src.networks.aelp_networks import (
    aelp_L,
    aelp_network_additional_node_seeding,
    aelp_network_additional_nodes_type,
    aelp_network_hilbert_node_label_assignment,
    aelp_network_local_topological_descriptor,
    aelp_network_global_topological_descriptor,
    aelp_network_global_morphological_descriptor
)
from src.networks.auelp_networks import auelp_network_topology
from src.networks.abelp_networks import abelp_network_topology
from src.networks.apelp_networks import apelp_network_topology
from src.helpers.node_placement_utils import (
    initial_node_seeding,
    additional_node_seeding
)

def run_aelp_L(args):
    aelp_L(*args)

def run_initial_node_seeding(args):
    initial_node_seeding(*args)

def run_additional_node_seeding(args):
    additional_node_seeding(*args)

def run_auelp_network_topology(args):
    auelp_network_topology(*args)

def run_abelp_network_topology(args):
    abelp_network_topology(*args)

def run_apelp_network_topology(args):
    apelp_network_topology(*args)

def run_aelp_network_additional_node_seeding(args):
    aelp_network_additional_node_seeding(*args)

def run_aelp_network_additional_nodes_type(args):
    aelp_network_additional_nodes_type(*args)

def run_aelp_network_hilbert_node_label_assignment(args):
    aelp_network_hilbert_node_label_assignment(*args)

def run_aelp_network_local_topological_descriptor(args):
    aelp_network_local_topological_descriptor(*args)

def run_aelp_network_global_topological_descriptor(args):
    aelp_network_global_topological_descriptor(*args)

def run_aelp_network_global_morphological_descriptor(args):
    aelp_network_global_morphological_descriptor(*args)