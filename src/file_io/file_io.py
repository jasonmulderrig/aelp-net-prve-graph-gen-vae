import os
import pathlib

def root_filepath_str(network: str) -> str:
    """Root filepath generator for periodic artificial end-linked
    polymer networks.

    This function ensures that a root baseline filepath for files
    involved with periodic artificial end-linked polymer networks exists
    as a directory, and then returns the root baseline filepath. The
    filepath must match the directory structure of the local computer.
    For Windows machines, the backslash must be represented as a double
    backslash. For Linux/Mac machines, the forwardslash can be directly
    represented as a forwardslash.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; either "auelp", "abelp", or "apelp" (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial bimodal end-linked polymer networks ("abelp"), or artificial polydisperse end-linked polymer networks ("apelp")).
    
    Returns:
        str: The root baseline filepath.
    
    """
    # For MacOS
    # root_filepath = f"/Users/jasonmulderrig/research/projects/aelp-net-prve-graph-gen-vae/{network}/"
    # For Windows OS
    root_filepath = f"C:\\Users\\mulderjp\\projects\\aelp-net-prve-graph-gen-vae\\{network}\\"
    # For Linux
    # root_filepath = f"/p/home/jpm2225/projects/aelp-net-prve-graph-gen-vae/{network}/"
    if os.path.isdir(root_filepath) == False:
        pathlib.Path(root_filepath).mkdir(parents=True, exist_ok=True)
    return root_filepath

def filepath_str(network: str) -> str:
    """Filepath generator for periodic artificial end-linked polymer
    networks.

    This function returns the baseline filepath for files involved with
    periodic artificial end-linked polymer networks.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; either "auelp", "abelp", or "apelp" (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial bimodal end-linked polymer networks ("abelp"), or artificial polydisperse end-linked polymer networks ("apelp")).
    
    Returns:
        str: The baseline filepath.
    
    """
    # For MacOS
    # filepath = f"/Users/jasonmulderrig/research/projects/aelp-net-prve-graph-gen-vae/{network}/raw/"
    # For Windows OS
    filepath = f"C:\\Users\\mulderjp\\projects\\aelp-net-prve-graph-gen-vae\\{network}\\raw\\"
    # For Linux
    # filepath = f"/p/home/jpm2225/projects/aelp-net-prve-graph-gen-vae/{network}/raw/"
    if os.path.isdir(filepath) == False:
        pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
    return filepath

def _filename_str(date: str, batch: str, sample: int) -> str:
    """Baseline filename string generator for periodic artificial
    end-linked polymer networks.

    This function returns the baseline filename string for files
    involved with periodic artificial end-linked polymer networks.

    Args:
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
    
    Returns:
        str: The baseline filename string.
    
    """
    return f"{date}{batch}{sample:d}"

def filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int) -> str:
    """Baseline filename generator for periodic artificial end-linked
    polymer networks.

    This function returns the baseline filename for files involved with
    periodic artificial end-linked polymer networks. The baseline
    filename is explicitly prefixed with the filepath to the directory
    that the files ought to be saved to (and loaded from for future
    use). This filepath is set by the user, and must match the directory
    structure of the local computer. The baseline filename is then
    appended to the filepath. It is incumbent on the user to save a data
    file that records the network parameter values that correspond to
    each network sample in the batch (i.e., a "lookup table").

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; either "auelp", "abelp", or "apelp" (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial bimodal end-linked polymer networks ("abelp"), or artificial polydisperse end-linked polymer networks ("apelp")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
    
    Returns:
        str: The baseline filename.
    
    """
    return filepath_str(network) + _filename_str(date, batch, sample)

def L_filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int) -> str:
    """Filename for simulation box side lengths.

    This function returns the filename for the simulation box side
    lengths.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; either "auelp", "abelp", or "apelp" (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial bimodal end-linked polymer networks ("abelp"), or artificial polydisperse end-linked polymer networks ("apelp")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
    
    Returns:
        str: The simulation box side lengths filename.
    
    """
    return filename_str(network, date, batch, sample) + "-L" + ".dat"

def _config_filename_str(
        date: str,
        batch: str,
        sample: int,
        config: int) -> str:
    """Configuration filename string.

    This function returns the configuration filename string.

    Args:
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
    
    Returns:
        str: The configuration filename string.
    
    """
    return _filename_str(date, batch, sample) + f"C{config:d}"

def config_filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int) -> str:
    """Configuration filename prefix.

    This function returns the configuration filename prefix.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; either "auelp", "abelp", or "apelp" (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial bimodal end-linked polymer networks ("abelp"), or artificial polydisperse end-linked polymer networks ("apelp")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
    
    Returns:
        str: The configuration filename prefix.
    
    """
    return (
        filepath_str(network)
        + _config_filename_str(date, batch, sample, config)
    )