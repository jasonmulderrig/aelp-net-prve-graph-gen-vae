######################################################################################################################################
graph-generative variational autoencoder analysis on periodic representative volume elements of artificial end-linked polymer networks
######################################################################################################################################

A repository of research codes that utilize graph-generative variational autoencoders to learn and faithfully generate the microstructural topology of periodic representative volume elements of artificial end-linked polymer networks. Functionality is provided in this repository to computationally synthesize and analyze periodic representative volume elements of artificial end-linked polymer networks. The networks of focus include (in alphabetical order): artificial bimodal end-linked polymer networks, artificial polydisperse end-linked polymer networks, and artificial uniform end-linked polymer networks. The base code that generates these networks is also provided in the `aelp-net-prve <https://github.com/jasonmulderrig/aelp-net-prve>`_ GitHub repository.

*****
Setup
*****

Once the contents of the repository have been cloned or downloaded, the Conda environment associated with the project needs to be installed. The installation of this Conda environment and some essential packages is outlined in the ``conda-environment-install.txt`` text file. All required packages are listed in the ``requirements.txt`` file.

*********
Structure
*********

The core functions in this repository are modularly distributed in Python files that reside in the following source directories:

* ``src/descriptors``
* ``src/file_io``
* ``src/helpers``
* ``src/models``
* ``src/networks``

The core functions can then be called upon in Python files (or Jupyter notebooks) for various tasks. The following directories contain various Python files that synthesize and analyze the aforementioned types of artificial end-linked polymer networks:

* ``abelp``
* ``apelp``
* ``auelp``

Importantly, configuration settings for each of these networks are stored in an appropriately named sub-directory within the ``configs/networks`` directory (``configs/networks/abelp``, ``configs/networks/apelp``, ``configs/networks/auelp``). Each of these sub-directories contain a YAML file defining a wide variety of parameter configuration settings. Moreover, each of these sub-directories contain two more sub-directories, ``topology`` and ``descriptors``. Within each of these sub-directories are YAML files that define parameter configuration settings specifically related to network topology and descriptor calculations, respectively. The Hydra package is employed to load in the settings from the YAML files.

The following Python files synthesize various artificial end-linked polymer networks, calculate descriptors, and consolidate the resulting data when run in the order provided:

* In the ``abelp`` directory: ``abelp_networks_topology_synthesis.py`` -> ``abelp_networks_topology_augmentation.py`` -> ``abelp_networks_topology_descriptors.py`` -> ``abelp_networks_topology_consolidation.py``
* In the ``apelp`` directory: ``apelp_networks_topology_synthesis.py`` -> ``apelp_networks_topology_augmentation.py`` -> ``apelp_networks_topology_descriptors.py`` -> ``apelp_networks_topology_consolidation.py``
* In the ``auelp`` directory: ``auelp_networks_topology_synthesis.py`` -> ``auelp_networks_topology_augmentation.py`` -> ``auelp_networks_topology_descriptors.py`` -> ``auelp_networks_topology_consolidation.py``

The functionality of the synthesis-augmentation-descriptors-consolidation files is combined into a stand-alone ``one-shot data creation`` file:

* In the ``abelp`` directory: ``abelp_networks_topology_one_shot_data_creation.py``
* In the ``apelp`` directory: ``apelp_networks_topology_one_shot_data_creation.py``
* In the ``auelp`` directory: ``auelp_networks_topology_one_shot_data_creation.py``

In addition, several Python files are supplied in the ``abelp``, ``apelp``, and ``auelp`` directories that plot the spatially-embedded structure of a given network and analyze the statistics of various network topological features.

*****
Usage
*****

**Before running any of the code, it is required that the user verify the baseline filepath in the ``filepath_str()`` function of the ``file_io.py`` Python file in the ``file_io`` directory. Note that filepath string conventions are operating system-sensitive.**

*************************
Example timing benchmarks
*************************

Timing benchmarks are comparable to those provided in the `aelp-net-prve <https://github.com/jasonmulderrig/aelp-net-prve>`_ GitHub repository.