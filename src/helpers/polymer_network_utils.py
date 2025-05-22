def m_arg_stoich_func(n: float, k: float) -> float:
    """Number of chains.

    This function calculates the number of chains, given the number of
    cross-linkers and the maximum cross-linker degree/functionality.
    This calculation assumes a stoichiometric mixture of cross-linkers
    and chains.

    Args:
        n (float): Number of cross-linkers.
        k (float): Maximum cross-linker degree/functionality.

    Returns:
        float: Number of chains.
    """
    return n * k / 2

def m_arg_en_func(n_en: float, en: float) -> float:
    """Number of chains.

    This function calculates the number of chains, given the number of
    chain segment particles and the (average) number of segment
    particles per chain.

    Args:
        n_en (float): Number of chain segment particles.
        en (float): (Average) Number of segment particles per chain.

    Returns:
        float: Number of chains.
    """
    return n_en / en

def en_arg_m_func(n_en: float, m: float) -> float:
    """(Average) Number of segment particles per chain

    This function calculates the (average) number of segment particles
    per chain, given the number of chain segment particles and the
    number of chains.

    Args:
        n_en (float): Number of chain segment particles.
        m (float): Number of chains.

    Returns:
        float: (Average) Number of segments per chain.
    """
    return n_en / m

def n_arg_stoich_func(m: float, k: float) -> float:
    """Number of cross-linkers.

    This function calculates the number of cross-linkers, given the
    number of chains and the maximum cross-linker degree/functionality.
    This calculation assumes a stoichiometric mixture of cross-linkers
    and chains.

    Args:
        m (float): Number of chains.
        k (float): Maximum cross-linker degree/functionality.

    Returns:
        float: Number of chains.
    """
    return 2 * m / k

def n_en_arg_m_func(m: float, en: float) -> float:
    """Number of chain segment particles.

    This function calculates the number of chain segment particles,
    given the number of chains and the (average) number of segment
    particles per chain.

    Args:
        m (float): Number of chains.
        en (float): (Average) Number of segment particles per chain.

    Returns:
        float: Number of chain segment particles.
    """
    return m * en

def n_arg_n_tot_func(n_tot: float, n_other: float) -> float:
    """Number of particles (chain segment particles or cross-linkers).

    This function calculates the number of particles (chain segment
    particles or cross-linkers), given the number of constituents and
    the number of the other type of particle (cross-linkers or chain
    segment particles, respectively).

    Args:
        n_tot (float): Number of constituents.
        n_other (float): Number of the other type of particles (cross-linkers or chain segment particles).

    Returns:
        float: Number of particles (chain segment particles or
        cross-linkers).
    """
    return n_tot - n_other

def n_arg_f_func(f: float, n_tot: float) -> float:
    """Number of particles (chain segment particles or cross-linkers).

    This function calculates the number of particles (chain segment
    particles or cross-linkers), given the particle number fraction and
    the number of constituents.

    Args:
        f (float): Particle (chain segment or cross-linker) number fraction.
        n_tot (float): Number of constituents.

    Returns:
        float: Number of particles (chain segment particles or cross-linkers).
    """
    return f * n_tot

def n_tot_arg_n_func(n_en: float, n: float) -> float:
    """Number of constituents.

    This function calculates the number of constituents,
    given the number of chain segment particles and the number of
    cross-linkers.

    Args:
        n_en (float): Number of chain segment particles.
        n (float): Number of cross-linkers.

    Returns:
        float: Number of constituents.
    """
    return n_en + n

def n_tot_arg_f_func(n: float, f: float) -> float:
    """Number of constituents.
    
    This function calculates the number of constituents, given the
    number of particles (chain segment particles or cross-linkers) and
    its number fraction.

    Args:
        n (float): Number of particles (chain segments or cross-linkers)
        f (float): Particle (chain segment or cross-linker) number fraction.

    Returns:
        float: Number of constituents.
    """
    return n / f

def f_arg_n_func(n: float, n_tot: float) -> float:
    """Particle (chain segment or cross-linker) number fraction.

    This function calculates the particle (chain segment or
    cross-linker) number fraction, given the number of particles (chain
    segment particless or cross-linkers) and the number of constituents.

    Args:
        n (float): Number of particles (chain segment particles or
        cross-linkers)
        n_tot (float): Number of constituents.

    Returns:
        float: Particle (chain segment or cross-linker) number fraction.
    """
    return n / n_tot

def f_arg_f_func(f_other: float) -> float:
    """Particle (chain segment or cross-linker) number fraction.

    This function calculates the particle (chain segment or
    cross-linker) number fraction, given the other particle
    (cross-linker or chain segment, respectively) number fraction.

    Args:
        f_other (float): Other particle (cross-linker or chain segment, respectively) number fraction.

    Returns:
        float: Particle (chain segment or cross-linker) number fraction.
    """
    return 1 - f_other