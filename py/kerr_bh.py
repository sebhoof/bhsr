######################
#  Kerr BH functions #
######################

import numpy as np

from .constants import GNewton

def rg(mbh: float) -> float:
    """
    Calculate the "gravitational radius" of a Kerr black hole.

    Parameters:
        mbh (float): The mass of the black hole in Msol.

    Returns:
        float: The gravitational radius of the black hole in eV^-1.
    """
    return GNewton*mbh

def alpha(ma: float, mbh: float) -> float:
    """
    Compute the dimensionless coupling constant alpha.

    Parameters:
        ma (float): The mass of the ultralight boson in eV.
        mbh (float): The mass of the black hole in Msol.

    Returns:
        float: The dimensionless coupling constant alpha.
    """
    return rg(mbh)*ma

def r_plus(mbh: float, astar: float) -> float:
    """
    Calculates the radius of the inner event horizon of a Kerr black hole.

    Parameters:
    mbh (float): The mass of the black hole in Msol.
    astar (float): The dimensionless spin parameter of the black hole.

    Returns:
    float: The radius of the inner event horizon in eV^-1.
    """
    return rg(mbh)*(1 + np.sqrt(1 - astar*astar))

def r_minus(mbh: float, astar: float) -> float:
    """
    Calculates the radius of the outer event horizon of a Kerr black hole.

    Parameters:
    mbh (float): The mass of the black hole in Msol.
    astar (float): The dimensionless spin parameter of the black hole.

    Returns:
    float: The radius of the outer event horizon in eV^-1.
    """
    return rg(mbh)*(1 - np.sqrt(1 - astar*astar))

def omH(mbh: float, astar: float) -> float:
    """
    Calculate the angular velocity of the event horizon of a Kerr black hole.

    Parameters:
    mbh (float): The mass of the black hole in Msol.
    astar (float): The dimensionless spin parameter of the black hole.

    Returns:
    float: The angular velocity of the event horizon in eV.
    """
    return 0.5*astar/r_plus(mbh, astar)