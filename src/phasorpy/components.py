"""Component analysis of phasor coordinates.

The ``phasorpy.components`` module provides functions to:

- calculate fractions of two components of known location by projecting to
  line between components:

  - :py:func:`two_fractions_from_phasor`

- calculate phasor coordinates of second component if only one is
  known (not implemented)

- calculate fractions of three or four known components by using higher
  harmonic information (not implemented)

- calculate fractions of two or three components of known location by
  resolving graphically with histogram (not implemented)

- blindly resolve fractions of n components by using harmonic
  information (not implemented)

"""

from __future__ import annotations

__all__ = ['two_fractions_from_phasor']

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import math

import numpy

from ._utils import project_phasor_to_line
from phasorpy.phasor import phasor_from_signal


def two_fractions_from_phasor(
    real: ArrayLike,
    imag: ArrayLike,
    real_components: ArrayLike,
    imag_components: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return fractions of two components from phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    real_components: array_like
        Real coordinates of the first and second components.
    imag_components: array_like
        Imaginary coordinates of the first and second components.

    Returns
    -------
    fraction_of_first_component : ndarray
        Fractions of the first component.
    fraction_of_second_component : ndarray
        Fractions of the second component.

    Notes
    -----
    For the moment, calculation of fraction of components from different
    channels or frequencies is not supported. Only one pair of components can
    be analyzed and will be broadcasted to all channels/frequencies.

    Raises
    ------
    ValueError
        If the real and/or imaginary coordinates of the known components are
        not of size 2.
        If the two components have the same coordinates.

    Examples
    --------
    >>> two_fractions_from_phasor(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.9], [0.4, 0.3]
    ... ) # doctest: +NUMBER
    (array([0.44, 0.56, 0.68]), array([0.56, 0.44, 0.32]))

    """
    real_components = numpy.asarray(real_components)
    imag_components = numpy.asarray(imag_components)
    if real_components.shape != (2,):
        raise ValueError(f'{real_components.shape=} != (2,)')
    if imag_components.shape != (2,):
        raise ValueError(f'{imag_components.shape=} != (2,)')
    first_component_phasor = numpy.array(
        [real_components[0], imag_components[0]]
    )
    second_component_phasor = numpy.array(
        [real_components[1], imag_components[1]]
    )
    total_distance_between_components = math.hypot(
        (second_component_phasor[0] - first_component_phasor[0]),
        (second_component_phasor[1] - first_component_phasor[1]),
    )
    if math.isclose(total_distance_between_components, 0, abs_tol=1e-6):
        raise ValueError('components must have different coordinates')
    projected_real, projected_imag = project_phasor_to_line(
        real, imag, real_components, imag_components
    )
    distances_to_first_component = numpy.hypot(
        numpy.asarray(projected_real) - first_component_phasor[0],
        numpy.asarray(projected_imag) - first_component_phasor[1],
    )
    fraction_of_second_component = (
        distances_to_first_component / total_distance_between_components
    )
    return 1 - fraction_of_second_component, fraction_of_second_component


def multi_component_unmixing_from_phasor(
        multi_harmonic_real: ArrayLike,
        multi_harmonic_imag: ArrayLike,
        matrixA: ArrayLike, 
        /,
) -> tuple[NDArray[Any]]:
    """ Return fractions in each pixel from multiple components.

    Parametres
    ----------
    multi_harmonic_real : array_like
        Real components of the phasor coordinate for many harmonics.
    multi_harmonic_imag : array_like
        Imaginary components of the phasor coordinate for many harmonics.
    matrixA : array_like 
        Coefficiency matrix for each components. 

    Returns
    -------
    fractions : ndarray
        Fractions of each components. 

    Raises
    ------
    ValueError
        If multi_harmonic_real and multi_harmonic_imag have different shape.
        If coefficiency matrix is empty.
    
    Examples
    --------
    >>> multi_component_unmixing_from_phasor(
    ...     multi_harmonic_real=[-0.4129963 , -0.07156281], 
    ...     multi_harmonic_imag=[ 0.44520192, -0.17987708],
    ...     matrixA=[[-0.62442218, -0.47591901, -0.15022351, -0.5899417 ],
    ...     [ 0.10858431, -0.0977382 , -0.24658903,  0.02135662],
    ...     [ 0.21045545,  0.53110438,  0.61459059,  0.41559306],
    ...     [-0.19656654, -0.3379749 , -0.06848862, -0.29857087],
    ...     [ 1.        ,  1.        ,  1.        ,  1.        ]]
    ... ) # doctest: +NUMBER
    (array([0.3, 0.1, 0.4, 0.2]))

    >>> multi_harmonic_real =
    ...     [[[-0.39595987, -0.08619137],
    ...     [-0.39595987, -0.08619137],
    ...     [-0.39595987, -0.08619137]],
    ...     [[-0.39595987, -0.08619137],
    ...     [-0.39595987, -0.08619137],
    ...     [-0.39595987, -0.08619137]],
    ...     [[-0.39595987, -0.08619137],
    ...     [-0.39595987, -0.08619137],
    ...     [-0.39595987, -0.08619137]]]
        
    >>> multi_harmonic_imag = 
    ...     [[[ 0.45231211, -0.16960909],
    ...     [ 0.45231211, -0.16960909],
    ...     [ 0.45231211, -0.16960909]],
    ...     [[ 0.45231211, -0.16960909],
    ...     [ 0.45231211, -0.16960909],
    ...     [ 0.45231211, -0.16960909]],
    ...     [[ 0.45231211, -0.16960909],
    ...     [ 0.45231211, -0.16960909],
    ...     [ 0.45231211, -0.16960909]]]
    >>> matrixA=[[-0.62442218, -0.47591901, -0.15022351, -0.5899417 ],
    ...     [ 0.10858431, -0.0977382 , -0.24658903,  0.02135662],
    ...     [ 0.21045545,  0.53110438,  0.61459059,  0.41559306],
    ...     [-0.19656654, -0.3379749 , -0.06848862, -0.29857087],
    ...     [ 1.        ,  1.        ,  1.        ,  1.        ]]
        
    >>> multi_component_unmixing_from_phasor(
    ...     multi_harmonic_real, 
    ...     multi_harmonic_imag,
    ...     matrixA) # doctest: +NUMBER
    array([[[0.32982308, 0.11319236, 0.41102329, 0.14604279],
    ...         [0.32982308, 0.11319236, 0.41102329, 0.14604279],
    ...         [0.32982308, 0.11319236, 0.41102329, 0.14604279]],
    ...        [[0.32982308, 0.11319236, 0.41102329, 0.14604279],
    ...         [0.32982308, 0.11319236, 0.41102329, 0.14604279],
    ...         [0.32982308, 0.11319236, 0.41102329, 0.14604279]],
    ...        [[0.32982308, 0.11319236, 0.41102329, 0.14604279],
    ...         [0.32982308, 0.11319236, 0.41102329, 0.14604279],
    ...         [0.32982308, 0.11319236, 0.41102329, 0.14604279]]])
    """
    multi_harmonic_real = numpy.asarray(multi_harmonic_real)
    multi_harmonic_imag = numpy.asarray(multi_harmonic_imag)
    matrixA = numpy.asarray(matrixA)

    if multi_harmonic_real.shape != multi_harmonic_imag.shape:
        raise ValueError("multi_harmonic_real and multi_harmonic_imag"
                         "have different shape")
    if matrixA.size == 0:
        raise ValueError("matrixA is empty")
    
    ncomp = matrixA.shape[0] - 1
    nh = math.floor(ncomp / 2)
    if len(multi_harmonic_real.shape) == 1:
        vecB = [multi_harmonic_real[j] for j in range(nh)] \
            + [multi_harmonic_imag[j] for j in range(nh)] + [1]
        return numpy.linalg.lstsq(matrixA, vecB, rcond=None)[0]
    else:
        fractions = numpy.zeros([multi_harmonic_real.shape[0], 
                                 multi_harmonic_imag.shape[1], ncomp])
        for r in range(multi_harmonic_real.shape[0]):
            for c in range(multi_harmonic_real.shape[1]):
                vecB = [multi_harmonic_real[r, c, j] for j in range(nh)] \
                    + [multi_harmonic_imag[r, c, j] for j in range(nh)] + [1]
                fractions[r, c] = numpy.linalg.lstsq(matrixA, vecB, 
                                                     rcond=None)[0]
        return fractions


def multi_harmonic_phasor(
        signal: ArrayLike,
        /,
        *,
        harmonic: int=2,
) -> tuple[NDArray[Any]]:
    """Return the real and imag part of the phasor tranform for many harmonics 

    Parametres
    ----------
    signal : array_like
        Frequency-domain, time-domain, or hyperspectral data.
    harmonic : int 
        Harmonics to return. Must be >= 2.
    
    Returns
    -------
    multi_harmonic_real: ndarray
        Array with real coordinate of phasor for two or more 
        harmonics.
    multi_harmonic_imag: ndarray
        Array with imaginary coordinate of phasor for two or more
        harmonics.

    Raises
    ------
    ValueErro
        signal input is empty
        harmonics must be at leat 2

    Examples
    --------
    >>> signal = array([ 242,  459,  711, 1169, 1667, 2450, 3187, 4107,
    ...                     4857, 6059, 6687,7402, 7589, 7344, 7187,
    ...                     6648, 5850, 4902, 4210, 3375, 2681, 2166,
    ...                     1829, 1491, 1268,  979,  786,  651,  497,
    ...                     413,  359,  265,  209, 176,  128])
    >>> multi_harmonic_real, multi_harmonic_imag = multi_harmonic_phasor(
    ...                                                        signal, 2)
    (array([-0.4079422 , -0.07375106]), array([ 0.44282093, -0.17471873]))
    """
    if signal.size == 0:
        raise ValueError("signal input is empty")
    if harmonic <= 1:
        raise ValueError("harmonics must be at leat 2")
    
    if len(signal.shape) == 1:
        multi_hamonic_real = numpy.empty([harmonic])
        multi_hamonic_imag = numpy.empty([harmonic])
        for i in range(harmonic):
            _, real, imag = phasor_from_signal(signal, harmonic=i + 1)
            multi_hamonic_real[i] = real
            multi_hamonic_imag[i] = imag
    else:
        multi_hamonic_real = numpy.empty([signal.shape[0], signal.shape[1], harmonic])
        multi_hamonic_imag = numpy.empty([signal.shape[0], signal.shape[1], harmonic])
        for i in range(harmonic):
            _, real, imag = phasor_from_signal(signal, harmonic=i + 1)
            multi_hamonic_real[:, :, i] = real
            multi_hamonic_imag[:, :, i] = imag
    return multi_hamonic_real, multi_hamonic_imag

