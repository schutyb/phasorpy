"""
Component analysis
==================

An introduction to component analysis in the phasor space.

"""

# %%
# Import required modules, functions, and classes:

import matplotlib.pyplot as plt
import numpy
import math 

from phasorpy.components import *
from phasorpy.phasor import phasor_from_lifetime, phasor_from_signal
from phasorpy.plot import PhasorPlot

part1 = False
if part1:
    # %%
    # Fractions of combination of two components
    # ------------------------------------------
    #
    # The phasor coordinate of a combination of two lifetime components lie on
    # the line between the two components. For example, a combination with 25%
    # contribution of a component with lifetime 8.0 ns and 75% contribution of
    # a second component with lifetime 1.0 ns at 80 MHz:

    frequency = 80.0
    components_lifetimes = [8.0, 1.0]
    component_fractions = [0.25, 0.75]
    real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, component_fractions
    )
    components_real, components_imag = phasor_from_lifetime(
        frequency, components_lifetimes
    )
    plot = PhasorPlot(frequency=frequency, title='Combination of two components')
    plot.plot(components_real, components_imag, fmt='o-')
    plot.plot(real, imag)
    plot.show()

    # %%
    # If the location of both components is known, their contributions
    # to the phasor point that lies on the line between the components
    # can be calculated:

    (
        fraction_of_first_component,
        fraction_of_second_component,
    ) = two_fractions_from_phasor(real, imag, components_real, components_imag)
    print(f'Fraction of first component:  {fraction_of_first_component:.3f}')
    print(f'Fraction of second component: {fraction_of_second_component:.3f}')

    # %%
    # Contribution of two known components in multiple phasors
    # --------------------------------------------------------
    #
    # Phasors can have different contributions of two components with known
    # phasor coordinates:

    real, imag = numpy.random.multivariate_normal(
        (0.6, 0.35), [[8e-3, 1e-3], [1e-3, 1e-3]], (100, 100)
    ).T
    plot = PhasorPlot(
        frequency=frequency,
        title='Phasor with contribution of two known components',
    )
    plot.hist2d(real, imag, cmap='plasma')
    plot.plot(*phasor_from_lifetime(frequency, components_lifetimes), fmt='o-')
    plot.show()

    # %%
    # If the phasor coordinates of two components contributing to multiple
    # phasors are known, their fractional contributions to each phasor coordinate
    # can be calculated and plotted as histograms:

    (
        fraction_from_first_component,
        fraction_from_second_component,
    ) = two_fractions_from_phasor(real, imag, components_real, components_imag)
    fig, ax = plt.subplots()
    ax.hist(
        fraction_from_first_component.flatten(),
        range=(0, 1),
        bins=100,
        alpha=0.75,
        label='First',
    )
    ax.hist(
        fraction_from_second_component.flatten(),
        range=(0, 1),
        bins=100,
        alpha=0.75,
        label='Second',
    )
    ax.set_title('Histograms of fractions of first and second component')
    ax.set_xlabel('Fraction')
    ax.set_ylabel('Counts')
    ax.legend()
    plt.tight_layout()
    plt.show()

# %%
# sphinx_gallery_thumbnail_number = 2

# %%
from scipy.stats import gaussian_kde
def photon_distribution(D, number_photons):
    """
    Distribuye un número específico de fotones en función de una distribución 
    de densidad de probabilidad.

    Parameters
    ----------
    D : ndarray
        Curva espectral que se toma como 'modelo' para distribuir una 
        determinada cantidad de fotones.
    number_photons : int
        Cantidad de fotones que se distribuiran de acuerdo con la curva de
        densidad de probabilidad 'D' .

    Returns
    -------
    photon_distribution : ndarray
        Distribución de fotones generada a lo largo del rango de longitudes de
        onda disponibles considerando un paso h = 1000 .

    """

    wavelengths = numpy.arange(len(D))

    # Estimación de densidad de kernel
    kde = gaussian_kde(wavelengths, weights=D)

    # Valores para evaluar la distribución suavizada
    x_values = numpy.linspace(min(wavelengths), max(wavelengths), 1000)

    # Evaluar la densidad de kernel suavizada en los valores deseados
    smoothed_data = kde(x_values)

    # Generar number_photons fotones en función de la distribución suavizada
    if isinstance(number_photons, int):
        photon_distribution = numpy.random.choice(
            x_values, size=number_photons, p=smoothed_data / numpy.sum(smoothed_data)
        )
    else:
        raise ValueError(
            "The value of number_photons must be an integer greater than or equal to 1"
        )

    return photon_distribution


single_spectrums = True
if single_spectrums:
    import pandas
    df = pandas.read_csv("/Users/schutyb/Documents/GitHub/phasorpy/tutorials/datacomp/pure_spectrums.csv")

    c1 = photon_distribution(df["alexa"], 30000)
    c2 = photon_distribution(df["atto"], 10000)
    c3 = photon_distribution(df["cerulean"], 40000)
    c4 = photon_distribution(df["mito_orange"], 20000)

    comp5 = c1 + c2 + c3 + c4

    _, real1, imag1 = phasor_from_signal(df["alexa"])
    _, real2, imag2 = phasor_from_signal(df["atto"])
    _, real3, imag3 = phasor_from_signal(df["cerulean"])
    _, real4, imag4 = phasor_from_signal(df["mito_orange"])

    plt.figure()
    plt.plot(df["lambda"], df["alexa"], "r", label="alexa594")
    plt.plot(df["lambda"], df["atto"], "g", label="atto488")
    plt.plot(df["lambda"], df["cerulean"], "b", label="cerulean")
    plt.plot(df["lambda"], df["mito_orange"], "orange", label="mitotracker_orange")
    plt.legend()

    plt.show()


experimental_case = False
if experimental_case:
    # Read the experimental image
    image = numpy.load("/Users/schutyb/Documents/GitHub/phasorpy/tutorials/datacomp/experimental_image.npy")

    import tifffile

    im = tifffile.imread("/Users/schutyb/Downloads/PhasorSpectralUnmixing_Figshare/Data/examples5comps/3_todo_2.lsm")
    dc, real, imag = phasor_from_signal(im, axis=0)

    plt.figure()
    plt.imshow(dc)

    plot = PhasorPlot(allquadrants=True, title='Components and Phasor')
    plot.hist2d(real, imag, cmap='Blues')
    plt.show()


