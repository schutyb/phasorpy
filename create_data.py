# %%
# sphinx_gallery_thumbnail_number = 2
# %%
import matplotlib.pyplot as plt
import numpy
import math 
from scipy.stats import gaussian_kde

from phasorpy.phasor import phasor_from_signal
from phasorpy.plot import PhasorPlot
from phasorpy.components import multi_harmonic_phasor, multi_component_unmixing_from_phasor

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
        densidad de probabilidad 'D'.

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


def expandir_con_ceros(signal, n_zeros_left, n_zeros_right):
    """
    Expande una señal agregando ceros a la izquierda y a la derecha.

    Args:
    signal (numpy array): La señal original.
    n_zeros_left (int): Número de ceros a agregar a la izquierda.
    n_zeros_right (int): Número de ceros a agregar a la derecha.

    Returns:
    numpy array: La señal expandida con ceros.
    """
    # Agregar ceros a la izquierda
    left_zeros = numpy.zeros(n_zeros_left)
    # Agregar ceros a la derecha
    right_zeros = numpy.zeros(n_zeros_right)
    # Concatenar con ceros a la izquierda y a la derecha
    expanded_signal = numpy.concatenate((left_zeros, signal, right_zeros))
    return expanded_signal


single_spectrums = False
if single_spectrums:
    import pandas
    df = pandas.read_csv("/Users/schutyb/Documents/GitHub/phasorpy/tutorials/datacomp/pure_spectrums.csv")

    # distribuye 1M fotones de la sigueinte forma
    c1 = photon_distribution(df["alexa"], 300000)
    c2 = photon_distribution(df["atto"], 100000)
    c3 = photon_distribution(df["cerulean"], 400000)
    c4 = photon_distribution(df["mito_orange"], 200000)

    comp1 = numpy.histogram(c1, bins=35)[0]
    comp2 = numpy.histogram(c2, bins=35)[0]
    comp3 = numpy.histogram(c3, bins=35)[0]
    comp4 = numpy.histogram(c4, bins=35)[0]
    data = comp1 + comp2 + comp3 + comp4

    plotty = True
    if plotty:
        plt.figure(1)
        plt.plot(df["lambda"], comp1, "r", label="alexa594")
        plt.plot(df["lambda"], comp2, "g", label="atto488")
        plt.plot(df["lambda"], comp3, "b", label="cerulean")
        plt.plot(df["lambda"], comp4, "orange", label="mitotracker_orange")
        plt.plot(df["lambda"], data, "k", label="Combination")
        plt.legend()


    phasor_c1 = phasor_from_signal(comp1, harmonic=[1, 2])
    phasor_c2 = phasor_from_signal(comp2, harmonic=[1, 2])
    phasor_c3 = phasor_from_signal(comp3, harmonic=[1, 2])
    phasor_c4 = phasor_from_signal(comp4, harmonic=[1, 2])
    phasor_exp_comp = phasor_from_signal(data, harmonic=[1, 2])

    datos = {"lambda": df["lambda"],
             "Component1" : comp1,
             "Component2" : comp2,
             "Component3" : comp3,
             "Component4" : comp4,
             "Combination" : data
    }

    dff = pandas.DataFrame(datos)

    pure_components = 'pure_components.csv'

    store = True
    if store:
        dff.to_csv(pure_components, index=False)

    if plotty:
        plt.figure(2)
        plt.plot(df["lambda"], df["alexa"], "r", label="alexa594")
        plt.plot(df["lambda"], df["atto"], "g", label="atto488")
        plt.plot(df["lambda"], df["cerulean"], "b", label="cerulean")
        plt.plot(df["lambda"], df["mito_orange"], "orange", label="mitotracker_orange")
        plt.legend()
        plt.show()

    # Unmixing 
    matrixA = numpy.asarray([[phasor_c1[1][0], phasor_c2[1][0], phasor_c3[1][0], phasor_c4[1][0]], 
                            [phasor_c1[1][1], phasor_c2[1][1], phasor_c3[1][1], phasor_c4[1][1]],
                            [phasor_c1[2][0], phasor_c2[2][0], phasor_c3[2][0], phasor_c4[2][0]],
                            [phasor_c1[2][1], phasor_c2[2][1], phasor_c3[2][1], phasor_c4[2][1]],
                            numpy.ones(4)])
    
    multi_harmonic_real, multi_harmonic_imag = multi_harmonic_phasor(data)
    fractions = multi_component_unmixing_from_phasor(multi_harmonic_real, multi_harmonic_imag, matrixA)

    multi_harmonic_real = \
    [[[-0.39595987, -0.08619137],
      [-0.39595987, -0.08619137],
      [-0.39595987, -0.08619137]],
      [[-0.39595987, -0.08619137],
       [-0.39595987, -0.08619137],
       [-0.39595987, -0.08619137]],
       [[-0.39595987, -0.08619137],
        [-0.39595987, -0.08619137],
        [-0.39595987, -0.08619137]]]
        
    multi_harmonic_imag = \
        [[[ 0.45231211, -0.16960909],
          [ 0.45231211, -0.16960909],
          [ 0.45231211, -0.16960909]],
          [[ 0.45231211, -0.16960909],
           [ 0.45231211, -0.16960909],
           [ 0.45231211, -0.16960909]],
           [[ 0.45231211, -0.16960909],
            [ 0.45231211, -0.16960909],
            [ 0.45231211, -0.16960909]]]
    
    matrixA=[[-0.62442218, -0.47591901, -0.15022351, -0.5899417 ],
             [ 0.10858431, -0.0977382 , -0.24658903,  0.02135662],
             [ 0.21045545,  0.53110438,  0.61459059,  0.41559306],
             [-0.19656654, -0.3379749 , -0.06848862, -0.29857087],
             [ 1.        ,  1.        ,  1.        ,  1.        ]]
        
    fractions = multi_component_unmixing_from_phasor(multi_harmonic_real, 
                                         multi_harmonic_imag,
                                         matrixA)

    # test image as input
    aux = numpy.tile(data, [3, 3, 1])
    multi_harmonic_real, multi_harmonic_imag = multi_harmonic_phasor(aux)
    fractions = multi_component_unmixing_from_phasor(multi_harmonic_real, multi_harmonic_imag, matrixA)

experimental_case = False
if experimental_case:
    # Read the experimental image
    import tifffile

    im = tifffile.imread("/Users/schutyb/Downloads/PhasorSpectralUnmixing_Figshare/Data/examples5comps/3_todo_2.lsm")
    dc, real, imag = phasor_from_signal(im, axis=0)

    plt.figure()
    plt.imshow(dc)

    plot = PhasorPlot(allquadrants=True, title='Components and Phasor')
    plot.hist2d(real, imag, cmap='Blues')
    plt.show()

# %% Read mat and simulate with attos

attos = False
if attos:

    import h5py
    from scipy import signal

    # Open the .mat file
    mat_file1 = h5py.File('/Users/schutyb/Documents/GitHub/phasorpy/tutorials/datacomp/probes/alexa594.mat', 'r')
    mat_file2 = h5py.File('/Users/schutyb/Documents/GitHub/phasorpy/tutorials/datacomp/probes/atto425.mat', 'r')
    mat_file3 = h5py.File('/Users/schutyb/Documents/GitHub/phasorpy/tutorials/datacomp/probes/atto488.mat', 'r')
    mat_file4 = h5py.File('/Users/schutyb/Documents/GitHub/phasorpy/tutorials/datacomp/probes/atto550.mat', 'r')
    mat_file5 = h5py.File('/Users/schutyb/Documents/GitHub/phasorpy/tutorials/datacomp/probes/atto647N.mat', 'r')

    # Access specific variables
    variable1 = mat_file1['alexa594'][:][1]
    variable2 = mat_file2['atto425'][:][1]
    variable3 = mat_file3['atto488'][:][1]
    variable4 = mat_file4['atto550'][:][1]
    variable5 = mat_file5['atto647N'][:][1]

    # Resampling
    lamb = numpy.linspace(400, 800, 200)
    var1 = signal.resample(variable1, 200, lamb)
    var2 = signal.resample(variable2, 200, lamb)
    var3 = signal.resample(variable3, 200, lamb)
    var4 = signal.resample(variable4, 200, lamb)
    var5 = signal.resample(variable5, 200, lamb)

    var1 = expandir_con_ceros(var1[0], 0, 200) + 1
    var2 = expandir_con_ceros(var2[0], 50, 150) + 1
    var3 = expandir_con_ceros(var3[0], 100, 100) + 1
    var4 = expandir_con_ceros(var4[0], 150, 50) + 1
    var5 = expandir_con_ceros(var5[0], 200, 0) + 1

    lamb = numpy.linspace(400, 800, 100)

    plotty = False
    if plotty:
        plt.plot(lamb, var1)
        plt.plot(lamb, var2)
        plt.plot(lamb, var3)
        plt.plot(lamb, var4)
        plt.plot(lamb, var5)
        plt.show()
    
    # distribuye 1M fotones de la sigueinte forma
    c1 = photon_distribution(var1, 300000)
    c2 = photon_distribution(var2, 100000)
    c3 = photon_distribution(var3, 400000)
    c4 = photon_distribution(var4, 200000)
    # c5 = photon_distribution(var5, 200000)

    comp1 = numpy.histogram(c1, bins=100)[0]
    comp2 = numpy.histogram(c2, bins=100)[0]
    comp3 = numpy.histogram(c3, bins=100)[0]
    comp4 = numpy.histogram(c4, bins=100)[0]
    # comp5 = numpy.histogram(c5, bins=100)[0]
    data = comp1 + comp2 + comp3 + comp4

    datos = {"lambda": lamb,
             "Component1" : comp1,
             "Component2" : comp2,
             "Component3" : comp3,
             "Component4" : comp4,
             "Combination" : data, 
    }

    import pandas
    dff = pandas.DataFrame(datos)

    components_4 = 'components_4_test.csv'

    store = True
    if store:
        dff.to_csv(components_4, index=False)

# %% use deltas
delt = True
if delt:
    import numpy as np

    def generar_delta(n, posicion, high):
        """
        Genera una función delta en la posición especificada.

        Args:
        n (int): Longitud del array.
        posicion (int): Posición donde se ubicará el delta (0-indexed).

        Returns:
        numpy array: Array con la función delta.
        """
        delta = np.zeros(n)
        delta[posicion] = high
        return delta

    # Longitud del array
    longitud = 100

    # Generar 4 funciones delta en diferentes posiciones
    delta1 = generar_delta(longitud, 10, 1) 
    delta2 = generar_delta(longitud, 40, 4)
    delta3 = generar_delta(longitud, 70, 2)
    delta4 = generar_delta(longitud, 80, 3)

    # Mostrar los arrays generados
    plt.figure(1)
    plt.plot(delta1)
    plt.plot(delta2)
    plt.plot(delta3)
    plt.plot(delta4)
    plt.show()

    # distribuye 1M fotones de la sigueinte forma
    #c1 = photon_distribution(delta1, 300000)
    #c2 = photon_distribution(delta2, 100000)
    #c3 = photon_distribution(delta3, 400000)
    #c4 = photon_distribution(delta4, 200000)
    # c5 = photon_distribution(var5, 200000)

    # comp1 = numpy.histogram(c1, bins=100)[0]
    # comp2 = numpy.histogram(c2, bins=100)[0]
    # comp3 = numpy.histogram(c3, bins=100)[0]
    # comp4 = numpy.histogram(c4, bins=100)[0]
    # comp5 = numpy.histogram(c5, bins=100)[0]
    data = delta1 + delta2 + delta3 + delta4

    datos = {"lambda": numpy.linspace(400, 800, 100),
                "Component1" : delta1,
                "Component2" : delta2,
                "Component3" : delta3,
                "Component4" : delta4,
                "Combination" : data, 
    }

    import pandas
    dff = pandas.DataFrame(datos)

    components_delta_4 = 'components_deltas_test.csv'

    store = True
    if store:
        dff.to_csv(components_delta_4, index=False)