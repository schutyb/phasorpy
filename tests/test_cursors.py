"""Tests for the phasorpy.cursors module."""

import numpy
import pytest
from numpy.testing import assert_array_equal

from phasorpy.cursors import label_from_phasor_circular, label_from_ranges


def test_label_from_phasor_circular():
    # real, imag, center, radius=radius
    real = numpy.array([-0.5, -0.5, 0.5, 0.5])
    imag = numpy.array([-0.5, 0.5, -0.5, 0.5])
    center = numpy.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    radius = [0.1, 0.1, 0.1, 0.1]
    labels = label_from_phasor_circular(real, imag, center, radius=radius)
    assert labels.dtype == 'uint8'
    assert_array_equal(labels, [1.0, 2.0, 3.0, 4.0])


def test_label_from_phasor_circular_erros():
    # Test ValueErrors
    real = numpy.array([-0.5, -0.5, 0.5, 0.5])
    imag = numpy.array([-0.5, 0.5, -0.5, 0.5])
    center = numpy.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    radius = [0.1, 0.1, 0.1, 0.1]
    with pytest.raises(ValueError):
        label_from_phasor_circular(
            real, imag, center, radius=[-0.1, 0.1, 0.1, 0.1]
        )
    with pytest.raises(ValueError):
        label_from_phasor_circular(
            numpy.array([-0.5, -0.5, 0.5]), imag, center, radius=radius
        )


def test_label_from_ranges():
    # Test label from ranges
    values = numpy.array([[3.3, 6, 8], [15, 20, 7]])
    ranges = numpy.array([(2, 8), (10, 15), (20, 25)])
    labels = label_from_ranges(values, ranges)
    assert labels.dtype == 'uint8'
    assert_array_equal(labels, [[1, 1, 0], [0, 3, 1]])
