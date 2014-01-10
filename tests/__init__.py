# -*- coding: utf-8 -*-

##  IMPORTS  ##
import numpy
from numpy.testing import assert_allclose
from pyvows import Vows, expect


## Numpy setup
numpy.set_printoptions(suppress=True)


## assert almost equal
@Vows.assertion
def to_almost_equal(topic, expected, rtol=1e-5, atol=1e-8):
    assert_allclose(topic, expected, rtol, atol)

