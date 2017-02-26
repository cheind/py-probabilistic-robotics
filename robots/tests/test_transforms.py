import numpy as np
from robots import transforms

def test_hom():

    a = np.array([[2, 3, 5], [3, 4, 6]])

    np.testing.assert_allclose(
        transforms.hom(a, n=1., axis=0), 
        np.array([[2, 3, 5], [3, 4, 6], [1, 1, 1]]))

    np.testing.assert_allclose(
        transforms.hom(a, n=1., axis=1), 
        np.array([[2, 3, 5, 1], [3, 4, 6, 1]]))

def test_hnorm():
    a = np.array([[2, 4, 6], [8, 4, 6]])
    b = transforms.hom(a, n=2., axis=0)

    np.testing.assert_allclose(
        transforms.hnorm(b, axis=0), 
        np.array([[1, 2, 3], [4, 2, 3]]))

    np.testing.assert_allclose(
        transforms.hnorm(b, axis=0, divide=False), 
        np.array([[2, 4, 6], [8, 4, 6]]))

    b = transforms.hom(a, n=1., axis=1)
    np.testing.assert_allclose(
        transforms.hnorm(b, axis=1, divide=True), 
        np.array([[2, 4, 6], [8, 4, 6]]))

    b = transforms.hom(a, n=2., axis=1)
    np.testing.assert_allclose(
        transforms.hnorm(b, axis=1, divide=False), 
        np.array([[2, 4, 6], [8, 4, 6]]))
