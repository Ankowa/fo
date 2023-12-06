import pytest
import numpy as np
from simulation import OscillatorsSimulation


@pytest.fixture
def sim():
    return OscillatorsSimulation([1, 1], [1, 2, 3])


@pytest.mark.parametrize(
    "arr, out",
    [
        (
            np.array([1, 1, 1, 2, 3, 4, 4, 6, 7, 5, 7, 8, 9, 9]),
            [
                np.array([0, 1, 2]),
                np.array([5, 6]),
                np.array([7, 8, 9, 10]),
                np.array([12, 13]),
            ],
        ),
        (
            np.array([1, 4, 3, 2, 3, 5, 5]),
            [np.array([1, 2, 3, 4]), np.array([5, 6])],
        ),
        (np.array([1, 2, 5, 3, 4, 2, 8]), [np.array([1, 2, 3, 4, 5])]),
        (
            np.array([1, 1, 2, 2, 3, 3]),
            [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])],
        ),
        (np.array([1, 3, 2, 4]), [np.array([1, 2])]),
        (
            np.array([1, 1, 2, 2, 3, 4, 4, 7, 5, 5, 8, 10, 10, 9]),
            [
                np.array([0, 1]),
                np.array([2, 3]),
                np.array([5, 6]),
                np.array([7, 8, 9]),
                np.array([11, 12, 13]),
            ],
        ),
        (
            np.array([0, 0, 0, 1, 1, 2, 4, 3, 7, 5, 6, 11, 11, 9, 12, 13, 13, 15]),
            [
                np.array([0, 1, 2]),
                np.array([3, 4]),
                np.array([6, 7]),
                np.array([8, 9, 10]),
                np.array([11, 12, 13]),
                np.array([15, 16]),
            ],
        ),
    ],
)
def test_get_indices_of_collisions(sim, arr, out):
    for i in range(len(out)):
        assert np.array_equal(sim.get_indices_of_collisions(arr)[i], out[i])


def test_fail_get_indices_of_collisions(sim):
    with pytest.raises(AssertionError):
        sim.get_indices_of_collisions(np.array([1, 2, 3]))
