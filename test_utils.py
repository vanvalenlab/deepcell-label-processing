""" Test the util functions in utils.py """

import numpy as np

import utils


def test_tile_around_center_small():
    """Test the tile_around_center function"""
    arr = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]])
    output = utils.tile_around_center(arr, 2, 1, 1)
    assert np.array_equal(output, np.array([[[[6]], [[7]], [[10]], [[11]]]]))
    output = utils.tile_around_center(arr, 2, 2, 2)
    assert np.array_equal(
        output,
        np.array(
            [
                [
                    [[1, 2], [5, 6]],
                    [[3, 4], [7, 8]],
                    [[9, 10], [13, 14]],
                    [[11, 12], [15, 16]],
                ]
            ]
        ),
    )


def test_tile_around_center_larger():
    """Test the tile_around_center function on a larger, asymmetric array"""
    arr = np.array(
        [
            [
                [
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [9, 10, 11, 12, 13, 14, 15, 16],
                    [17, 18, 19, 20, 21, 22, 23, 24],
                    [25, 26, 27, 28, 29, 30, 31, 32],
                ]
            ]
        ]
    )
    output = utils.tile_around_center(arr, 2, 2, 2)
    assert np.array_equal(
        output,
        np.array(
            [
                [
                    [[3, 4], [11, 12]],
                    [[5, 6], [13, 14]],
                    [[19, 20], [27, 28]],
                    [[21, 22], [29, 30]],
                ]
            ]
        ),
    )
