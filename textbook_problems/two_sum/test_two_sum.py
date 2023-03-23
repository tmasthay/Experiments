import pytest
from two_sum import two_sum
import numpy as np

def test_two_sum_basic():
    nums = np.array([2, 7, 11, 15], dtype=int)
    target = 9
    assert set(two_sum(nums, target)) == set([0, 1])

def test_two_sum_negative_numbers():
    nums = np.array([-3, 4, 3, 90], dtype=int)
    target = 0
    assert set(two_sum(nums, target)) == set([0, 2])

def test_two_sum_single_pair():
    nums = np.array([5, 7, 1, 8], dtype=int)
    target = 9
    assert set(two_sum(nums, target)) == set([2, 3])

def test_two_sum_multiple_pairs():
    nums = np.array([1, 3, 5, 7, 9], dtype=int)
    target = 12
    # Here we are considering only one solution would exist, so one of the pairs (3, 9) or (5, 7) is correct.
    assert set(two_sum(nums, target)) in [set([1, 4]), set([2, 3])]

def test_two_sum_same_element():
    nums = np.array([3, 2, 4], dtype=int)
    target = 6
    assert set(two_sum(nums, target)) == set([1, 2])

def test_two_sum_same_element_empty():
    nums = np.array([3, 2, 5], dtype=int)
    target = 6
    assert set(two_sum(nums, target)) == set(())

def test_two_sum_same_element_repeats():
    nums = np.array([3, 2, 3, 3, 5], dtype=int)
    target = 6
    assert set(two_sum(nums, target)) in [set([0,2]), set([0,3]), set([2,3])]