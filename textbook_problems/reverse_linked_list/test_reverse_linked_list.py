import pytest
from typing import Optional, List
from reverse_linked_list import *

def test_reverse_linked_list():
    input_list = [1, 2, 3, 4, 5]
    expected_output_list = [5, 4, 3, 2, 1]

    input_linked_list = list_to_linked_list(input_list)
    reversed_linked_list = reverse_linked_list(input_linked_list)
    reversed_list = linked_list_to_list(reversed_linked_list)

    assert reversed_list == expected_output_list

def test_reverse_empty_linked_list():
    input_list = []
    expected_output_list = []

    input_linked_list = list_to_linked_list(input_list)
    reversed_linked_list = reverse_linked_list(input_linked_list)
    reversed_list = linked_list_to_list(reversed_linked_list)

    assert reversed_list == expected_output_list

def test_reverse_single_node_linked_list():
    input_list = [1]
    expected_output_list = [1]

    input_linked_list = list_to_linked_list(input_list)
    reversed_linked_list = reverse_linked_list(input_linked_list)
    reversed_list = linked_list_to_list(reversed_linked_list)

    assert reversed_list == expected_output_list

def test_reverse_linked_list_1():
    input_list = [1, 2, 3, 4, 5]
    expected_output_list = [5, 4, 3, 2, 1]
    input_linked_list = list_to_linked_list(input_list)
    reversed_linked_list = reverse_linked_list(input_linked_list)
    reversed_list = linked_list_to_list(reversed_linked_list)
    assert reversed_list == expected_output_list

def test_reverse_linked_list_2():
    input_list = [10, 20, 30]
    expected_output_list = [30, 20, 10]
    input_linked_list = list_to_linked_list(input_list)
    reversed_linked_list = reverse_linked_list(input_linked_list)
    reversed_list = linked_list_to_list(reversed_linked_list)
    assert reversed_list == expected_output_list

def test_reverse_linked_list_3():
    input_list = [1, 1, 1, 1, 1]
    expected_output_list = [1, 1, 1, 1, 1]
    input_linked_list = list_to_linked_list(input_list)
    reversed_linked_list = reverse_linked_list(input_linked_list)
    reversed_list = linked_list_to_list(reversed_linked_list)
    assert reversed_list == expected_output_list

def test_reverse_linked_list_4():
    input_list = [1, 2, 1, 2, 1]
    expected_output_list = [1, 2, 1, 2, 1]
    input_linked_list = list_to_linked_list(input_list)
    reversed_linked_list = reverse_linked_list(input_linked_list)
    reversed_list = linked_list_to_list(reversed_linked_list)
    assert reversed_list == expected_output_list

def test_reverse_linked_list_5():
    input_list = [1, 2, 3, 2, 1]
    expected_output_list = [1, 2, 3, 2, 1]
    input_linked_list = list_to_linked_list(input_list)
    reversed_linked_list = reverse_linked_list(input_linked_list)
    reversed_list = linked_list_to_list(reversed_linked_list)
    assert reversed_list == expected_output_list

def test_reverse_linked_list_6():
    input_list = [7]
    expected_output_list = [7]
    input_linked_list = list_to_linked_list(input_list)
    reversed_linked_list = reverse_linked_list(input_linked_list)
    reversed_list = linked_list_to_list(reversed_linked_list)
    assert reversed_list == expected_output_list

def test_reverse_linked_list_7():
    input_list = [1, 2]
    expected_output_list = [2, 1]
    input_linked_list = list_to_linked_list(input_list)
    reversed_linked_list = reverse_linked_list(input_linked_list)
    reversed_list = linked_list_to_list(reversed_linked_list)
    assert reversed_list == expected_output_list
