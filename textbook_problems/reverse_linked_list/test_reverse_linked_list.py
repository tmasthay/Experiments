import pytest
from typing import Optional, List
from reverse_linked_list import *

# Helper function to convert a list to a linked list
def list_to_linked_list(lst: List[int]) -> Optional[ListNode]:
    if not lst:
        return None
    head = ListNode(lst[0])
    current = head
    for val in lst[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

# Helper function to convert a linked list to a list
def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
    lst = []
    current = head
    while current:
        lst.append(current.val)
        current = current.next
    return lst

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
