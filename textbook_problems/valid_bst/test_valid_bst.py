import pytest
from typing import Optional
from valid_bst import *

def test_single_node():
    root = TreeNode(5)
    assert is_valid_bst(root) == True

def test_valid_bst():
    root = TreeNode(10)
    root.left = TreeNode(5)
    root.right = TreeNode(15)
    root.left.left = TreeNode(3)
    root.left.right = TreeNode(7)
    assert is_valid_bst(root) == True

def test_invalid_bst():
    root = TreeNode(10)
    root.left = TreeNode(5)
    root.right = TreeNode(15)
    root.left.left = TreeNode(3)
    root.left.right = TreeNode(12)  # Invalid, as 12 > 10
    assert is_valid_bst(root) == False

def test_valid_bst_with_negative_numbers():
    root = TreeNode(0)
    root.left = TreeNode(-1)
    root.right = TreeNode(2)
    assert is_valid_bst(root) == True

def test_invalid_bst_with_negative_numbers():
    root = TreeNode(0)
    root.left = TreeNode(-1)
    root.right = TreeNode(-2)  # Invalid, as -2 < 0
    assert is_valid_bst(root) == False
