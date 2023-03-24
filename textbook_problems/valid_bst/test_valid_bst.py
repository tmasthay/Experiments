import pytest
from typing import Optional
from valid_bst import *
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

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

# Test 1: Empty tree
def test_empty_tree():
    assert is_valid_bst(None) == True

# Test 3: Two node tree
def test_two_node_tree():
    root = TreeNode(2)
    root.left = TreeNode(1)
    assert is_valid_bst(root) == True

# Test 4: Two node tree with duplicate values
def test_two_node_tree_with_duplicate_values():
    root = TreeNode(2)
    root.left = TreeNode(2)
    assert is_valid_bst(root) == True

# Test 5: Three node tree with duplicate values
def test_three_node_tree_with_duplicate_values():
    root = TreeNode(2)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    assert is_valid_bst(root) == True

# Test 6: Three node tree with non-duplicate values
def test_three_node_tree_with_non_duplicate_values():
    root = TreeNode(2)
    root.left = TreeNode(1)
    root.right = TreeNode(3)
    assert is_valid_bst(root) == True

# Test 7: Four node tree with non-duplicate values
def test_four_node_tree_with_non_duplicate_values():
    root = TreeNode(4)
    root.left = TreeNode(2)
    root.right = TreeNode(5)
    root.left.left = TreeNode(1)
    assert is_valid_bst(root) == True

# Test 8: Four node tree with duplicate values
def test_four_node_tree_with_duplicate_values():
    root = TreeNode(2)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.right.right = TreeNode(3)
    assert is_valid_bst(root) == True

# Test 9: Balanced seven node tree with non-duplicate values
def test_balanced_seven_node_tree_with_non_duplicate_values():
    root = TreeNode(4)
    root.left = TreeNode(2)
    root.right = TreeNode(6)
    root.left.left = TreeNode(1)
    root.left.right = TreeNode(3)
    root.right.left = TreeNode(5)
    root.right.right = TreeNode(7)
    assert is_valid_bst(root) == True

# Test 10: Unbalanced seven node tree with non-duplicate values
def test_unbalanced_seven_node_tree_with_non_duplicate_values():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.left.left = TreeNode(3)
    root.left.left.left = TreeNode(4)
    root.right = TreeNode(5)
    root.right.right = TreeNode(6)
    root.right.right.right = TreeNode(7)
    assert is_valid_bst(root) == False

# Test 11: Tree with negative values
def test_tree_with_negative_values():
    root = TreeNode(0)
    root.left = TreeNode(-1)
    root.right = TreeNode(2)
    assert is_valid_bst(root) == True

# Test 12: Tree with positive and negative values
def test_tree_with_positive_and_negative_values():
    root = TreeNode(-2)
    root.left = TreeNode(-3)
    root.right = TreeNode(3)
    root.right.left = TreeNode(1)
    root.right.right = TreeNode(4)
    assert is_valid_bst(root) == True

# Test 13: Tree with all negative values
def test_tree_with_all_negative_values():
    root = TreeNode(-2)
    root.left = TreeNode(-3)
    root.left.left = TreeNode(-4)
    root.right = TreeNode(-1)
    assert is_valid_bst(root) == True

# Test 14: Tree with all positive values
def test_tree_with_all_positive_values():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.left.left = TreeNode(3)
    root.right = TreeNode(4)
    assert is_valid_bst(root) == False

# Test 15: Tree with duplicate values
def test_tree_with_duplicate_values():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.left.left = TreeNode(3)
    root.right = TreeNode(2)
    root.right.right = TreeNode(3)
    assert is_valid_bst(root) == False

# Test 16: Tree with large number of nodes
def test_large_tree():
    root = TreeNode(50)
    for i in range(1, 101):
        insert_node(root, i)
    assert is_valid_bst(root) == True

# Test 17: Tree with negative and positive infinity
def test_tree_with_inf():
    root = TreeNode(0)
    root.left = TreeNode(float('-inf'))
    root.right = TreeNode(float('inf'))
    assert is_valid_bst(root) == True

# Test 18: Tree with very large values
def test_tree_with_large_values():
    root = TreeNode(999999999999999999)
    root.left = TreeNode(888888888888888888)
    root.left.left = TreeNode(777777777777777777)
    root.right = TreeNode(1000000000000000000)
    assert is_valid_bst(root) == True

# Test 19: Tree with very small values
def test_tree_with_small_values():
    root = TreeNode(0.000000000000000001)
    root.left = TreeNode(0.0000000000000000001)
    root.left.left = TreeNode(0.00000000000000000001)
    root.right = TreeNode(0.000000000000000002)
    assert is_valid_bst(root) == True

# Test 20: Tree with floating point errors
def test_tree_with_floating_point_errors():
    root = TreeNode(0.1)
    root.left = TreeNode(0.2)
    root.right = TreeNode(0.3)
    assert is_valid_bst(root) == False

def test_is_valid_bst_complexity():
    min_size = 100
    max_size = 80000
    N = 1000
    input_lengths = np.array(range(min_size, max_size, N))
    execution_times = []

    valid_range = 1000
    num_correct = 0
    p = 1.0
    for length in input_lengths:
        u = np.round(valid_range * np.random.random(length))
        is_sorted = np.random.random() < p
        if( is_sorted ):
            u = np.sort(u)
        root = binary_tree(u)       
        #assert(np.all(u == in_order_extract(root)))
        t = time.time()
        computed_sorted = is_valid_bst(root)
        execution_times.append(time.time() - t)
        if( computed_sorted == is_sorted ):
            num_correct += 1
    acc = float(num_correct) / float(N)
    
    #Get regression info
    p = np.polyfit(np.log(input_lengths), np.log(execution_times), 1)
    reg_execution = np.exp(p[1]) * input_lengths**p[0]

    filename = 'valid_bst_plots/performance.pdf'
    if( os.path.exists(filename) ):
        os.system('rm %s'%filename)

    palette = sns.color_palette("Set1", n_colors=3)
    palette = [plt.matplotlib.colors.to_rgb(color) for color in palette]
    plt.rcParams.update({'text.usetex' : True})
    plt.subplots(figsize=(10,8))
    plt.plot(
        input_lengths, 
        execution_times, 
        color=palette[0],
        linestyle='-',
        label='Actual Runtime'
    )
    plt.plot(
        input_lengths, 
        reg_execution, 
        color=palette[1],
        linestyle='-.',
        label='Regressed Runtime'
    )
    plt.xlabel('Input Length')
    plt.ylabel('Execution Time (s)')
    plt.title('is_valid_bst Algorithm Complexity')
    plt.text(
        0.6,
        0.5,
        r'$m = %.4f \approx 1.0 \\Accuracy=%.4f$'%(p[0],acc),
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )
    plt.legend()
    plt.savefig(filename)
    plot_gen = os.path.exists(filename)

    tol = 0.1
    slope = p[0]
    print(p)
    pass_test = np.abs(1.0 - slope) < tol
    assert pass_test, \
        "slope=%.2e, slope_pass=%s, plot_gen=%s"%(slope, pass_test, plot_gen)
