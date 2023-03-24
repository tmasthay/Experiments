from typing import Optional
import numpy as np

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def binary_tree(x):
    if( len(x) == 0 ): return None
    if( len(x) == 1 ): return TreeNode(x[0])
    random_pivot = min(len(x) - 1, int(len(x) * np.random.random()))
    root = TreeNode(x[random_pivot])
    root.left = binary_tree(x[:random_pivot])
    root.right = binary_tree(x[(random_pivot+1):])
    return root

# Helper function to insert nodes into a binary search tree
def insert_node(root, val):
    if not root:
        root = TreeNode(val)
    elif val < root.val:
        root.left = insert_node(root.left, val)
    else:
        root.right = insert_node(root.right, val)
    return root

def in_order_print(root: Optional[TreeNode]):
    if( root == None ):
        return
    else:
        in_order_print(root.left)
        print(root.val)
        in_order_print(root.right)

def in_order_extract(root: Optional[TreeNode]):
    def helper(subtree: Optional[TreeNode], runner):
        if( root == None ):
            return
        else:
            helper(root.left, runner)
            runner.append(root.val)
            helper(root.right, runner)
    v = []
    helper(root, v)
    return np.array(v)

def is_valid_bst(root: Optional[TreeNode]) -> bool:
    def helper(subtree: Optional[TreeNode], runner):
        # if( subtree == None ):
        #     return True
        # else:
        #     left_valid = helper(subtree.left, leftmost, rightmost)
        #     if( rightmost > subtree.val ): return False
        #     right_valid = helper(subtree.right, leftmost, rightmost)
        #     return left_valid and right_valid
        if( subtree != None ):
            helper(subtree.left, runner)
            runner.append(subtree.val)
            helper(subtree.right, runner)
    v = []
    helper(root, v)
    if( len(v) < 10 ):
        print(v, flush=True)
    is_sorted = np.all([v[i-1] <= v[i] for i in range(1,len(v))])
    return is_sorted

if( __name__ == "__main__" ):
    # root = TreeNode(10)
    # root.left = TreeNode(5)
    # root.right = TreeNode(15)
    # root.left.left = TreeNode(3)
    # root.left.right = TreeNode(12)  # Invalid, as 12 > 10
    # is_sorted = is_valid_bst(root)
    # print(is_sorted)
    x = np.array(range(200))
    root = binary_tree(x)
    in_order_print(root)
    

