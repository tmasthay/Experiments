from typing import Optional,List

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

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

def reverse_linked_list(head: Optional[ListNode]) -> Optional[ListNode]:
    def helper(h: Optional[ListNode], runner: Optional[ListNode]):
        if( h is None ):
            return runner
        
        next_node = h.next
        h.next = runner
        return helper(next_node, h)
    return helper(head, None)


if( __name__ == "__main__" ):
    input_list = [1, 2, 3, 4, 5]
    expected_output_list = [5, 4, 3, 2, 1]

    input_linked_list = list_to_linked_list(input_list)
    reversed_linked_list = reverse_linked_list(input_linked_list)
    reversed_list = linked_list_to_list(reversed_linked_list)

    print(reversed_list)
