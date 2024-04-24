from pf import independent_parfor

# Example usage
def process_item(item):
    return item * 1

# Usage example
iterator = list(range(116))  # Adjusted iterator length for demonstration
final_result = independent_parfor(process_item, num_cpus=60, iterator=iterator)
print("Final result:", final_result)