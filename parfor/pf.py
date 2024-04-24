from multiprocessing import Pool, cpu_count
from time import time

def process_chunk(args):
    start_idx, end_idx, iterator, callback, callback_args, callback_kwargs = args
    results = []
    for i in range(start_idx, end_idx):
        results.append(callback(iterator[i], *callback_args, **callback_kwargs))
    return results

def independent_parfor(callback, num_cpus=None, iterator=None, *args, **kwargs):
    if num_cpus is None:
        num_cpus = cpu_count()

    total_loops = len(iterator)
    loops_per_cpu = total_loops // num_cpus
    remainder = total_loops % num_cpus

    # Start timer
    start = time()

    # Prepare the list of arguments for each process
    chunks = []
    start_idx = 0
    for i in range(num_cpus):
        end_idx = start_idx + loops_per_cpu + (1 if i < remainder else 0)
        chunks.append((start_idx, end_idx, iterator, callback, args, kwargs))
        start_idx = end_idx

    # Submit tasks to the pool using imap for ordered results
    with Pool(processes=num_cpus) as pool:
        results = pool.imap(process_chunk, chunks)

        # Accumulate results in order
        final_result = []
        for chunk_result in results:
            final_result.extend(chunk_result)

    return final_result

# # Example usage
# def process_item(item):
#     return item * 1

# # Usage example
# iterator = list(range(116))  # Adjusted iterator length for demonstration
# final_result = independent_parfor(process_item, num_cpus=10, iterator=iterator)
# print("Final result:", final_result)
