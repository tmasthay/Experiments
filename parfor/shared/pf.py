import torch
from multiprocessing import Pool, cpu_count, Array

def init_pool(shared_results):
    global shared_arrays
    shared_arrays = shared_results

def process_chunk(args):
    start_idx, end_idx, iterator, callback = args
    offset = 10  # each tensor has 10 elements
    for i in range(start_idx, end_idx):
        result = callback(iterator[i])  # Returns a tuple of torch.Tensor, each of size 10
        for j, res in enumerate(result):
            # Flatten each tensor into the corresponding shared array
            shared_arrays[j][i * offset:(i + 1) * offset] = res.numpy()

class ParallelProcessor:
    def __init__(self, callback, num_cpus=None, iterator=None, *args, **kwargs):
        self.num_cpus = num_cpus if num_cpus else cpu_count()
        self.iterator = iterator
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        self.num_elements = 5  # Expecting a 5-tuple of results
        # Each element of each tuple is a tensor of 10 doubles, multiplied by the number of items
        self.shared_results = [Array('d', len(iterator) * 10) for _ in range(self.num_elements)]

    def run(self):
        total_loops = len(self.iterator)
        loops_per_cpu = total_loops // self.num_cpus
        remainder = total_loops % self.num_cpus
        
        chunks = []
        start_idx = 0
        for i in range(self.num_cpus):
            end_idx = start_idx + loops_per_cpu + (1 if i < remainder else 0)
            chunks.append((start_idx, end_idx, self.iterator, self.callback))
            start_idx = end_idx

        with Pool(processes=self.num_cpus, initializer=init_pool, initargs=(self.shared_results,)) as pool:
            pool.map(process_chunk, chunks)

        # Convert each shared array to a list and return a tuple of lists
        return tuple([list(arr[i * 10:(i + 1) * 10]) for i in range(len(self.iterator))] for arr in self.shared_results)

# Example modified get_spline function
def get_spline(item):
    # Now returns a 5-tuple of torch.Tensors, each containing 10 random doubles
    return tuple(torch.rand(10) for _ in range(5))

# Usage example
iterator = list(range(10))  # Smaller range for demonstration
processor = ParallelProcessor(get_spline, num_cpus=4, iterator=iterator)
final_result = processor.run()
print("Final result:", final_result)
