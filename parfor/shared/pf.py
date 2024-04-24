import torch
from multiprocessing import Pool, cpu_count, Array

def init_pool(shared_results):
    global shared_arrays
    shared_arrays = shared_results

def process_chunk(args):
    start_idx, end_idx, iterator, callback, chunk_sizes = args
    for i in range(start_idx, end_idx):
        result = callback(iterator[i], chunk_sizes)  # Passing chunk_sizes to callback
        index = 0
        for j, res in enumerate(result):
            size = chunk_sizes[j]
            abs_start_idx = i * sum(chunk_sizes) + index
            abs_end_idx = abs_start_idx + size
            numpy_data = res.numpy()
            if len(numpy_data) != (abs_end_idx - abs_start_idx):
                raise ValueError(f"Expected size {abs_end_idx - abs_start_idx}, got {len(numpy_data)}")
            shared_arrays[j][index:index + size] = res.numpy()
            index += size

class ParallelProcessor:
    def __init__(self, callback, num_cpus=None, iterator=None, chunk_sizes=None, *args, **kwargs):
        self.num_cpus = num_cpus if num_cpus else cpu_count()
        self.iterator = iterator
        self.callback = callback
        self.chunk_sizes = chunk_sizes
        self.args = args
        self.kwargs = kwargs
        self.num_elements = len(chunk_sizes)
        # Initialize shared memory arrays based on total size required for each chunk
        self.shared_results = [Array('d', len(iterator) * size) for size in chunk_sizes]

    def run(self):
        total_loops = len(self.iterator)
        loops_per_cpu = total_loops // self.num_cpus
        remainder = total_loops % self.num_cpus
        
        chunks = []
        start_idx = 0
        for i in range(self.num_cpus):
            end_idx = start_idx + loops_per_cpu + (1 if i < remainder else 0)
            chunks.append((start_idx, end_idx, self.iterator, self.callback, self.chunk_sizes))
            start_idx = end_idx

        with Pool(processes=self.num_cpus, initializer=init_pool, initargs=(self.shared_results,)) as pool:
            pool.map(process_chunk, chunks)

        # Reconstruct results into lists for each element size
        results = []
        for j, arr in enumerate(self.shared_results):
            size = self.chunk_sizes[j]
            result_list = [arr[i * size:(i + 1) * size] for i in range(len(self.iterator))]
            results.append(result_list)
        return tuple(results)

# Example modified get_spline function
def get_spline(item, chunk_sizes):
    # Returns a tuple of torch.Tensors with sizes specified by chunk_sizes
    return tuple(torch.rand(size) for size in chunk_sizes)

# Usage example
chunk_sizes = [10, 20, 10, 3]
iterator = list(range(10))  # Smaller range for demonstration
processor = ParallelProcessor(get_spline, num_cpus=4, iterator=iterator, chunk_sizes=chunk_sizes)
final_result = processor.run()
print("Final result:", final_result)
