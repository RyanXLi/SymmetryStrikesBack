import torch

def do_work(memory_size_gb, gpuids):
    # Calculate the number of bytes to allocate based on the input in gigabytes
    bytes_to_allocate = memory_size_gb * 1024 * 1024 * 1024  # GB to Bytes

    # List to hold the tensors that occupy the VRAM
    memory_occupiers = []

    # Iterate over all available GPUs and allocate memory
    for i in gpuids:
        # Set device to current GPU
        device = torch.device(f'cuda:{i}')
        print(f"Allocating {memory_size_gb}GB on GPU {i}")

        # Allocate memory and add the tensor to the list
        # We use empty_like to create a tensor of the same size
        tensor = torch.empty(int(bytes_to_allocate / 4), dtype=torch.float32, device=device)
        memory_occupiers.append(tensor)

    # Keep the script alive to maintain the memory allocation
    input("Memory is allocated. Press Enter to release memory and exit...")

if __name__ == '__main__':
    # Specify the amount of VRAM to use per GPU in GB
    vram_to_use = 10  # Change this number based on your needs
    gpuids = [0, 3]
    do_work(vram_to_use, gpuids)
