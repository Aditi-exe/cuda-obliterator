#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <windows.h> // for sleep
#include <conio.h> // for _kbhit() and _getch()
#include <vector>
#include <cstdlib>



__global__ void burnGPU(float* data, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size){
        float val = data[idx];
        // Perform some operation on val, e.g., multiply by a constant
        for (int i = 0; i < 1000000; ++i){
            val = val * 1.000001f + 0.000001f; // Simulating a long computation
        }
        data[idx] = val;
    }
}

/* * Enable non-blocking input for POSIX systems
 * This function sets the terminal to non-blocking mode so that _kbhit() can be used to check for user input without blocking.
 * ONLY FOR POSIX SYSTEMS
 * For Windows, the terminal settings are not changed, as _kbhit() and _getch() handle non-blocking input natively.
 * If you want to enable non-blocking input in Windows, you can use the Windows API functions.

// set terminal to non blocking mode
void enableNonBlockingInput(){
    termios t;
    tcgetattr(STDIN_FILENO, &t);
    t.c_lflag &= ~ICANON; // Disable line buffering
    t.c_lflag &= ~ECHO;   // Disable echo
    tcsetattr(STDIN_FILENO, TCSANOW, &t);
    fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK); // Set non-blocking mode; non blocking input
}
*/

/*
 * Restore terminal settings to normal (blocking) mode
 * This function should be called before the program exits to avoid leaving the terminal in a non-blocking state.
 * ONLY FOR POSIX SYSTEMS
 * For Windows, the terminal settings are not changed, as _kbhit() and _getch() handle non-blocking input natively.
 * If you want to restore terminal settings in Windows, you can use the Windows API functions.
 
// reset terminal to normal (blocking) mode
void restoreTerminal(){
    termios t;
    tcgetattr(STDIN_FILENO, &t);
    t.c_lflag |= ICANON; // Enable line buffering
    t.c_lflag |= ECHO;   // Enable echo
    tcsetattr(STDIN_FILENO, TCSANOW, &t);
}
*/
// Note: The restoreTerminal function is commented out because it is not used in this Windows-specific



int main(){
    std::srand(std::time(nullptr)); // Seed for random number generation
    std::vector<void*> allocations;

    std::cout << "CUDA Device Information:\n";
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device Name: " << prop.name << "\n";

    std::cout << "Running GPU memory load... press any key to stop.\n";

    while(true){
        // allocate random amount between 10MB to 200MB
        size_t sizeMB = 10 + std::rand() % 191; // Random size between 10MB and 200MB
        size_t sizeBytes = sizeMB * 1024 * 1024; // Convert to bytes

        void* ptr;
        cudaError_t err = cudaMalloc(&ptr, sizeBytes);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";
            //std::cerr << "Failed to allocate " << sizeMB << " MB on GPU.\n";
            break;
        }   

        cudaMemset(ptr, 0, sizeBytes); // Initialize memory to zero
        allocations.push_back(ptr); // Store the pointer for later deallocation

        std::cout << "Looping...\n";
        std::cout << "Allocated " << sizeMB << "MB on GPU.\n";

        Sleep(1000);

        if (_kbhit()){
            _getch(); // Consume the key press
            std::cout << "Key pressed. Exiting.\n";
            break;
        }
    }

    // free everything
    for (auto ptr : allocations){
        cudaFree(ptr);
    }

    return 0;
}












    /* This code is a CUDA program that continuously performs computations on the GPU while allowing the user to stop the computation by pressing 'q' or 'Q'.
     * It initializes a large array on the GPU, performs a long computation in a loop, and checks for user input to stop the computation.
    
     const int size = 1 << 20;
    float* d_data;

    cudaMalloc((void**)&d_data, size * sizeof(float));
    cudaMemset(d_data, 1, size * sizeof(float)); // Initialize data to 1.0f

    dim3 threads(256);
    dim3 blocks((size + threads.x - 1) / threads.x);

    //enableNonBlockingInput(); // Set terminal to non-blocking mode

    std::cout << "Press 'q' to stop the computation..." << std::endl;

    while(true){
        burnGPU<<<blocks, threads>>>(d_data, size);
        cudaDeviceSynchronize(); // Wait for GPU to finish

        // log gpu memory usage every second
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);

        size_t used_mem = total_mem - free_mem;
        std::cout << "Used GPU memory: " << used_mem / (1024 * 1024) << " MB"
                    << " / Total GPU memory: " << total_mem / (1024 * 1024) << " MB\n" << std::endl;

        Sleep(1000); // Simulate some delay for the next computation

        // Check for user input
        if(_kbhit()){
            char c = _getch(); // Get the character without waiting for Enter
            if(c == 'q' || c == 'Q'){ // If 'q' or 'Q' is pressed, exit the loop
                std::cout << "Stopping computation..." << std::endl;
                break;
            }
        }

        // Optional: Print progress or status
        std::cout << "Computation in progress..." << std::endl;
    }

    cudaFree(d_data); // Free GPU memory
    return 0;
    */


