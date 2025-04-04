# CudaImageProcessor

This project demonstrates CUDA-based image processing with a command-line interface. It supports:

- Converting an image to grayscale (using a CUDA kernel)
- Compressing an Image
- Resizing an image using nearest-neighbor interpolation (CUDA)
- Applying a convolution filter (edge detection) (CUDA)

The input image is read with OpenCV and, after processing, saved to the user-specified output path. The output format is automatically determined by the file extension (e.g., .png, .jpg).

## Build and Run Instructions

1. Create a build directory:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
2. Run
   ```bash
   ./CudaImageProcessor
