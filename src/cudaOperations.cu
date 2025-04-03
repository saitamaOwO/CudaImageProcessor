#include <cuda_runtime.h>
#include "cudaOperations.h"
#include <stdio.h>
#include <stdlib.h>

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Grayscale conversion kernel
__global__ void grayscaleKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        output[y * width + x] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

extern "C" void convertToGrayscaleCUDA(const unsigned char* h_input, unsigned char* h_output, int width, int height, int channels) {
    unsigned char *d_input, *d_output;
    size_t colorBytes = width * height * channels * sizeof(unsigned char);
    size_t grayBytes = width * height * sizeof(unsigned char);

    cudaCheckError(cudaMalloc(&d_input, colorBytes));
    cudaCheckError(cudaMalloc(&d_output, grayBytes));
    cudaCheckError(cudaMemcpy(d_input, h_input, colorBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    grayscaleKernel<<<grid, block>>>(d_input, d_output, width, height, channels);
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(h_output, d_output, grayBytes, cudaMemcpyDeviceToHost));
    cudaFree(d_input);
    cudaFree(d_output);
}

// Nearest-neighbor image resize kernel
__global__ void resizeKernel(const unsigned char* input, unsigned char* output,
                             int inWidth, int inHeight, int channels,
                             int outWidth, int outHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < outWidth && y < outHeight) {
        float scaleX = static_cast<float>(inWidth) / outWidth;
        float scaleY = static_cast<float>(inHeight) / outHeight;
        int srcX = min(static_cast<int>(x * scaleX), inWidth - 1);
        int srcY = min(static_cast<int>(y * scaleY), inHeight - 1);
        int srcIdx = (srcY * inWidth + srcX) * channels;
        int dstIdx = (y * outWidth + x) * channels;
        for (int c = 0; c < channels; c++) {
            output[dstIdx + c] = input[srcIdx + c];
        }
    }
}

extern "C" void resizeImageCUDA(const unsigned char* h_input, unsigned char* h_output,
                                  int inWidth, int inHeight, int channels,
                                  int outWidth, int outHeight) {
    unsigned char *d_input, *d_output;
    size_t inBytes = inWidth * inHeight * channels * sizeof(unsigned char);
    size_t outBytes = outWidth * outHeight * channels * sizeof(unsigned char);

    cudaCheckError(cudaMalloc(&d_input, inBytes));
    cudaCheckError(cudaMalloc(&d_output, outBytes));
    cudaCheckError(cudaMemcpy(d_input, h_input, inBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((outWidth + block.x - 1) / block.x, (outHeight + block.y - 1) / block.y);
    resizeKernel<<<grid, block>>>(d_input, d_output, inWidth, inHeight, channels, outWidth, outHeight);
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(h_output, d_output, outBytes, cudaMemcpyDeviceToHost));
    cudaFree(d_input);
    cudaFree(d_output);
}

// Convolution kernel
__global__ void convolutionKernel(const unsigned char* input, unsigned char* output,
                                  int width, int height, int channels,
                                  const float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kHalf = kernelSize / 2;
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int ky = -kHalf; ky <= kHalf; ky++) {
                for (int kx = -kHalf; kx <= kHalf; kx++) {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);
                    int imgIdx = (iy * width + ix) * channels + c;
                    int kIdx = (ky + kHalf) * kernelSize + (kx + kHalf);
                    sum += kernel[kIdx] * static_cast<float>(input[imgIdx]);
                }
            }
            int pixel = min(max(static_cast<int>(sum), 0), 255);
            output[(y * width + x) * channels + c] = static_cast<unsigned char>(pixel);
        }
    }
}

extern "C" void applyConvolutionCUDA(const unsigned char* h_input, unsigned char* h_output,
                                       int width, int height, int channels,
                                       const float* h_kernel, int kernelSize) {
    unsigned char *d_input, *d_output;
    float *d_kernel;
    size_t imgBytes = width * height * channels * sizeof(unsigned char);
    size_t kernelBytes = kernelSize * kernelSize * sizeof(float);

    cudaCheckError(cudaMalloc(&d_input, imgBytes));
    cudaCheckError(cudaMalloc(&d_output, imgBytes));
    cudaCheckError(cudaMalloc(&d_kernel, kernelBytes));
    cudaCheckError(cudaMemcpy(d_input, h_input, imgBytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    convolutionKernel<<<grid, block>>>(d_input, d_output, width, height, channels, d_kernel, kernelSize);
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(h_output, d_output, imgBytes, cudaMemcpyDeviceToHost));
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

// Grayscale-to-Color Conversion using Pseudo‑Color Mapping Kernel
__global__ void grayscaleToPseudoColorKernel(const unsigned char* gray, unsigned char* color, const unsigned char* lut, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char intensity = gray[idx];
        int lutIdx = intensity * 3;
        int colorIdx = idx * 3;
        color[colorIdx]     = lut[lutIdx];      
        color[colorIdx + 1] = lut[lutIdx + 1];     
        color[colorIdx + 2] = lut[lutIdx + 2];   
    }
}

extern "C" void grayscaleToColorCUDA(const unsigned char* h_gray, unsigned char* h_color, int width, int height) {
    unsigned char *d_gray, *d_color, *d_lut;
    size_t grayBytes = width * height * sizeof(unsigned char);
    size_t colorBytes = width * height * 3 * sizeof(unsigned char);
    size_t lutBytes = 256 * 3 * sizeof(unsigned char);
    unsigned char h_lut[256 * 3];
    for (int i = 0; i < 256; i++) {
        float normalized = i / 255.0f;
        float r, g, b;
        if (normalized < 0.33f) {
            r = 0;
            g = 255 * (normalized / 0.33f);
            b = 255;
        } else if (normalized < 0.66f) {
            r = 255 * ((normalized - 0.33f) / 0.33f);
            g = 255;
            b = 255 * (1 - ((normalized - 0.33f) / 0.33f));
        } else {
            r = 255;
            g = 255 * (1 - ((normalized - 0.66f) / 0.34f));
            b = 0;
        }
        h_lut[i * 3]     = static_cast<unsigned char>(b);
        h_lut[i * 3 + 1] = static_cast<unsigned char>(g);
        h_lut[i * 3 + 2] = static_cast<unsigned char>(r);
    }

    cudaCheckError(cudaMalloc(&d_gray, grayBytes));
    cudaCheckError(cudaMalloc(&d_color, colorBytes));
    cudaCheckError(cudaMalloc(&d_lut, lutBytes));
    cudaCheckError(cudaMemcpy(d_gray, h_gray, grayBytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_lut, h_lut, lutBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    grayscaleToPseudoColorKernel<<<grid, block>>>(d_gray, d_color, d_lut, width, height);
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(h_color, d_color, colorBytes, cudaMemcpyDeviceToHost));
    cudaFree(d_gray);
    cudaFree(d_color);
    cudaFree(d_lut);
}