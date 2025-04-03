#ifndef CUDA_OPERATIONS_H
#define CUDA_OPERATIONS_H

#ifdef __cplusplus
extern "C" {
#endif

void convertToGrayscaleCUDA(const unsigned char* h_input, unsigned char* h_output, int width, int height, int channels);

void resizeImageCUDA(const unsigned char* h_input, unsigned char* h_output,
                     int inWidth, int inHeight, int channels,
                     int outWidth, int outHeight);

void applyConvolutionCUDA(const unsigned char* h_input, unsigned char* h_output,
                          int width, int height, int channels,
                          const float* h_kernel, int kernelSize);

void grayscaleToColorCUDA(const unsigned char* h_gray, unsigned char* h_color, int width, int height);


void compressImageCUDA(const unsigned char* h_input, unsigned char* h_output,
    int inWidth, int inHeight, int channels,
    int outWidth, int outHeight);

void applyEdgeDetectionCUDA(const unsigned char* h_input, unsigned char* h_output,
    int width, int height, int channels,
    const float* h_kernel, int kernelSize);



#ifdef __cplusplus
}
#endif

#endif