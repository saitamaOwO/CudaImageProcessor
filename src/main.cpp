#include <opencv2/opencv.hpp>
#include <iostream>
#include "cudaOperations.h"

using namespace std;

cv::Mat g_storedColor;

void processGrayscale(const string &inputPath, const string &outputPath) {
    cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
    if(image.empty()) {
         cerr << "Error loading image." << endl;
         return;
    }
    g_storedColor = image.clone();

    cv::Mat result(image.rows, image.cols, CV_8UC1);
    convertToGrayscaleCUDA(image.data, result.data, image.cols, image.rows, image.channels());
    if(cv::imwrite(outputPath, result))
         cout << "Grayscale image saved to " << outputPath << endl;
    else
         cerr << "Error saving image." << endl;
}

void processResize(const string &inputPath, const string &outputPath, int newWidth, int newHeight) {
    cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
    if(image.empty()){
         cerr << "Error loading image." << endl;
         return;
    }
    cv::Mat result(newHeight, newWidth, image.type());
    resizeImageCUDA(image.data, result.data, image.cols, image.rows, image.channels(), newWidth, newHeight);
    if(cv::imwrite(outputPath, result))
         cout << "Resized image saved to " << outputPath << endl;
    else
         cerr << "Error saving image." << endl;
}
void processCompression(const string &inputPath, const string &outputPath, int quality) {
    cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error loading image." << endl;
        return;
    }
    
    vector<int> compressionParams;
    compressionParams.push_back(cv::IMWRITE_JPEG_QUALITY);
    compressionParams.push_back(quality);

    if (cv::imwrite(outputPath, image, compressionParams))
        cout << "Compressed image saved to " << outputPath << " with quality: " << quality << endl;
    else
        cerr << "Error saving compressed image." << endl;
}

void processEdgeShapeDetection(const string &inputPath, const string &outputPath, int method) {
    cv::Mat image = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error loading image." << endl;
        return;
    }
    
    cv::Mat result;
    if (method == 1) { // Sobel
        float sobelX[9] = { -1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1 };
        float sobelY[9] = { -1, -2, -1,
                             0,  0,  0,
                             1,  2,  1 };
        cv::Mat gradX(image.size(), image.type());
        cv::Mat gradY(image.size(), image.type());
        applyConvolutionCUDA(image.data, gradX.data, image.cols, image.rows, 1, sobelX, 3);
        applyConvolutionCUDA(image.data, gradY.data, image.cols, image.rows, 1, sobelY, 3);
        result.create(image.rows, image.cols, image.type());
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                int gx = gradX.at<uchar>(y, x);
                int gy = gradY.at<uchar>(y, x);
                int mag = static_cast<int>(sqrt(gx * gx + gy * gy));
                result.at<uchar>(y, x) = static_cast<uchar>(min(mag, 255));
            }
        }
    } else if (method == 2) { // Prewitt
        float prewittX[9] = { -1, 0, 1,
                              -1, 0, 1,
                              -1, 0, 1 };
        float prewittY[9] = { -1, -1, -1,
                               0,  0,  0,
                               1,  1,  1 };
        cv::Mat gradX(image.size(), image.type());
        cv::Mat gradY(image.size(), image.type());
        applyConvolutionCUDA(image.data, gradX.data, image.cols, image.rows, 1, prewittX, 3);
        applyConvolutionCUDA(image.data, gradY.data, image.cols, image.rows, 1, prewittY, 3);
        result.create(image.rows, image.cols, image.type());
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                int gx = gradX.at<uchar>(y, x);
                int gy = gradY.at<uchar>(y, x);
                int mag = static_cast<int>(sqrt(gx * gx + gy * gy));
                result.at<uchar>(y, x) = static_cast<uchar>(min(mag, 255));
            }
        }
    } else if (method == 3) { // Laplacian
        float laplacian[9] = { 0,  1, 0,
                               1, -4, 1,
                               0,  1, 0 };
        result.create(image.rows, image.cols, image.type());
        applyConvolutionCUDA(image.data, result.data, image.cols, image.rows, 1, laplacian, 3);
    } else {
        cerr << "Invalid edge detection method." << endl;
        return;
    }
    
    if (cv::imwrite(outputPath, result))
        cout << "Edge detection processed image saved to " << outputPath << endl;
    else
        cerr << "Error saving image." << endl;
}

void revertGrayscaleToColor(const string &inputPath, const string &outputPath) {
    if (!g_storedColor.empty()) {
         if (cv::imwrite(outputPath, g_storedColor))
              cout << "Restored original color image saved to " << outputPath << endl;
         else
              cerr << "Error saving restored color image." << endl;
    } else {
         cerr << "No original color image stored. Cannot restore." << endl;
    }
}

int main() {
    int mainChoice;
    while (true) {
         cout << "\n--- CUDA Image Processing Menu ---" << endl;
         cout << "1: Compression" << endl;
         cout << "2: Convert to Grayscale" << endl;
         cout << "3: Resize Image" << endl;
         cout << "4: Convolution Operation (Edge & Shape Detection)" << endl;
         cout << "5: Restore Original" << endl;
         cout << "0: Exit" << endl;
         cout << "Enter your choice: ";
         cin >> mainChoice;
         if (mainChoice == 0) break;

         string inputPath, outputPath;
         cout << "Enter input image file path: ";
         cin >> inputPath;
         cout << "Enter output image file path: ";
         cin >> outputPath;

         if (mainChoice == 1) {
            int quality;
            cout << "Enter compression quality (0-100): ";
            cin >> quality;
            processCompression(inputPath, outputPath, quality);
        }
         if (mainChoice == 2) {
              processGrayscale(inputPath, outputPath);
         } else if (mainChoice == 3) {
              int newWidth, newHeight;
              cout << "Enter new width: ";
              cin >> newWidth;
              cout << "Enter new height: ";
              cin >> newHeight;
              processResize(inputPath, outputPath, newWidth, newHeight);
         } else if (mainChoice == 4) {
                  int method;
                  cout << "\nEdge & Shape Detection Options:" << endl;
                  cout << "1: Sobel" << endl;
                  cout << "2: Prewitt" << endl;
                  cout << "3: Laplacian" << endl;
                  cout << "Enter your choice: ";
                  cin >> method;
                  processEdgeShapeDetection(inputPath, outputPath, method);
         } else if (mainChoice == 5) {
              revertGrayscaleToColor(inputPath, outputPath);
         } else {
              cout << "Invalid option. Try again." << endl;
         }
    }
    return 0;
}