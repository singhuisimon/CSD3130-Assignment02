#include "SeamCarver.h"
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * @brief Print usage instructions
 */
void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <image_path> [width_%] [height_%] [method]" << std::endl;
    std::cout << "  image_path: Path to input image (required)" << std::endl;
    std::cout << "  width_%:    Target width as percentage 1-100 (optional, default: 80)" << std::endl;
    std::cout << "  height_%:   Target height as percentage 1-100 (optional, default: 80)" << std::endl;
    std::cout << "  method:     'dp' for Dynamic Programming or 'greedy' (optional, default: dp)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " image.jpg                # Resize to 80% width, 80% height" << std::endl;
    std::cout << "  " << programName << " image.jpg 70 60          # Resize to 70% width, 60% height" << std::endl;
    std::cout << "  " << programName << " image.jpg 75 75 greedy   # Resize to 75% using Greedy" << std::endl;
    std::cout << "  " << programName << " image.jpg 90 100         # Resize to 90% width, keep height" << std::endl;
}

/**
 * @brief Create output directory if it doesn't exist
 * @param outputDir Path to output directory
 * @return true if directory exists or was created successfully
 */
bool ensureOutputDirectory(const std::string& outputDir) {
    try {
        if (!fs::exists(outputDir)) {
            fs::create_directories(outputDir);
            std::cout << "Created output directory: " << outputDir << std::endl;
        }
        return true;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating output directory: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Validate percentage value
 * @param value Percentage value to validate
 * @param name Name of the parameter (for error messages)
 * @return true if valid (1-100)
 */
bool validatePercentage(double value, const std::string& name) {
    if (value < 1.0 || value > 100.0) {
        std::cerr << "Error: " << name << " must be between 1 and 100 (got " << value << ")" << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Seam Carving - Image Resizing" << std::endl;
    std::cout << "  Percentage-Based Dimensions" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Check arguments
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string imagePath = argv[1];
    
    // Define output directory
    std::string outputDir = "output";
    
    try {
        // Create output directory if it doesn't exist
        if (!ensureOutputDirectory(outputDir)) {
            std::cerr << "Failed to create output directory. Using current directory." << std::endl;
            outputDir = ".";  // Fallback to current directory
        }
        
        // Initialize seam carver
        SeamCarver carver(imagePath);
        cv::Mat originalImage = carver.getOriginalImage();
        int originalWidth = originalImage.cols;
        int originalHeight = originalImage.rows;
        
        // Parse percentage arguments (default: 80%)
        double widthPercent = (argc > 2) ? std::stod(argv[2]) : 80.0;
        double heightPercent = (argc > 3) ? std::stod(argv[3]) : 80.0;
        std::string method = (argc > 4) ? argv[4] : "dp";
        bool useDP = (method == "dp" || method == "DP");
        
        // Validate percentages
        if (!validatePercentage(widthPercent, "Width percentage")) {
            return 1;
        }
        if (!validatePercentage(heightPercent, "Height percentage")) {
            return 1;
        }
        
        // Calculate actual dimensions from percentages
        int newWidth = static_cast<int>(originalWidth * widthPercent / 100.0);
        int newHeight = static_cast<int>(originalHeight * heightPercent / 100.0);
        
        // Display resize information
        std::cout << "Original dimensions: " << originalWidth << "x" << originalHeight << std::endl;
        std::cout << "Target percentages:  " << widthPercent << "% x " << heightPercent << "%" << std::endl;
        std::cout << "Calculated dimensions: " << newWidth << "x" << newHeight << std::endl;
        std::cout << std::endl;
        
        // Validate calculated dimensions
        if (newWidth > originalWidth || newHeight > originalHeight) {
            std::cerr << "Error: Percentages > 100% not supported (image expansion)." << std::endl;
            return 1;
        }
        
        if (newWidth <= 0 || newHeight <= 0) {
            std::cerr << "Error: Calculated dimensions are too small." << std::endl;
            return 1;
        }
        
        if (newWidth == originalWidth && newHeight == originalHeight) {
            std::cout << "Note: Target dimensions equal original. No resizing needed." << std::endl;
            return 0;
        }
        
        // Time the resize operation
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Resize using specified method
        cv::Mat resizedImage = carver.resizeImage(newWidth, newHeight, useDP);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "\nProcessing time: " << duration.count() << " ms" << std::endl;
        
        // Build output file paths with percentage in filename
        std::string methodStr = useDP ? "dp" : "greedy";
        std::string outputFilename = outputDir + "/output_" + methodStr + "_" + 
                                    std::to_string(static_cast<int>(widthPercent)) + "w_" + 
                                    std::to_string(static_cast<int>(heightPercent)) + "h_" +
                                    std::to_string(newWidth) + "x" + 
                                    std::to_string(newHeight) + ".png";
        
        // Save resized image
        cv::imwrite(outputFilename, resizedImage);
        std::cout << "Saved resized image to: " << outputFilename << std::endl;
        
        // Create comparison image (standard resize vs seam carving)
        cv::Mat standardResize;
        cv::resize(originalImage, standardResize, cv::Size(newWidth, newHeight));
        
        cv::Mat comparison;
        cv::hconcat(standardResize, resizedImage, comparison);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Success! Files saved to '" << outputDir << "/' folder:" << std::endl;
        std::cout << "  - " << fs::path(outputFilename).filename().string() << std::endl;
        // std::cout << "  - " << fs::path(comparisonFilename).filename().string() << std::endl;
        std::cout << "========================================" << std::endl;
        
    } catch (const std::invalid_argument&) {
        // Removed 'e' - not used in error message
        std::cerr << "Error: Invalid percentage value. Please use numbers between 1-100." << std::endl;
        return 1;
    } catch (const std::out_of_range&) {
        // Removed 'e' - not used in error message
        std::cerr << "Error: Percentage value out of range." << std::endl;
        return 1;
    } catch (const std::exception& e) {
        // Keep 'e' here - it's used in error message
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}