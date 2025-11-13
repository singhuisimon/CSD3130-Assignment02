#include "SeamCarver.h"
#include <iostream>
#include <algorithm>
#include <cmath>

SeamCarver::SeamCarver(const std::string& imagePath) {
    image = cv::imread(imagePath);
    if (image.empty()) {
        throw std::runtime_error("Could not load image from: " + imagePath);
    }
    originalImage = image.clone();
    std::cout << "Loaded image with dimensions: " << image.cols << "x" << image.rows << std::endl;
}

cv::Mat SeamCarver::calculateEnergy(const cv::Mat& img) {
    // Convert to grayscale for energy calculation
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    // Convert to double precision for calculation
    gray.convertTo(gray, CV_64F);
    
    // Apply Sobel filter to get gradients in x and y directions
    cv::Mat sobelX, sobelY;
    cv::Sobel(gray, sobelX, CV_64F, 1, 0, 3);  // Gradient in X direction
    cv::Sobel(gray, sobelY, CV_64F, 0, 1, 3);  // Gradient in Y direction
    
    // Calculate gradient magnitude as energy
    cv::Mat energy;
    cv::magnitude(sobelX, sobelY, energy);
    
    return energy;
}

std::vector<int> SeamCarver::findVerticalSeamDP(const cv::Mat& energy) {
    int rows = energy.rows;
    int cols = energy.cols;
    
    // Create DP table
    cv::Mat dp = cv::Mat::zeros(rows, cols, CV_64F);
    
    // Initialize first row
    for (int j = 0; j < cols; j++) {
        dp.at<double>(0, j) = energy.at<double>(0, j);
    }
    
    // Fill the DP table row by row
    for (int i = 1; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double currentEnergy = energy.at<double>(i, j);
            
            // Find minimum of three possible previous pixels
            double minPrevEnergy = dp.at<double>(i-1, j);  // Directly above
            
            // Diagonal-left (if exists)
            if (j > 0) {
                minPrevEnergy = std::min(minPrevEnergy, dp.at<double>(i-1, j-1));
            }
            
            // Diagonal-right (if exists)
            if (j < cols - 1) {
                minPrevEnergy = std::min(minPrevEnergy, dp.at<double>(i-1, j+1));
            }
            
            dp.at<double>(i, j) = currentEnergy + minPrevEnergy;
        }
    }
    
    // Backtrack to find the seam path
    std::vector<int> seam(rows);
    
    // Start from minimum energy pixel in last row
    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(dp.row(rows-1), &minVal, nullptr, &minLoc, nullptr);
    int j = minLoc.x;
    seam[rows-1] = j;
    
    // CORRECTED: Backtrack from bottom to top
    // At each step, find which parent in the previous row led to current position
    for (int i = rows - 2; i >= 0; i--) {
        // We're at column j in row i+1
        // Find which column in row i could have led here
        // Candidates: j-1 (moved diagonal-right), j (moved down), j+1 (moved diagonal-left)
        
        int bestJ = j;  // Default: came from directly above
        double bestEnergy = dp.at<double>(i, j);
        
        // Check if coming from diagonal-left (j-1) is better
        if (j > 0) {
            double leftEnergy = dp.at<double>(i, j-1);
            if (leftEnergy < bestEnergy) {
                bestEnergy = leftEnergy;
                bestJ = j - 1;
            }
        }
        
        // Check if coming from diagonal-right (j+1) is better
        if (j < cols - 1) {
            double rightEnergy = dp.at<double>(i, j+1);
            if (rightEnergy < bestEnergy) {
                bestEnergy = rightEnergy;
                bestJ = j + 1;
            }
        }
        
        j = bestJ;
        seam[i] = j;
    }
    
    return seam;
}

std::vector<int> SeamCarver::findVerticalSeamGreedy(const cv::Mat& energy) {
    int rows = energy.rows;
    int cols = energy.cols;
    
    std::vector<int> seam(rows);
    
    // Start from minimum energy pixel in first row
    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(energy.row(0), &minVal, nullptr, &minLoc, nullptr);
    int j = minLoc.x;
    seam[0] = j;
    
    // Greedy selection: at each row, choose minimum among 3 neighbors
    for (int i = 1; i < rows; i++) {
        double minEnergy = energy.at<double>(i, j);
        int minJ = j;
        
        // Check diagonal-left
        if (j > 0) {
            double leftEnergy = energy.at<double>(i, j-1);
            if (leftEnergy < minEnergy) {
                minEnergy = leftEnergy;
                minJ = j - 1;
            }
        }
        
        // Check diagonal-right
        if (j < cols - 1) {
            double rightEnergy = energy.at<double>(i, j+1);
            if (rightEnergy < minEnergy) {
                minEnergy = rightEnergy;
                minJ = j + 1;
            }
        }
        
        j = minJ;
        seam[i] = j;
    }
    
    return seam;
}

std::vector<int> SeamCarver::findHorizontalSeamDP(const cv::Mat& energy) {
    // Transpose energy map, find vertical seam, return as horizontal
    cv::Mat transposedEnergy;
    cv::transpose(energy, transposedEnergy);
    return findVerticalSeamDP(transposedEnergy);
}

std::vector<int> SeamCarver::findHorizontalSeamGreedy(const cv::Mat& energy) {
    // Transpose energy map, find vertical seam, return as horizontal
    cv::Mat transposedEnergy;
    cv::transpose(energy, transposedEnergy);
    return findVerticalSeamGreedy(transposedEnergy);
}

cv::Mat SeamCarver::removeVerticalSeam(const cv::Mat& img, const std::vector<int>& seam) {
    int rows = img.rows;
    int cols = img.cols;
    
    // Create new image with one less column
    cv::Mat newImage(rows, cols - 1, img.type());
    
    for (int i = 0; i < rows; i++) {
        int seamCol = seam[i];
        
        // Copy pixels before the seam
        if (seamCol > 0) {
            img.row(i).colRange(0, seamCol).copyTo(newImage.row(i).colRange(0, seamCol));
        }
        
        // Copy pixels after the seam
        if (seamCol < cols - 1) {
            img.row(i).colRange(seamCol + 1, cols).copyTo(
                newImage.row(i).colRange(seamCol, cols - 1));
        }
    }
    
    return newImage;
}

cv::Mat SeamCarver::removeHorizontalSeam(const cv::Mat& img, const std::vector<int>& seam) {
    int rows = img.rows;
    int cols = img.cols;
    
    // Create new image with one less row
    cv::Mat newImage(rows - 1, cols, img.type());
    
    for (int j = 0; j < cols; j++) {
        int seamRow = seam[j];
        
        // Copy pixels before the seam
        if (seamRow > 0) {
            img.col(j).rowRange(0, seamRow).copyTo(newImage.col(j).rowRange(0, seamRow));
        }
        
        // Copy pixels after the seam
        if (seamRow < rows - 1) {
            img.col(j).rowRange(seamRow + 1, rows).copyTo(
                newImage.col(j).rowRange(seamRow, rows - 1));
        }
    }
    
    return newImage;
}

cv::Mat SeamCarver::resizeImage(int newWidth, int newHeight, bool useDP) {
    cv::Mat currentImage = image.clone();
    int currentHeight = currentImage.rows;
    int currentWidth = currentImage.cols;
    
    std::cout << "Resizing from (" << currentWidth << "x" << currentHeight 
              << ") to (" << newWidth << "x" << newHeight << ")" << std::endl;
    std::cout << "Using method: " << (useDP ? "Dynamic Programming" : "Greedy") << std::endl;
    
    // Remove vertical seams (reduce width)
    int numVerticalSeams = currentWidth - newWidth;
    if (numVerticalSeams > 0) {
        std::cout << "Removing " << numVerticalSeams << " vertical seams..." << std::endl;
        for (int i = 0; i < numVerticalSeams; i++) {
            cv::Mat energy = calculateEnergy(currentImage);
            
            std::vector<int> seam;
            if (useDP) {
                seam = findVerticalSeamDP(energy);
            } else {
                seam = findVerticalSeamGreedy(energy);
            }
            
            currentImage = removeVerticalSeam(currentImage, seam);
            
            if ((i + 1) % 10 == 0 || (i + 1) == numVerticalSeams) {
                std::cout << "  Removed " << (i + 1) << "/" << numVerticalSeams 
                          << " vertical seams" << std::endl;
            }
        }
    }
    
    // Remove horizontal seams (reduce height)
    int numHorizontalSeams = currentHeight - newHeight;
    if (numHorizontalSeams > 0) {
        std::cout << "Removing " << numHorizontalSeams << " horizontal seams..." << std::endl;
        for (int i = 0; i < numHorizontalSeams; i++) {
            cv::Mat energy = calculateEnergy(currentImage);
            
            std::vector<int> seam;
            if (useDP) {
                seam = findHorizontalSeamDP(energy);
            } else {
                seam = findHorizontalSeamGreedy(energy);
            }
            
            currentImage = removeHorizontalSeam(currentImage, seam);
            
            if ((i + 1) % 10 == 0 || (i + 1) == numHorizontalSeams) {
                std::cout << "  Removed " << (i + 1) << "/" << numHorizontalSeams 
                          << " horizontal seams" << std::endl;
            }
        }
    }
    
    std::cout << "Resizing complete!" << std::endl;
    image = currentImage;
    return currentImage;
}

cv::Mat SeamCarver::visualizeSeam(const std::vector<int>& seam, bool isVertical) {
    cv::Mat visImage = image.clone();
    
    if (isVertical) {
        for (int i = 0; i < static_cast<int>(seam.size()); i++) {
            int j = seam[i];
            if (j >= 0 && j < visImage.cols && i >= 0 && i < visImage.rows) {
                visImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);  // Red color in BGR
            }
        }
    } else {
        for (int j = 0; j < static_cast<int>(seam.size()); j++) {
            int i = seam[j];
            if (i >= 0 && i < visImage.rows && j >= 0 && j < visImage.cols) {
                visImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);  // Red color in BGR
            }
        }
    }
    
    return visImage;
}