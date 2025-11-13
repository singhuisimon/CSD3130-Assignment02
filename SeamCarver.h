#ifndef SEAM_CARVER_H
#define SEAM_CARVER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <limits>

/**
 * @brief SeamCarver class for content-aware image resizing
 * 
 * Implements seam carving algorithm using both Dynamic Programming
 * and Greedy approaches for comparison.
 */
class SeamCarver {
public:
    /**
     * @brief Construct a new Seam Carver object
     * @param imagePath Path to the input image
     */
    explicit SeamCarver(const std::string& imagePath);

    /**
     * @brief Calculate energy map using gradient magnitude (Sobel filter)
     * @param image Input image
     * @return Energy map as CV_64F matrix
     */
    cv::Mat calculateEnergy(const cv::Mat& image);

    /**
     * @brief Find minimum energy vertical seam using Dynamic Programming
     * @param energy Energy map
     * @return Vector of column indices (one per row)
     */
    std::vector<int> findVerticalSeamDP(const cv::Mat& energy);

    /**
     * @brief Find vertical seam using Greedy Algorithm
     * @param energy Energy map
     * @return Vector of column indices (one per row)
     */
    std::vector<int> findVerticalSeamGreedy(const cv::Mat& energy);

    /**
     * @brief Find minimum energy horizontal seam using Dynamic Programming
     * @param energy Energy map
     * @return Vector of row indices (one per column)
     */
    std::vector<int> findHorizontalSeamDP(const cv::Mat& energy);

    /**
     * @brief Find horizontal seam using Greedy Algorithm
     * @param energy Energy map
     * @return Vector of row indices (one per column)
     */
    std::vector<int> findHorizontalSeamGreedy(const cv::Mat& energy);

    /**
     * @brief Remove a vertical seam from the image
     * @param image Input image
     * @param seam Vector of column indices
     * @return Image with seam removed (width reduced by 1)
     */
    cv::Mat removeVerticalSeam(const cv::Mat& image, const std::vector<int>& seam);

    /**
     * @brief Remove a horizontal seam from the image
     * @param image Input image
     * @param seam Vector of row indices
     * @return Image with seam removed (height reduced by 1)
     */
    cv::Mat removeHorizontalSeam(const cv::Mat& image, const std::vector<int>& seam);

    /**
     * @brief Resize image to new dimensions using seam carving
     * @param newWidth Target width
     * @param newHeight Target height
     * @param useDP Use Dynamic Programming (true) or Greedy (false)
     * @return Resized image
     */
    cv::Mat resizeImage(int newWidth, int newHeight, bool useDP = true);

    /**
     * @brief Visualize a seam on the image
     * @param seam Vector of indices
     * @param isVertical True for vertical seam, false for horizontal
     * @return Image with seam highlighted in red
     */
    cv::Mat visualizeSeam(const std::vector<int>& seam, bool isVertical = true);

    /**
     * @brief Get the current image
     * @return Current image
     */
    cv::Mat getImage() const { return image.clone(); }

    /**
     * @brief Get the original image
     * @return Original image
     */
    cv::Mat getOriginalImage() const { return originalImage.clone(); }

private:
    cv::Mat image;          ///< Current working image
    cv::Mat originalImage;  ///< Original image (preserved)
};

#endif // SEAM_CARVER_H