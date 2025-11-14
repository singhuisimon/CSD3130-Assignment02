#ifndef SEAM_CARVER_H
#define SEAM_CARVER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <limits>

/**
 * @brief SeamCarver class for content-aware image resizing
 *
 * Implements seam carving algorithm using:
 *  - Dynamic Programming (DP)
 *  - Greedy
 *  - Graph-based shortest path (Dijkstra on a pixel graph)
 */
class SeamCarver {
public:
    /**
     * @brief Construct from an image file on disk.
     */
    explicit SeamCarver(const std::string& imagePath);

    /**
     * @brief Compute gradient-based energy map for an image.
     * The result is CV_64F, same size as input.
     */
    cv::Mat calculateEnergy(const cv::Mat& img);

    // ----- DP seam finding -----

    /**
     * @brief Find minimal-energy vertical seam using DP.
     * @param energy CV_64F energy image
     * @return seam[row] = column index of seam pixel in that row
     */
    std::vector<int> findVerticalSeamDP(const cv::Mat& energy);

    /**
     * @brief Find minimal-energy horizontal seam using DP.
     * Implemented via transpose + vertical DP.
     * @param energy CV_64F energy image
     * @return seam[col] = row index of seam pixel in that column
     */
    std::vector<int> findHorizontalSeamDP(const cv::Mat& energy);

    // ----- Greedy seam finding -----

    /**
     * @brief Find a vertical seam using a simple greedy walk.
     */
    std::vector<int> findVerticalSeamGreedy(const cv::Mat& energy);

    /**
     * @brief Find a horizontal seam using greedy walk.
     */
    std::vector<int> findHorizontalSeamGreedy(const cv::Mat& energy);

    // ----- Graph-cut seam finding -----
    // Model the image as a layered graph and run Dijkstra to find
    // the minimum-cost s->t path; this is equivalent to computing a
    // seam via a generic graph shortest-path method rather than DP.

    /**
     * @brief Find vertical seam using a graph shortest-path formulation.
     * @param energy CV_64F energy image
     * @return seam[row] = column index
     */
    std::vector<int> findVerticalSeamGraphCut(const cv::Mat& energy);

    /**
     * @brief Find horizontal seam using graph formulation.
     * Implemented via transpose + vertical graph search.
     */
    std::vector<int> findHorizontalSeamGraphCut(const cv::Mat& energy);

    // ----- Image modification -----

    /**
     * @brief Remove a vertical seam from the given image.
     */
    cv::Mat removeVerticalSeam(const cv::Mat& img, const std::vector<int>& seam);

    /**
     * @brief Remove a horizontal seam from the given image.
     */
    cv::Mat removeHorizontalSeam(const cv::Mat& img, const std::vector<int>& seam);

    /**
     * @brief Resize the internal image using DP or greedy seams.
     * @param newWidth  desired width
     * @param newHeight desired height
     * @param useDP true = DP, false = greedy
     */
    cv::Mat resizeImage(int newWidth, int newHeight, bool useDP);

    /**
     * @brief Resize the internal image using the graph-based seam finder.
     * Only shrinking (newWidth <= originalWidth, newHeight <= originalHeight)
     * is supported.
     */
    cv::Mat resizeImageGraphCut(int newWidth, int newHeight);

    /**
     * @brief Visualize a single seam in red over the current image.
     * @param seam      seam indices
     * @param isVertical true for vertical (one column index per row),
     *                   false for horizontal (one row index per column)
     */
    cv::Mat visualizeSeam(const std::vector<int>& seam, bool isVertical);

    // @brief Get the current working image (after any resizing).
    cv::Mat getImage() const { return image.clone(); }

    // @brief Get the original, unmodified image.
    cv::Mat getOriginalImage() const { return originalImage.clone(); }

private:
    cv::Mat image;          // Current working image
    cv::Mat originalImage;  // Original image (preserved)
};

#endif // SEAM_CARVER_H
