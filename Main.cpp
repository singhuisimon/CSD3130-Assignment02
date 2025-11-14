#include "SeamCarver.h"
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <memory>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>

// ---- GUI / OpenGL / ImGui ----
#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl2.h"

namespace fs = std::filesystem;

// For OpenGL texture output
struct ImageTexture {
    GLuint id = 0;
    int width = 0;
    int height = 0;

    void destroy() {
        if (id != 0) {
            glDeleteTextures(1, &id);
            id = 0;
        }
        width = height = 0;
    }
};

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
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating output directory: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Convert cv::Mat (BGR/GRAY/BGRA) to an OpenGL RGBA texture
 * @param img Image in the form of cv::Mat
 * @param outTex Output texture in OpenGL format
 * @return true if successful conversion of cv::Mat to OpenGL RGBA texture
 */
bool LoadTextureFromMat(const cv::Mat& img, ImageTexture& outTex) {
    if (img.empty()) return false;

    cv::Mat rgba;
    switch (img.channels()) {
    case 1:  cv::cvtColor(img, rgba, cv::COLOR_GRAY2RGBA); break;
    case 3:  cv::cvtColor(img, rgba, cv::COLOR_BGR2RGBA);  break;
    case 4:  cv::cvtColor(img, rgba, cv::COLOR_BGRA2RGBA); break;
    default:
        std::cerr << "Unsupported number of channels: " << img.channels() << "\n";
        return false;
    }

    GLuint texId;
    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        rgba.cols,
        rgba.rows,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        rgba.data
    );

    outTex.destroy();
    outTex.id = texId;
    outTex.width = rgba.cols;
    outTex.height = rgba.rows;
    return true;
}

/**
 * @brief Draws a red seam on a copy of a base image to visualize a seam step
 * @param baseImage Base image being resized
 * @param seam The seam data to be drawn on the image
 * @param isVertical Flag for checking if seam is vertical
 * @return a copy of the base image with the red seam drawn over it
 */
cv::Mat drawSeamOnImage(const cv::Mat& baseImage,
                        const std::vector<int>& seam,
                        bool isVertical) {
    cv::Mat vis = baseImage.clone();
    if (vis.empty() || seam.empty())
        return vis;

    if (isVertical) {
        // seam[i] is column for row i
        for (int i = 0; i < (int)seam.size(); ++i) {
            int j = seam[i];
            if (i >= 0 && i < vis.rows && j >= 0 && j < vis.cols) {
                vis.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255); // BGR red
            }
        }
    }
    else {
        // seam[j] is row for column j
        for (int j = 0; j < (int)seam.size(); ++j) {
            int i = seam[j];
            if (i >= 0 && i < vis.rows && j >= 0 && j < vis.cols) {
                vis.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            }
        }
    }

    return vis;
}

/**
 * @brief Runs the GUI for the seam carving operations
 */
int run_gui() {
    // Init GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return 1;
    }

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Seam Carving GUI", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // Setup Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Set up docking
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL2_Init();

    // GUI state
    static char imagePath[512] = "test.jpg";

    // Current interactive view
    ImageTexture imgTex;            

    bool imageLoaded = false;
    std::string lastError;

    std::unique_ptr<SeamCarver> carver;
    cv::Mat currentImage;           // working image (after seam removals)
    int seamsRemoved = 0;

    // Original dimensions (for sliders)
    int originalWidth = 0;
    int originalHeight = 0;
    int targetWidth = 0;
    int targetHeight = 0;
    float targetWidthPercent = 100.0f;
    float targetHeightPercent = 100.0f;

    // Options
    bool useVerticalForStep = true;   // direction for the manual "Step" button

    // Method selection: 0 = DP, 1 = Greedy, 2 = Graph
    int methodIndex = 0;
    const char* methodNames[] = { "DP", "Greedy", "Graph (Dijkstra)" };

    // Auto-run flags
    bool autoRunVertical = false;
    bool autoRunHorizontal = false;
    bool autoRunFull = false;

    // Stats / status
    bool hasResizeStats = false;
    long long lastProcessingMs = 0;
    int lastResizedWidth = 0;
    int lastResizedHeight = 0;
    int lastMethodIndex = 0;
    bool fullResizeRunning = false;
    std::chrono::high_resolution_clock::time_point resizeStartTime;

    std::string guiStatusMessage;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Global dockspace over main viewport
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

        // --------------------------------------------------------------------
        // Controls window
        // --------------------------------------------------------------------
        ImGui::Begin("Controls");
        ImGui::InputText("Image path", imagePath, IM_ARRAYSIZE(imagePath));

        if (ImGui::Button("Load image")) {
            try {
                carver = std::make_unique<SeamCarver>(imagePath);
                currentImage = carver->getOriginalImage().clone();
                seamsRemoved = 0;

                originalWidth = currentImage.cols;
                originalHeight = currentImage.rows;
                targetWidth = originalWidth;
                targetHeight = originalHeight;
                targetWidthPercent = 100.0f;
                targetHeightPercent = 100.0f;

                autoRunVertical = false;
                autoRunHorizontal = false;
                autoRunFull = false;
                fullResizeRunning = false;
                hasResizeStats = false;
                guiStatusMessage.clear();

                if (!LoadTextureFromMat(currentImage, imgTex)) {
                    lastError = "Failed to upload texture from loaded image.";
                    imageLoaded = false;
                }
                else {
                    lastError.clear();
                    imageLoaded = true;
                }
            }
            catch (const std::exception& e) {
                lastError = e.what();
                imageLoaded = false;
                carver.reset();
            }
        }

        if (!lastError.empty()) {
            ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "%s", lastError.c_str());
        }

        if (imageLoaded && carver) {
            ImGui::Separator();
            ImGui::Text("Current size: %d x %d", currentImage.cols, currentImage.rows);
            ImGui::Text("Original:     %d x %d", originalWidth, originalHeight);

            // Target width: slider (px) + input (percent)
            ImGui::Text("Target width");
            ImGui::PushID("target_width");
            bool widthChangedSlider = ImGui::SliderInt("px", &targetWidth, 1, originalWidth);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(90.0f);
            bool widthChangedPercent = ImGui::InputFloat("%", &targetWidthPercent, 1.0f, 5.0f, "%.1f");
            ImGui::PopID();

            if (widthChangedSlider) {
                targetWidth = std::max(1, std::min(targetWidth, originalWidth));
                targetWidthPercent = 100.0f * targetWidth / (float)originalWidth;
            }
            if (widthChangedPercent) {
                if (targetWidthPercent < 1.0f) targetWidthPercent = 1.0f;
                if (targetWidthPercent > 100.0f) targetWidthPercent = 100.0f;
                targetWidth = (int)std::round(originalWidth * targetWidthPercent / 100.0f);
                targetWidth = std::max(1, std::min(targetWidth, originalWidth));
            }

            // Target height: slider (px) + input (percent)
            ImGui::Text("Target height");
            ImGui::PushID("target_height");
            bool heightChangedSlider = ImGui::SliderInt("px", &targetHeight, 1, originalHeight);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(90.0f);
            bool heightChangedPercent = ImGui::InputFloat("%", &targetHeightPercent, 1.0f, 5.0f, "%.1f");
            ImGui::PopID();

            if (heightChangedSlider) {
                targetHeight = std::max(1, std::min(targetHeight, originalHeight));
                targetHeightPercent = 100.0f * targetHeight / (float)originalHeight;
            }
            if (heightChangedPercent) {
                if (targetHeightPercent < 1.0f) targetHeightPercent = 1.0f;
                if (targetHeightPercent > 100.0f) targetHeightPercent = 100.0f;
                targetHeight = (int)std::round(originalHeight * targetHeightPercent / 100.0f);
                targetHeight = std::max(1, std::min(targetHeight, originalHeight));
            }

            ImGui::Text("Target %%: %.1f%% x %.1f%%  -> %d x %d",
                targetWidthPercent, targetHeightPercent,
                targetWidth, targetHeight);

            ImGui::Separator();
            ImGui::Text("Seam method:");
            ImGui::Combo("Method", &methodIndex, methodNames, IM_ARRAYSIZE(methodNames));

            ImGui::Text("Direction for Step: %s",
                useVerticalForStep ? "Vertical (width)" : "Horizontal (height)");
            ImGui::SameLine();
            if (ImGui::Button("Toggle Step Direction")) {
                useVerticalForStep = !useVerticalForStep;
            }

            ImGui::Text("Seams removed: %d", seamsRemoved);

            // Helper: one seam step either vertical or horizontal
            auto stepSeamOnce = [&](bool vertical) -> bool {
                if (!carver || currentImage.empty())
                    return false;

                // Stop conditions
                if (vertical) {
                    if (currentImage.cols <= targetWidth)
                        return false;
                }
                else {
                    if (currentImage.rows <= targetHeight)
                        return false;
                }

                try {
                    cv::Mat energy = carver->calculateEnergy(currentImage);
                    std::vector<int> seam;

                    // Pick method
                    switch (methodIndex) {
                    case 0: // DP
                        seam = vertical
                            ? carver->findVerticalSeamDP(energy)
                            : carver->findHorizontalSeamDP(energy);
                        break;
                    case 1: // Greedy
                        seam = vertical
                            ? carver->findVerticalSeamGreedy(energy)
                            : carver->findHorizontalSeamGreedy(energy);
                        break;
                    case 2: // Graph
                    default:
                        seam = vertical
                            ? carver->findVerticalSeamGraphCut(energy)
                            : carver->findHorizontalSeamGraphCut(energy);
                        break;
                    }

                    if (seam.empty())
                        return false;

                    // Visualize seam on current image
                    cv::Mat vis = drawSeamOnImage(currentImage, seam, vertical);
                    LoadTextureFromMat(vis, imgTex);

                    // Remove the seam for the next step
                    if (vertical) {
                        currentImage = carver->removeVerticalSeam(currentImage, seam);
                    }
                    else {
                        currentImage = carver->removeHorizontalSeam(currentImage, seam);
                    }

                    seamsRemoved++;
                    return true;
                }
                catch (const std::exception& e) {
                    lastError = e.what();
                    return false;
                }
                };

            // Manual step (uses chosen direction & method)
            if (ImGui::Button("Step: show & remove next seam")) {
                autoRunVertical = false;
                autoRunHorizontal = false;
                autoRunFull = false;
                fullResizeRunning = false;
                stepSeamOnce(useVerticalForStep);
            }

            // Run Vertical (auto)
            if (ImGui::Button("Run Vertical")) {
                autoRunVertical = !autoRunVertical; // toggle
                autoRunHorizontal = false;
                autoRunFull = false;
                fullResizeRunning = false;
            }
            ImGui::SameLine();
            ImGui::Text(autoRunVertical ? "[Vertical running]" : "");

            // Run Horizontal (auto)
            if (ImGui::Button("Run Horizontal")) {
                autoRunHorizontal = !autoRunHorizontal; // toggle
                autoRunVertical = false;
                autoRunFull = false;
                fullResizeRunning = false;
            }
            ImGui::SameLine();
            ImGui::Text(autoRunHorizontal ? "[Horizontal running]" : "");

            // Run Full (vertical then horizontal)
            if (ImGui::Button("Run Full")) {
                if (!autoRunFull) {
                    autoRunFull = true;
                    autoRunVertical = false;
                    autoRunHorizontal = false;
                    fullResizeRunning = true;
                    resizeStartTime = std::chrono::high_resolution_clock::now();
                    hasResizeStats = false;
                    guiStatusMessage.clear();
                }
                else {
                    autoRunFull = false;
                    fullResizeRunning = false;
                }
            }
            ImGui::SameLine();
            ImGui::Text(autoRunFull ? "[Full running]" : "");

            // Reset
            if (ImGui::Button("Reset image")) {
                if (carver) {
                    currentImage = carver->getOriginalImage().clone();
                    seamsRemoved = 0;
                    targetWidth = originalWidth;
                    targetHeight = originalHeight;
                    targetWidthPercent = 100.0f;
                    targetHeightPercent = 100.0f;

                    autoRunVertical = false;
                    autoRunHorizontal = false;
                    autoRunFull = false;
                    fullResizeRunning = false;
                    hasResizeStats = false;
                    guiStatusMessage.clear();

                    LoadTextureFromMat(currentImage, imgTex);
                }
            }

            // Save resized image
            if (ImGui::Button("Save resized image")) {
                if (!imageLoaded || currentImage.empty()) {
                    guiStatusMessage = "No resized image to save.";
                }
                else {
                    std::string outputDir = "output";
                    if (!ensureOutputDirectory(outputDir)) {
                        guiStatusMessage = "Failed to create/find output directory.";
                    }
                    else {
                        float wPercentSave = 100.0f * currentImage.cols / (float)originalWidth;
                        float hPercentSave = 100.0f * currentImage.rows / (float)originalHeight;
                        int wPctInt = (int)std::round(wPercentSave);
                        int hPctInt = (int)std::round(hPercentSave);

                        std::string methodStr =
                            (methodIndex == 0) ? "dp" :
                            (methodIndex == 1) ? "greedy" : "graph";

                        std::string outputFilename = outputDir + "/output_" + methodStr + "_" +
                            std::to_string(wPctInt) + "w_" +
                            std::to_string(hPctInt) + "h_" +
                            std::to_string(currentImage.cols) + "x" +
                            std::to_string(currentImage.rows) + ".png";

                        bool ok = cv::imwrite(outputFilename, currentImage);
                        if (ok) {
                            guiStatusMessage = "Saved resized image to: " + outputFilename;
                        }
                        else {
                            guiStatusMessage = "Failed to save image to: " + outputFilename;
                        }
                    }
                }
            }

            // ----------------------------------------------------------------
            // Auto-run logic (one step per frame to keep visualization smooth)
            // ----------------------------------------------------------------
            if (autoRunVertical) {
                bool progressed = stepSeamOnce(true);
                if (!progressed) {
                    autoRunVertical = false;
                    if (!currentImage.empty())
                        LoadTextureFromMat(currentImage, imgTex);
                }
            }

            if (autoRunHorizontal) {
                bool progressed = stepSeamOnce(false);
                if (!progressed) {
                    autoRunHorizontal = false;
                    if (!currentImage.empty())
                        LoadTextureFromMat(currentImage, imgTex);
                }
            }

            if (autoRunFull) {
                bool progressed = false;
                if (currentImage.cols > targetWidth) {
                    progressed = stepSeamOnce(true);  // vertical phase
                }
                else if (currentImage.rows > targetHeight) {
                    progressed = stepSeamOnce(false); // horizontal phase
                }

                if (!progressed) {
                    autoRunFull = false;
                    if (!currentImage.empty())
                        LoadTextureFromMat(currentImage, imgTex);

                    if (fullResizeRunning) {
                        auto end = std::chrono::high_resolution_clock::now();
                        lastProcessingMs =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                end - resizeStartTime).count();
                        lastResizedWidth = currentImage.cols;
                        lastResizedHeight = currentImage.rows;
                        lastMethodIndex = methodIndex;
                        hasResizeStats = true;

                        std::string methodStr =
                            (methodIndex == 0) ? "DP" :
                            (methodIndex == 1) ? "Greedy" : "Graph (Dijkstra)";

                        guiStatusMessage = "Resize complete with " + methodStr +
                            " to " + std::to_string(lastResizedWidth) + "x" +
                            std::to_string(lastResizedHeight) + ".\n" +
                            "Processing time: " + std::to_string(lastProcessingMs) + " ms.";

                        fullResizeRunning = false;
                    }
                }
            }

            ImGui::Separator();

            if (hasResizeStats) {
                const char* mName = methodNames[lastMethodIndex];
                ImGui::Text("Last full resize:");
                ImGui::BulletText("Method: %s", mName);
                ImGui::BulletText("Final size: %d x %d", lastResizedWidth, lastResizedHeight);
                ImGui::BulletText("Processing time: %lld ms", lastProcessingMs);
            }

            if (!guiStatusMessage.empty()) {
                ImGui::Separator();
                ImGui::TextWrapped("%s", guiStatusMessage.c_str());
            }
        }

        ImGui::End(); // Controls

        // --------------------------------------------------------------------
        // Image window
        // --------------------------------------------------------------------
        ImGui::Begin("Image");
        if (imageLoaded && imgTex.id != 0) {
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float aspect = (imgTex.height > 0)
                ? (float)imgTex.height / (float)imgTex.width
                : 1.0f;

            float drawWidth = avail.x;
            float drawHeight = drawWidth * aspect;
            if (drawHeight > avail.y) {
                drawHeight = avail.y;
                drawWidth = drawHeight / aspect;
            }

            ImGui::Image(
                (void*)(intptr_t)imgTex.id,
                ImVec2(drawWidth, drawHeight)
            );
        }
        else {
            ImGui::Text("No image loaded yet.");
        }
        ImGui::End(); // Image

        // --------------------------------------------------------------------
        // Render
        // --------------------------------------------------------------------
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    imgTex.destroy();

    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

// ============================================================================
// Main entry point
// ============================================================================

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--gui") {
        return run_gui();
    }
}
