// Camera.h
#ifndef SOARMROBOT_OBJECTDETECTION_H
#define SOARMROBOT_OBJECTDETECTION_H

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Class to hold methods for OpenCV object detection
 */
class Camera {
public:
    /**
     * Constructor
     */
    Camera();

    /**
     * Destructor
     */
    ~Camera();

    /**
     * @brief Receives camera intrinsics from main.
     */
    void setIntrinsics(const rs2_intrinsics& intrinsics);

    /**
     * @brief Processes, displays, and handles user input for frames.
     * @param color_frame Raw color frame from camera.
     * @param depth_frame Raw depth frame from camera.
     */
    void processFrames(const rs2::frame& color_frame, const rs2::depth_frame& depth_frame);

    /**
     *
     */
    bool quit() const;

private:
    /**
     * @brief Converts a RealSense frame to an OpenCV Mat.
     */
    cv::Mat frame_to_mat(const rs2::frame& f);

    /**
     * @brief Runs all segmentation logic on the depth frame.
     * @param depth_frame TDepth frame.
     * @param mask_out Bianry mask
     * @return A list of bounding boxes for the found objects.
     */
    std::vector<cv::Rect> findObjects(rs2::depth_frame& depth_frame, cv::Mat& mask_out);

    rs2::colorizer color_map; // For visualizing depth
    bool running;             // quit flag

    rs2::spatial_filter spatial_filter; // Smoothing depth data
    rs2::temporal_filter temporal_filter;// Stabilizing depth data

    rs2_intrinsics depth_intrinsics; // Camera properties
    bool intrinsics_set;              // Flag
};

#endif //SOARMROBOT_OBJECTDETECTION_H