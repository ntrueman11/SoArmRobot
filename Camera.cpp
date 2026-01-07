#include "Camera.h"
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iomanip>

/**
 * @brief Constructor for the Camera class.
 */
Camera::Camera() : running(true), intrinsics_set(false) {

    cv::namedWindow("Live Feed", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Object Mask", cv::WINDOW_AUTOSIZE);

    // Initialize RealSense filters
    spatial_filter = rs2::spatial_filter();
    temporal_filter = rs2::temporal_filter();
}

/**
 * @brief Destructor for the Camera class.
 * Ensures all UI windows are properly closed upon exit.
 */
Camera::~Camera() {
    cv::destroyAllWindows();
}

/**
 * @brief Updates the camera's internal intrinsic parameters.
 * @param intrinsics The intrinsic properties of the depth stream.
 */
void Camera::setIntrinsics(const rs2_intrinsics& intrinsics) {
    this->depth_intrinsics = intrinsics;
    this->intrinsics_set = true;
}

/**
 * @brief Detects objects using depth protrusions and color filtering.
 * * @param depth_frame Input depth data for distance-based detection.
 * @param color_mat Input RGB data for color-based detection.
 * @param mask_out Output binary image showing detected shapes.
 * @return std::vector<cv::Rect> List of bounding boxes for found objects.
 */
std::vector<cv::Rect> Camera::findObjects(rs2::depth_frame& depth_frame, const cv::Mat& color_mat, cv::Mat& mask_out) {

    // Reduce depth noise and temporal flickering before processing
    depth_frame = temporal_filter.process(depth_frame);
    depth_frame = spatial_filter.process(depth_frame);
    cv::Mat depth_16u = frame_to_mat(depth_frame);

    // Create a kernel significantly larger than the target objects
    cv::Mat kernel_surface = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(60, 60));
    cv::Mat protrusions;

    // Highlights local objects by subtracting the "flat" desk surface from the image.
    cv::morphologyEx(depth_16u, protrusions, cv::MORPH_TOPHAT, kernel_surface);

    // Threshold to keep only pixels protruding at least 15mm from the estimated surface
    cv::threshold(protrusions, mask_out, 15, 255, cv::THRESH_BINARY);
    mask_out.convertTo(mask_out, CV_8UC1);

    // Complementary color mask to help detect objects
    cv::Mat hsv_mat, color_mask;
    cv::cvtColor(color_mat, hsv_mat, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_mat, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 60), color_mask);

    // Valid if picked up by either filter
    cv::bitwise_or(mask_out, color_mask, mask_out);

    // Clean up small noise specks and fill small holes within the object masks
    cv::Mat kernel_clean = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::morphologyEx(mask_out, mask_out, cv::MORPH_OPEN, kernel_clean);
    cv::morphologyEx(mask_out, mask_out, cv::MORPH_CLOSE, kernel_clean);

    // Find the boundaries of the cleaned binary blobs
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_out, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> detected_boxes;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        // Size filter to ensure high-confidence detections for downstream ML models
        if (area > 2000) {
            detected_boxes.push_back(cv::boundingRect(contour));
        }
    }
    return detected_boxes;
}

/**
 * @brief High-level frame handler for visualization and 3D coordinate logging.
 * Maps 2D detections back into real-world meters relative to the camera center.
 * @param color_frame The raw RGB frame from the camera.
 * @param depth_frame_raw The raw depth frame from the camera.
 */
void Camera::processFrames(const rs2::frame& color_frame, const rs2::depth_frame& depth_frame_raw) {

    cv::Mat color_mat = frame_to_mat(color_frame);
    rs2::depth_frame depth_frame = depth_frame_raw;

    cv::Mat final_mask;
    std::vector<cv::Rect> objects = findObjects(depth_frame, color_mat, final_mask);

    for (const auto& box : objects) {
        cv::rectangle(color_mat, box, cv::Scalar(0, 255, 0), 2);

        // Compute the center of the 2D bounding box
        cv::Point center = (box.tl() + box.br()) / 2;
        float distance = depth_frame.get_distance(center.x, center.y);

        if (distance > 0 && intrinsics_set) {
            float pixel[2] = { (float)center.x, (float)center.y };
            float point[3];

            // Map the 2D pixel + Depth into real-world 3D space
            rs2_deproject_pixel_to_point(point, &depth_intrinsics, pixel, distance);

            std::cout << "\rTarget Position (m)      X: " << std::fixed << std::setprecision(3)
              << point[0] << " | Y: " << point[1] << " | Z: " << point[2] << std::flush;
        } else {
            // object inside the 10cm blind spot
            // Need to adjust head/claw position
            std::cout << "\rBlind Spot" << std::flush;
        }
    }

    cv::imshow("Live Feed", color_mat);
    cv::imshow("Object Mask", final_mask);

    if (cv::waitKey(1) == 'q') running = false;
}

/**
 * @brief Getter to check the current status of the camera loop.
 * @return True if the user has not requested a quit.
 */
bool Camera::quit() const {
    return !running;
}

/**
 * @brief Helper function to convert RealSense frame data into OpenCV cv::Mat format.
 * @param f The RealSense frame to convert.
 * @return cv::Mat The resulting OpenCV matrix.
 */
cv::Mat Camera::frame_to_mat(const rs2::frame& f) {
    auto vf = f.as<rs2::video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    // Map the RealSense data buffer to an OpenCV matrix based on pixel format
    if (f.get_profile().format() == RS2_FORMAT_BGR8) {
        return cv::Mat(cv::Size(w, h), CV_8UC3, (void*)f.get_data(), cv::Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16) {
        return cv::Mat(cv::Size(w, h), CV_16UC1, (void*)f.get_data(), cv::Mat::AUTO_STEP);
    }
    throw std::runtime_error("Unsupported frame format during conversion.");
}