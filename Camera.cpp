// Camera.cpp
#include "Camera.h"
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iomanip>

Camera::Camera() : running(true), intrinsics_set(false) {
    cv::namedWindow("Color Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Depth Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Mask", cv::WINDOW_AUTOSIZE);
}

Camera::~Camera() {
    cv::destroyAllWindows();
}

void Camera::setIntrinsics(const rs2_intrinsics& intrinsics) {
    this->depth_intrinsics = intrinsics;
    this->intrinsics_set = true;
}

std::vector<cv::Rect> Camera::findObjects(rs2::depth_frame& depth_frame, cv::Mat& mask_out) {

    depth_frame = temporal_filter.process(depth_frame);
    depth_frame = spatial_filter.process(depth_frame);

    cv::Mat depth_16u_mat = frame_to_mat(depth_frame);

    cv::inRange(depth_16u_mat, 100, 1000, mask_out);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::morphologyEx(mask_out, mask_out, cv::MORPH_OPEN, kernel); // Remove noise specks
    cv::morphologyEx(mask_out, mask_out, cv::MORPH_CLOSE, kernel); // Fill holes in objects

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_out, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> boxes;
    for (const auto& contour : contours) {
        //Filter noise
        if (cv::contourArea(contour) > 1000) {
            boxes.push_back(cv::boundingRect(contour));
        }
    }
    return boxes;
}

void Camera::processFrames(const rs2::frame& color_frame, const rs2::depth_frame& depth_frame_in) {

    rs2::depth_frame depth_frame = depth_frame_in;

    cv::Mat mask; // This will be filled by findObjects
    std::vector<cv::Rect> boxes = findObjects(depth_frame, mask);


    cv::Mat color_mat = frame_to_mat(color_frame);

    for (const auto& box : boxes) {

        // Draw the green box
        cv::rectangle(color_mat, box, cv::Scalar(0, 255, 0), 2);

        // Find the center of the bounding box
        cv::Point center = (box.tl() + box.br()) / 2;

        float Z = depth_frame.get_distance(center.x, center.y);

        if (Z > 0 && intrinsics_set) {
            float pixel[2] = { (float)center.x, (float)center.y };
            float point[3];

            rs2_deproject_pixel_to_point(point, &depth_intrinsics, pixel, Z);

        }
    }

    rs2::frame visible_depth = color_map.process(depth_frame);
    cv::Mat depth_mat = frame_to_mat(visible_depth);

    // Show cameras
    cv::imshow("Color Image", color_mat);
    cv::imshow("Depth Image", depth_mat);
    cv::imshow("Mask", mask);

    int key = cv::waitKey(1);
    if (key == 'q') {
        running = false;
    }
}

bool Camera::quit() const {
    return !running;
}

cv::Mat Camera::frame_to_mat(const rs2::frame& f) {
    auto vf = f.as<rs2::video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8) {
        return cv::Mat(cv::Size(w, h), CV_8UC3, (void*)f.get_data(), cv::Mat::AUTO_STEP);
    }
    if (f.get_profile().format() == RS2_FORMAT_RGB8) {
        auto r = cv::Mat(cv::Size(w, h), CV_8UC3, (void*)f.get_data(), cv::Mat::AUTO_STEP);
        cv::cvtColor(r, r, cv::COLOR_RGB2BGR); // Fix color
        return r;
    }
    if (f.get_profile().format() == RS2_FORMAT_Z16) {
        return cv::Mat(cv::Size(w, h), CV_16UC1, (void*)f.get_data(), cv::Mat::AUTO_STEP);
    }
    if (f.get_profile().format() == RS2_FORMAT_Y8) {
        return cv::Mat(cv::Size(w, h), CV_8UC1, (void*)f.get_data(), cv::Mat::AUTO_STEP);
    }

    throw std::runtime_error("Unsupported frame format");
}