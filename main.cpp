#include "Camera.h"
#include <librealsense2/rs.hpp>
#include <iostream>
#include <iomanip>

int main() try {

    Camera camera;
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    rs2::align align_to_color(RS2_STREAM_COLOR);
    rs2::pipeline_profile profile = pipe.start(cfg);

    // Set Camera Intrinsics
    auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto intrinsics = depth_stream.get_intrinsics();
    camera.setIntrinsics(intrinsics);
    for (int i = 0; i < 10; ++i) pipe.wait_for_frames(); // Warm-up

    std::cout << "Camera Started" << std::endl;
    std::cout << "Press 'q' to quit..." << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    while (!camera.quit()) {

        rs2::frameset fs = pipe.wait_for_frames();
        fs = align_to_color.process(fs);

        rs2::frame color_frame = fs.get_color_frame();
        rs2::depth_frame depth_frame = fs.get_depth_frame();

        camera.processFrames(color_frame, depth_frame);

    }

    return 0;

} catch (const rs2::error& e) {
    std::cerr << "\nRealSense error: " << e.what() << "\n";
    return 1;
} catch (const std::exception& e) {
    std::cerr << "\n" << e.what() << "\n";
    return 1;
}