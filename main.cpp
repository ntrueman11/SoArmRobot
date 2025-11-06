#include <librealsense2/rs.hpp> 
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

int main() try {
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe.start(cfg);

    for (int i = 0; i < 10; ++i) pipe.wait_for_frames(); 

    std::cout << "--- Camera Started ---" << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    while (true) {
        rs2::frameset fs = pipe.wait_for_frames();
        rs2::depth_frame depth = fs.get_depth_frame();

        double dist_m = depth.get_distance(320, 240);

        std::cout << "\rCenter distance: " << dist_m << " m    ";

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

} catch (const rs2::error& e) {
    std::cerr << "\nRealSense error: " << e.what() << "\n";
    return 1;
} catch (const std::exception& e) {
    std::cerr << "\n" << e.what() << "\n";
    return 1;
}
