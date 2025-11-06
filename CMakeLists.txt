cmake_minimum_required(VERSION 3.15)
project(SoArmRobot CXX)
set(CMAKE_CXX_STANDARD 20)

include_directories("C:/Users/Nathan/Documents/RealSense SDK 2.0/include")

add_executable(SoArmRobot main.cpp)

target_link_libraries(SoArmRobot PRIVATE
        "C:/Users/Nathan/Documents/RealSense SDK 2.0/lib/x64/realsense2.lib"
)

set(RS_BIN "C:/Users/Nathan/Documents/RealSense SDK 2.0/bin/x64")
foreach(dll realsense2.dll rsutils.dll libusb-1.0.dll)
    if(EXISTS "${RS_BIN}/${dll}")
        add_custom_command(TARGET SoArmRobot POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${RS_BIN}/${dll}" $<TARGET_FILE_DIR:SoArmRobot>)
    endif()
endforeach()