cmake_minimum_required(VERSION 3.4.1)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/MNN/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/MNN/include/MNN/)

add_library(xindaface SHARED
        src/main/cpp/Algorithm_jni.cpp)

#target_link_libraries(xindaface MNN ${MNN_DEPS} ${MNN_OBJECTS_TO_LINK})
target_link_libraries(xindaface -Wl,--whole-archive MNN -Wl,--no-whole-archive jnigraphics android)
#target_link_libraries(xindaface ${MNN_DEPS} jnigraphics android)

message(STATUS "===================================")
message(STATUS "MNN_DEPS = ${MNN_DEPS}")
message(STATUS "MNN_OBJECTS_TO_LINK = ${MNN_OBJECTS_TO_LINK}")

#add_dependencies(MNNCore MNNCV MNNTransform MNNMath MNNCPU)

ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/MNN )