cmake_minimum_required(VERSION 3.7)

# complier options
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

enable_language(ASM)

option(MNN_USE_SYSTEM_LIB "For opencl and vulkan, use system lib or use dlopen" ON)
option(MNN_BUILD_HARD "Build -mfloat-abi=hard or not" OFF)
option(MNN_BUILD_SHARED_LIBS "MNN build shared or static lib" OFF)
option(MNN_WIN_RUNTIME_MT "MNN use /MT on Windows dll" OFF)
option(MNN_FORBID_MULTI_THREAD "Disable Multi Thread" OFF)
option(MNN_OPENMP "Use OpenMP's thread pool implementation. Does not work on iOS or Mac OS" OFF)
option(MNN_USE_THREAD_POOL "Use MNN's own thread pool implementation" ON)
option(MNN_BUILD_TRAIN "Build MNN's training framework" OFF)
option(MNN_BUILD_DEMO "Build demo/exec or not" OFF)
option(MNN_BUILD_TOOLS "Build tools/cpp or not" ON)
option(MNN_BUILD_QUANTOOLS "Build Quantized Tools or not" OFF)
option(MNN_EVALUATION "Build Evaluation Tools or not" OFF)
option(MNN_BUILD_CONVERTER "Build Converter" OFF)
option(MNN_SUPPORT_TFLITE_QUAN "Enable MNN's tflite quantized op" ON)
option(MNN_DEBUG_MEMORY "MNN Debug Memory Access" OFF)
option(MNN_DEBUG_TENSOR_SIZE "Enable Tensor Size" OFF)
option(MNN_GPU_TRACE "Enable MNN Gpu Debug" OFF)
option(MNN_PORTABLE_BUILD "Link the static version of third party libraries where possible to improve the portability of built executables" OFF)
option(MNN_SEP_BUILD "Build MNN Backends and expression seperately. Only works with MNN_BUILD_SHARED_LIBS=ON" OFF)
option(NATIVE_LIBRARY_OUTPUT "Native Library Path" OFF)
option(NATIVE_INCLUDE_OUTPUT "Native Include Path" OFF)
option(MNN_AAPL_FMWK "Build MNN.framework instead of traditional .a/.dylib" OFF)
option(MNN_WITH_PLUGIN "Build with plugin op support." OFF)
option(MNN_BUILD_MINI "Build MNN-MINI that just supports fixed shape models." OFF)
option(MNN_USE_SSE "Use SSE optimization for x86 if possiable" ON)
option(MNN_BUILD_CODEGEN "Build with codegen" OFF)
option(MNN_ENABLE_COVERAGE "Build with coverage enable" OFF)

if(MNN_FORBID_MULTI_THREAD)
    add_definitions(-DMNN_FORBIT_MULTI_THREADS)
endif()
if(MNN_SUPPORT_TFLITE_QUAN)
    add_definitions(-DMNN_SUPPORT_TFLITE_QUAN)
endif()
if(MNN_BUILD_MINI)
    add_definitions(-DMNN_BUILD_MINI)
endif()

# backend options
option(MNN_METAL "Enable Metal" OFF)
option(MNN_OPENCL "Enable OpenCL" ON)
option(MNN_OPENGL "Enable OpenGL" OFF)
option(MNN_VULKAN "Enable Vulkan" OFF)
option(MNN_ARM82 "Enable ARM82" OFF)
option(MNN_ONEDNN "Enable oneDNN" OFF)
option(MNN_AVX512 "Enable AVX512" OFF)
option(MNN_CUDA "Enable CUDA" OFF)
option(MNN_TENSORRT "Enable TensorRT" OFF)

if (MNN_USE_THREAD_POOL)
    message(STATUS "Use Threadpool, forbid openmp")
    set(MNN_OPENMP OFF)
    add_definitions(-DMNN_USE_THREAD_POOL)
endif()

add_definitions(-DMNN_USE_LOGCAT)
add_definitions(-fPIC)
add_definitions(-DMNN_BUILD_FOR_ANDROID)
add_definitions(-mfloat-abi=softfp -mfpu=neon)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${CMAKE_CURRENT_LIST_DIR}/include/
                    ${CMAKE_CURRENT_LIST_DIR}/source/
                    ${CMAKE_CURRENT_LIST_DIR}/express/
                    ${CMAKE_CURRENT_LIST_DIR}/tools/
                    ${CMAKE_CURRENT_LIST_DIR}/codegen/
                    ${CMAKE_CURRENT_LIST_DIR}/schema/current/
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/flatbuffers/include
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/half
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/imageHelper
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/OpenCLHeaders/
                  )

set(MNN_OBJECTS_TO_LINK "")
set(MNN_TARGETS "")

# Core
FILE(GLOB MNN_Core_SRC ${CMAKE_CURRENT_LIST_DIR}/source/core/*)
add_library(MNNCore OBJECT ${MNN_Core_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCore>)
list(APPEND MNN_TARGETS MNNCore)

# CV
FILE(GLOB MNN_CV_SRC ${CMAKE_CURRENT_LIST_DIR}/source/cv/*)
add_library(MNNCV OBJECT ${MNN_CV_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCV>)
list(APPEND MNN_TARGETS MNNCV)

# Math
FILE(GLOB MNN_Math_SRC ${CMAKE_CURRENT_LIST_DIR}/source/math/*)
add_library(MNNMath OBJECT ${MNN_Math_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNMath>)
list(APPEND MNN_TARGETS MNNMath)

# Transform
FILE(GLOB MNN_Transform_SRC ${CMAKE_CURRENT_LIST_DIR}/source/shape/* ${CMAKE_CURRENT_LIST_DIR}/source/geometry/*)
add_library(MNNTransform OBJECT ${MNN_Transform_SRC})
IF (NOT MNN_BUILD_MINI)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNTransform>)
ENDIF()
list(APPEND MNN_TARGETS MNNTransform)

# Utils
FILE(GLOB MNN_Utils_SRC ${CMAKE_CURRENT_LIST_DIR}/source/utils/*)
add_library(MNNUtils OBJECT ${MNN_Utils_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNUtils>)
list(APPEND MNN_TARGETS MNNUtils)

# CPU
FILE(GLOB MNN_CPU_SRC ${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/* ${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/compute/*)
add_library(MNNCPU OBJECT ${MNN_CPU_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCPU>)
list(APPEND MNN_TARGETS MNNCPU)

# AArch32/64 Assemblies
include(${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/arm/CMakeLists.txt)

SET(MNN_PUB_HDRS "")
SET(MNN_EXPR_PUB_HDRS "")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/MNNDefine.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/Interpreter.hpp")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/HalideRuntime.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/Tensor.hpp")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/ErrorCode.hpp")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/ImageProcess.hpp")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/Matrix.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/Rect.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/MNNForwardType.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/AutoTime.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/Expr.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/ExprCreator.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/MathOp.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/NeuralNetWorkOp.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/Optimizer.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/Executor.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/NN.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/Module.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/NeuralNetWorkOp.hpp")

set(MNN_DEPS "")
set(MNN_EXTRA_DEPENDS "")

if(MNN_OPENMP)
    message(STATUS "[*] Checking OpenMP")
    find_package(OpenMP)
    # For CMake < 3.9, we need to make the target ourselves
    if(NOT TARGET OpenMP::OpenMP_CXX)
        find_package(Threads REQUIRED)
        add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
        set_property(TARGET OpenMP::OpenMP_CXX
            PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
        # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
        set_property(TARGET OpenMP::OpenMP_CXX
            PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
    endif()
    # TODO: Don't pollute global CFLAGS
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    list(APPEND MNN_EXTRA_DEPENDS OpenMP::OpenMP_CXX)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions ")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math")

# Vulkan
IF(MNN_VULKAN)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/vulkan/)
  IF(MNN_SEP_BUILD)
    list(APPEND MNN_DEPS MNN_Vulkan)
  ELSE()
    list(APPEND MNN_TARGETS MNN_Vulkan)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_Vulkan>)
    list(APPEND MNN_EXTRA_DEPENDS ${MNN_VULKAN_LIBS})
  ENDIF()
ENDIF()

# OpenCL
IF(MNN_OPENCL)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/opencl/)
  IF(MNN_SEP_BUILD)
    list(APPEND MNN_DEPS MNN_CL)
  ELSE()
    list(APPEND MNN_TARGETS MNN_CL)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_CL>)
    list(APPEND MNN_EXTRA_DEPENDS ${MNN_OCL_LIBS})
  ENDIF()
ENDIF()

# OpenGL
IF(MNN_OPENGL)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/opengl/)
  IF(MNN_SEP_BUILD)
    list(APPEND MNN_DEPS MNN_GL)
  ELSE()
    list(APPEND MNN_TARGETS MNN_GL)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_GL>)
    list(APPEND MNN_EXTRA_DEPENDS GLESv3)
    list(APPEND MNN_EXTRA_DEPENDS EGL)
  ENDIF()
ENDIF()

IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR IOS_ARCH STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
# ARM82 Assemblies
  IF(MNN_ARM82)
    add_definitions(-DENABLE_ARMV82)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/arm82/)
    list(APPEND MNN_TARGETS MNN_Arm82)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_Arm82>)
  ENDIF()
ENDIF()

# Express
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/express/)
IF(MNN_SEP_BUILD)
  list(APPEND MNN_DEPS MNN_Express)
ELSE()
   list(APPEND MNN_TARGETS MNN_Express)
   list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_Express>)
ENDIF()

# NPU
IF(MNN_NPU)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/hiai/)
    IF(MNN_SEP_BUILD)
        list(APPEND MNN_DEPS MNN_NPU)
    ELSE()
        list(APPEND MNN_TARGETS MNN_NPU)
        list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_NPU>)
        list(APPEND MNN_EXTRA_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/source/backend/hiai/3rdParty/${ANDROID_ABI}/libhiai.so)
        list(APPEND MNN_EXTRA_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/source/backend/hiai/3rdParty/${ANDROID_ABI}/libhiai_ir_build.so)
        list(APPEND MNN_EXTRA_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/source/backend/hiai/3rdParty/${ANDROID_ABI}/libhiai_ir.so)
    ENDIF()
ENDIF()

IF(MNN_SEP_BUILD)
    add_library(MNN SHARED ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS})
    target_link_libraries(MNN PUBLIC ${MNN_EXTRA_DEPENDS})
ELSE()
    IF(MNN_BUILD_SHARED_LIBS)
        add_library(MNN SHARED ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS})
    ELSE()
        add_library(MNN STATIC ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS})
    ENDIF()
    target_link_libraries(MNN PUBLIC ${MNN_EXTRA_DEPENDS})
ENDIF()

add_dependencies(MNN MNNCore MNNCV MNNTransform MNNMath MNNCPU)

target_link_libraries(MNN PUBLIC log m)

if (NOT MNN_BUILD_SHARED_LIBS)
    # Static-link will not replace thread-related weak symbol in glibc with strong symbol
    # in pthread library, so we need use --whole-archive to pthread
    # https://stackoverflow.com/questions/35116327/when-g-static-link-pthread-cause-segmentation-fault-why
    set(MNN_DEPS -Wl,--whole-archive ${MNN_DEPS} -Wl,--no-whole-archive PARENT_SCOPE)
endif()



