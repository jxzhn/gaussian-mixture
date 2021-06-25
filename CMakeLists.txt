cmake_minimum_required (VERSION 2.8)

project(GMM)

# 是否使用 CUDA 优化版本
option(GPU_VERSION "Use optimized CUDA version" ON)

if(GPU_VERSION)
    enable_language(CUDA)
    add_library(gmm SHARED test.cpp gmm.cpp gmm_matrix_support.cu)
    set_property(TARGET gmm PROPERTY CUDA_ARCHITECTURES OFF)
    add_definitions(-DGPU_VERSION)
else()
    add_library(gmm SHARED test.cpp gmm.cpp gmm_matrix_support.cpp)
endif()

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")