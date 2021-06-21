cmake_minimum_required (VERSION 2.8)

project(gmm)

aux_source_directory(${PROJECT_SOURCE_DIR} SRC_LIST)

add_library(lib_shared SHARED ${SRC_LIST})
set_target_properties(lib_shared PROPERTIES OUTPUT_NAME "test")

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")