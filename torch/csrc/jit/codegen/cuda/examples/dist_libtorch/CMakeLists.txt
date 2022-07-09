cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(dist_example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

add_executable(dist_example main.cpp)
if (NCCL_HOME)
    target_include_directories(dist_example PUBLIC ${NCCL_HOME}/include)
    target_link_libraries(dist_example ${NCCL_HOME}/lib/libnccl.so)
endif()

target_link_libraries(dist_example ${TORCH_LIBRARIES})
