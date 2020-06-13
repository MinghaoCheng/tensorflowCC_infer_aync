# project
project (tf_test)
cmake_minimum_required (VERSION 3.7)

add_compile_options(-std=c++11 -O2)

# build type& parameters
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Wall")

# dir specification
set (PROJ_DIR ".")
set (SRC_DIR "${PROJ_DIR}/src")
set (DETECTION_DIR "${SRC_DIR}/detection")

# add folders of source files
aux_source_directory(${SRC_DIR} MAIN_SRC)
aux_source_directory(${SRC_DIR}/detection DETECTION_SRC)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
# set (Protobuf_LIBRARIES "/home/cheng/anaconda3/envs/general/lib/libprotobuf.so")
find_package(TensorflowCC REQUIRED HINTS "/home/cheng/libs/tensorflow/")

include_directories(${DETECTION_DIR})

add_executable(tf_test ${MAIN_SRC} ${DETECTION_SRC})
target_include_directories(tf_test PUBLIC ${DETECTION_DIR})
target_link_libraries(tf_test ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS} TensorflowCC::TensorflowCC ${Protobuf_LIBRARIES} ${CUDA_LIBRARIES})
