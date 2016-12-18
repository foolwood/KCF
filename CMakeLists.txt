cmake_minimum_required(VERSION 2.8)
project(kcf)
aux_source_directory(./src/ SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

IF (WIN32)
  set(OpenCV_DIR D:/opencv310/build)
ENDIF(WIN32)
IF (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") #c++11
ENDIF (UNIX)
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

 
