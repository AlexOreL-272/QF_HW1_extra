cmake_minimum_required(VERSION 3.2)

project(QF_HW1_extra)

set(CMAKE_PREFIX_PATH ~/Downloads/libtorch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(QF_HW1_extra
    src/process/process.cpp
    src/underlying/underlying.cpp
    src/option/option.cpp
    src/delta_calculator/calculator.cpp
    main.cpp
)  

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)