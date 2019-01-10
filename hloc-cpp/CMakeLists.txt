cmake_minimum_required(VERSION 2.8.3)
project(hloc_cpp)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

add_compile_options(-std=c++11 -Ofast -ffast-math -march=native -mavx2)

pybind_add_module(_hloc_cpp MODULE src/hloc.cc)

catkin_python_setup()

cs_install()
cs_export()

# For Python 3.6 use:
# --cmake-args -DPYBIND_PYTHON_EXECUTABLE=/usr/bin/python3.6

install(TARGETS _hloc_cpp LIBRARY DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION})
