cmake_minimum_required(VERSION 3.22.1)
project(optimizedSkinningPlugin)

include($ENV{DEVKIT_LOCATION}/cmake/pluginEntry.cmake)

# The DevKit sets a default standard. Apply the project standard afterward.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8 /openmp)
else()
    find_package(OpenMP REQUIRED)
endif()

link_directories($ENV{DEVKIT_LOCATION}/lib)

set(PROJECT_NAME optimizedSkinningPlugin)

set(SOURCE_FILES
    src/plugin_main.cpp
    src/skinClusterData.cpp
    src/algorithms.cpp
    src/cor_paper.cpp
    src/cor_contour.cpp
    src/ddm_v4.cpp
    src/dlb_gpu.cpp
    src/ocd_math.cpp 
    src/mos_curve_sampling.cpp
    src/optimizedSkinClusterMos.cpp
    src/optimizedSkinCluster.cpp
    src/optimizedSkinClusterCoR.cpp
    src/optimizedSkinClusterCmd.cpp
    src/optimizedSkinClusterCoRViewer.cpp
    src/optimizedSkinClusterCoRViewerOverride.cpp
)

set(LIBRARIES
    OpenMaya
    Foundation
    OpenMayaAnim
    OpenMayaUI
    OpenMayaRender
    clew
)

# Embed the kernel so the test plug-in can be loaded from the .mll alone.
set(DLB_KERNEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/kernel_dlb.cl")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${DLB_KERNEL_FILE}")
file(READ "${DLB_KERNEL_FILE}" DLB_OPENCL_SOURCE)
configure_file(src/dlb_gpu_kernel.h.in dlb_gpu_kernel.h @ONLY)
list(APPEND INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}")

build_plugin()

if(NOT MSVC)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()
