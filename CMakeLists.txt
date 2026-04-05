cmake_minimum_required(VERSION 3.16)
project(ChurnPrediction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── XGBoost ──────────────────────────────────────────────────────────────────
# Option A: installed via Homebrew / system package manager
#   brew install xgboost        (macOS)
#   sudo apt install libxgboost-dev  (Ubuntu)
#
# Option B: built from source
#   git clone --recursive https://github.com/dmlc/xgboost
#   cd xgboost && mkdir build && cd build
#   cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && make -j4 && make install
#
# Option C: conda
#   conda install -c conda-forge xgboost
#   then set -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
# ─────────────────────────────────────────────────────────────────────────────

find_package(xgboost QUIET)

if(xgboost_FOUND)
    message(STATUS "XGBoost found via CMake config")
    add_executable(churn_prediction churn_prediction.cpp)
    target_link_libraries(churn_prediction PRIVATE xgboost::xgboost)
else()
    message(STATUS "XGBoost CMake config not found; trying pkg-config / manual")

    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(XGBOOST QUIET xgboost)
    endif()

    if(XGBOOST_FOUND)
        message(STATUS "XGBoost found via pkg-config")
        add_executable(churn_prediction churn_prediction.cpp)
        target_include_directories(churn_prediction PRIVATE ${XGBOOST_INCLUDE_DIRS})
        target_link_libraries(churn_prediction PRIVATE ${XGBOOST_LIBRARIES})
        target_link_directories(churn_prediction PRIVATE ${XGBOOST_LIBRARY_DIRS})
    else()
        # Fallback: specify paths manually via CMake variables
        # cmake -B build -DXGBOOST_ROOT=/path/to/xgboost
        set(XGBOOST_ROOT "" CACHE PATH "Root directory of XGBoost installation")

        if(NOT XGBOOST_ROOT STREQUAL "")
            add_executable(churn_prediction churn_prediction.cpp)
            target_include_directories(churn_prediction PRIVATE ${XGBOOST_ROOT}/include)
            target_link_directories(churn_prediction PRIVATE ${XGBOOST_ROOT}/lib)
            target_link_libraries(churn_prediction PRIVATE xgboost)
        else()
            message(FATAL_ERROR
                "XGBoost not found.\n"
                "Install it (e.g. brew install xgboost) or set -DXGBOOST_ROOT=<path>")
        endif()
    endif()
endif()

# Optimisation flags
target_compile_options(churn_prediction PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-O0 -g>
)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
