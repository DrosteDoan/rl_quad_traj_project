# Copied from the crazy_track research repo (scripts/togt/CMakeLists.txt @ 58eed32), Lesson 6.
# Builds the togt_race driver against an external TOGT-Planner checkout.
#   cmake -S scripts/togt -B <build> -DTOGT_DIR=<TOGT-Planner clone> [-DEigen3_DIR=<eigen cmake dir>]
cmake_minimum_required(VERSION 3.25)
project(togt_race CXX)
set(TOGT_DIR "" CACHE PATH "Path to a TOGT-Planner checkout")
if(NOT TOGT_DIR)
  message(FATAL_ERROR "Pass -DTOGT_DIR=<path to TOGT-Planner>")
endif()
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)             # skip TOGT's gtest fetch
# RapidJSON is fetched by TOGT's CMakeLists; keep its own tests/docs out of the configure.
set(RAPIDJSON_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(RAPIDJSON_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(RAPIDJSON_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(RAPIDJSON_BUILD_CXX17 ON CACHE BOOL "" FORCE)
add_subdirectory(${TOGT_DIR} togt_planner)
# Upstream links rapidjson as $<COMPILE_ONLY:...>, whose include dir does not reach
# drolib's own sources in this configuration; add the fetched headers explicitly.
target_include_directories(drolib PRIVATE ${CMAKE_BINARY_DIR}/_deps/rapidjson-src/include)
add_executable(togt_race togt_race.cpp)
target_link_libraries(togt_race PRIVATE drolib)
target_compile_features(togt_race PRIVATE cxx_std_17)
