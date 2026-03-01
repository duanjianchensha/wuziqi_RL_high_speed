cmake_minimum_required(VERSION 3.18)
project(gomoku_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── 编译优化 ────────────────────────────────────
if(MSVC)
  add_compile_options(/O2 /arch:AVX2 /W3)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-O3 -march=native -Wall)
endif()

# ── pybind11（优先使用 git submodule，其次用 pip 安装的） ────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
  add_subdirectory(extern/pybind11)
else()
  find_package(pybind11 REQUIRED)
endif()

# ── Python ────────────────────────────────────
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# ── 模块目标 ──────────────────────────────────
pybind11_add_module(gomoku_cpp
  src/bindings.cpp
)

target_include_directories(gomoku_cpp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 将编译好的 .pyd/.so 放到项目根目录（Python 可直接 import）
set_target_properties(gomoku_cpp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
