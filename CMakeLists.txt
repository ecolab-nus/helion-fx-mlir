cmake_minimum_required(VERSION 3.20)

project(HelionFXMLIR LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LLVM_INSTALL_DIR "/mnt/fast/llvm-mlir" CACHE PATH "Path to your MLIR install")

list(APPEND CMAKE_PREFIX_PATH
  "${LLVM_INSTALL_DIR}/lib/cmake/mlir"
  "${LLVM_INSTALL_DIR}/lib/cmake/llvm"
)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddMLIR)
include(AddLLVM)
include(HandleLLVMOptions)

if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()

add_subdirectory(mlir)
