include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/include/helion/IR/HelionOps.td)

mlir_tablegen(HelionDialect.h.inc -gen-dialect-decls -dialect=helion)
mlir_tablegen(HelionDialect.cpp.inc -gen-dialect-defs -dialect=helion)
mlir_tablegen(HelionOps.h.inc -gen-op-decls)
mlir_tablegen(HelionOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(HelionOpsIncGen)

add_library(HelionDialect STATIC
  lib/IR/HelionDialect.cpp
  lib/IR/HelionOps.cpp
)

add_dependencies(HelionDialect HelionOpsIncGen)

target_include_directories(HelionDialect
  PUBLIC
    ${PROJECT_SOURCE_DIR}/mlir/include
    ${PROJECT_BINARY_DIR}/mlir/include
    ${PROJECT_BINARY_DIR}/mlir
)

target_link_libraries(HelionDialect
  PUBLIC
    MLIRIR
    MLIRSupport
)

mlir_check_link_libraries(HelionDialect)

add_executable(helion-opt
  tools/helion-opt/HelionOpt.cpp
)

target_link_libraries(helion-opt
  PRIVATE
    HelionDialect
    MLIRIR
    MLIRAffineDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRMlirOptMain
)

mlir_check_link_libraries(helion-opt)
