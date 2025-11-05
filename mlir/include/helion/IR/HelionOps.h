//===- HelionOps.h ---------------------------------------------*- C++ -*-===//
//
// Part of the Helion FX → MLIR prototype.
//
//===----------------------------------------------------------------------===//

#ifndef HELION_IR_HELIONOPS_H
#define HELION_IR_HELIONOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include <optional>

#include "helion/IR/HelionDialect.h"

#define GET_OP_CLASSES
#include "HelionOps.h.inc"

#endif // HELION_IR_HELIONOPS_H
