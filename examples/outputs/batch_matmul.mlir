=== Device IR ===
Graph 0: ForLoopGraphInfo
opcode         name            target                                     args                                                           kwargs
-------------  --------------  -----------------------------------------  -------------------------------------------------------------  --------
placeholder    arg0_1          arg0_1                                     ()                                                             {}
call_function  _new_var        <function _new_var at 0x7082d418d300>      (arg0_1,)                                                      {}
call_function  x               <function _host_tensor at 0x7082d415f740>  ('x',)                                                         {}
call_function  sym_size_int    aten.sym_size.int                          (arg0_1, 0)                                                    {}
call_function  sym_size_int_1  aten.sym_size.int                          (arg0_1, 1)                                                    {}
call_function  block_size_3    <function _get_symnode at 0x7082d415e660>  ('block_size_3',)                                              {}
call_function  load            <function load at 0x70822ece2ac0>          (x, [sym_size_int, sym_size_int_1, block_size_3], None, None)  {}
call_function  y               <function _host_tensor at 0x7082d415f740>  ('y',)                                                         {}
call_function  sym_size_int_2  aten.sym_size.int                          (arg0_1, 2)                                                    {}
call_function  load_1          <function load at 0x70822ece2ac0>          (y, [sym_size_int, block_size_3, sym_size_int_2], None, None)  {}
call_function  acc             aten.baddbmm.default                       (_new_var, load, load_1)                                       {}
output         output          output                                     ([acc],)                                                       {}
Graph 1: RootGraphInfo
opcode         name          target                                     args                                                                    kwargs
-------------  ------------  -----------------------------------------  ----------------------------------------------------------------------  --------
call_function  block_size_0  <function _get_symnode at 0x7082d415e660>  ('block_size_0',)                                                       {}
call_function  block_size_1  <function _get_symnode at 0x7082d415e660>  ('block_size_1',)                                                       {}
call_function  block_size_2  <function _get_symnode at 0x7082d415e660>  ('block_size_2',)                                                       {}
call_function  acc           <function full at 0x70822ef7a3e0>          ([block_size_0, block_size_1, block_size_2], 0.0, torch.float16, None)  {}
call_function  x_size2       <function _get_symnode at 0x7082d415e660>  ('x_size2',)                                                            {}
call_function  _for_loop     <function _for_loop at 0x7082d415fb00>     (0, [0], [x_size2], [acc])                                              {}
call_function  getitem       <built-in function getitem>                (_for_loop, 0)                                                          {}
call_function  _phi          <function _phi at 0x7082d418c0e0>          (acc, getitem)                                                          {}
call_function  out           <function _host_tensor at 0x7082d415f740>  ('out',)                                                                {}
call_function  store         <function store at 0x70822ece2660>         (out, [block_size_0, block_size_1, block_size_2], _phi, None)           {}
output         output        output                                     (None,)                                                                 {}


=== Nodes with symbols ===
Node arg0_1 : FakeTensor(..., size=(u2, u3, u4), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u2, u3, u4), dtype=torch.float16)
Node x : FakeTensor(..., size=(s77, s27, s53), dtype=torch.float16)
Node sym_size_int : u2
Node sym_size_int_1 : u3
Node block_size_3 : u5
Node load : FakeTensor(..., size=(u2, u3, u5), dtype=torch.float16)
Node y : FakeTensor(..., size=(s17, s94, s48), dtype=torch.float16)
Node sym_size_int_2 : u4
Node load_1 : FakeTensor(..., size=(u2, u5, u4), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u2, u3, u4), dtype=torch.float16)
Node block_size_0 : u2
Node block_size_1 : u3
Node block_size_2 : u4
Node acc : FakeTensor(..., size=(u2, u3, u4), dtype=torch.float16)
Node x_size2 : s53
Node _for_loop : [FakeTensor(..., size=(u2, u3, u4), dtype=torch.float16)]
Node getitem : FakeTensor(..., size=(u2, u3, u4), dtype=torch.float16)
Node _phi : FakeTensor(..., size=(u2, u3, u4), dtype=torch.float16)
Node out : FakeTensor(..., size=(s77, s27, s48), dtype=torch.float16)


=== Compile Environment ===
Block Sizes (4):
  Block 0: Size=s77, Var=u2, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 1: Size=s27, Var=u3, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 2: Size=s48, Var=u4, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 3: Size=s53, Var=u5, Reduction=False, Source=LoopSpecBlockSizeSource()
Shape Env (10):
  Var s77: 8
  Var s27: 4096
  Var s53: 512
  Var s17: 8
  Var s94: 512
  Var s48: 4096
  Var u2: 64
  Var u3: 64
  Var u4: 64
  Var u5: 64


=== MLIR Dump ===
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @batch_matmul(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c4096 = arith.constant 4096 : index
    %c8 = arith.constant 8 : index
    %c512 = arith.constant 512 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 8 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 4096 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 4096 : index} : () -> index
    %3 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 512 : index} : () -> index
    %4 = arith.ceildivui %c8, %0 : index
    %5 = arith.ceildivui %c4096, %1 : index
    %6 = arith.ceildivui %c4096, %2 : index
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (symbol(%4), symbol(%5), symbol(%6)) {
      %7 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf16>
      %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
      %9 = arith.ceildivui %c512, %3 : index
      %10 = scf.for %arg6 = %c0 to %9 step %c1 iter_args(%arg7 = %8) -> (tensor<?x?x?xf16>) {
        %15 = arith.muli %arg3, %0 : index
        %16 = arith.muli %arg4, %1 : index
        %17 = arith.muli %arg6, %3 : index
        %subview_0 = memref.subview %arg0[%15, %16, %17] [%0, %1, %3] [1, 1, 1] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
        %18 = bufferization.to_tensor %subview_0 : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to tensor<?x?x?xf16>
        %19 = arith.muli %arg5, %2 : index
        %subview_1 = memref.subview %arg1[%15, %17, %19] [%0, %3, %2] [1, 1, 1] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
        %20 = bufferization.to_tensor %subview_1 : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to tensor<?x?x?xf16>
        %21 = arith.index_cast %0 : index to i64
        %22 = arith.cmpi eq, %21, %21 : i64
        cf.assert %22, "mismatching contracting dimension"
        %23 = arith.index_cast %3 : index to i64
        %24 = arith.cmpi eq, %23, %23 : i64
        cf.assert %24, "mismatching contracting dimension"
        %25 = linalg.batch_matmul ins(%18, %20 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%8 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %26 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %arg7 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%7 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %in_2: f16, %out: f16):
          %27 = arith.addf %in, %in_2 : f16
          linalg.yield %27 : f16
        } -> tensor<?x?x?xf16>
        scf.yield %26 : tensor<?x?x?xf16>
      }
      %11 = arith.muli %arg3, %0 : index
      %12 = arith.muli %arg4, %1 : index
      %13 = arith.muli %arg5, %2 : index
      %subview = memref.subview %arg2[%11, %12, %13] [%0, %1, %2] [1, 1, 1] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
      %14 = bufferization.to_buffer %10 : tensor<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
      memref.copy %14, %subview : memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
    }
    return
  }
}


mlir-opt validation succeeded.

