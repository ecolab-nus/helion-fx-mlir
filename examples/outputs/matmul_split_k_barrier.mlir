=== Device IR ===
Graph 0: ForLoopGraphInfo
opcode         name            target                                     args                                             kwargs
-------------  --------------  -----------------------------------------  -----------------------------------------------  --------
placeholder    arg0_1          arg0_1                                     ()                                               {}
call_function  _new_var        <function _new_var at 0x797d23229260>      (arg0_1,)                                        {}
call_function  a               <function _host_tensor at 0x797d232b36a0>  ('a',)                                           {}
call_function  sym_size_int    aten.sym_size.int                          (arg0_1, 0)                                      {}
call_function  block_size_3    <function _get_symnode at 0x797d232b25c0>  ('block_size_3',)                                {}
call_function  load            <function load at 0x797c7dc72b60>          (a, [sym_size_int, block_size_3], None, None)    {}
call_function  b               <function _host_tensor at 0x797d232b36a0>  ('b',)                                           {}
call_function  sym_size_int_1  aten.sym_size.int                          (arg0_1, 1)                                      {}
call_function  load_1          <function load at 0x797c7dc72b60>          (b, [block_size_3, sym_size_int_1], None, None)  {}
call_function  acc             aten.addmm.default                         (_new_var, load, load_1)                         {}
output         output          output                                     ([acc],)                                         {}
Graph 1: RootGraphInfo
opcode         name          target                                     args                                                                    kwargs
-------------  ------------  -----------------------------------------  ----------------------------------------------------------------------  --------
call_function  block_size_0  <function _get_symnode at 0x797d232b25c0>  ('block_size_0',)                                                       {}
call_function  block_size_1  <function _get_symnode at 0x797d232b25c0>  ('block_size_1',)                                                       {}
call_function  acc           <function full at 0x797c7dc52340>          ([block_size_0, block_size_1], 0.0, torch.float32, device(type='cpu'))  {}
call_function  block_size_2  <function _get_symnode at 0x797d232b25c0>  ('block_size_2',)                                                       {}
call_function  tile_begin    <function tile_begin at 0x797c7dc8a700>    (block_size_2,)                                                         {}
call_function  tile_end      <function tile_end at 0x797c7dc8ab60>      (block_size_2,)                                                         {}
call_function  _for_loop     <function _for_loop at 0x797d232b3a60>     (0, [tile_begin], [tile_end], [acc])                                    {}
call_function  getitem       <built-in function getitem>                (_for_loop, 0)                                                          {}
call_function  _phi          <function _phi at 0x797d23228040>          (acc, getitem)                                                          {}
call_function  tile_id       <function tile_id at 0x797c7dc8b600>       (block_size_2,)                                                         {}
call_function  tmp           <function _host_tensor at 0x797d232b36a0>  ('tmp',)                                                                {}
call_function  store         <function store at 0x797c7dc72700>         (tmp, [block_size_0, block_size_1, tile_id], _phi, None)                {}
output         output        output                                     (None,)                                                                 {}
Graph 2: RootGraphInfo
opcode         name          target                                     args                                                                      kwargs
-------------  ------------  -----------------------------------------  ------------------------------------------------------------------------  --------
call_function  tmp           <function _host_tensor at 0x797d232b36a0>  ('tmp',)                                                                  {}
call_function  block_size_4  <function _get_symnode at 0x797d232b25c0>  ('block_size_4',)                                                         {}
call_function  block_size_5  <function _get_symnode at 0x797d232b25c0>  ('block_size_5',)                                                         {}
call_function  load          <function load at 0x797c7dc72b60>          (tmp, [block_size_4, block_size_5, slice(None, None, None)], None, None)  {}
call_function  sum_1         aten.sum.dim_IntList                       (load, [-1])                                                              {}
call_function  out           <function _host_tensor at 0x797d232b36a0>  ('out',)                                                                  {}
call_function  store         <function store at 0x797c7dc72700>         (out, [block_size_4, block_size_5], sum_1, None)                          {}
output         output        output                                     (None,)                                                                   {}
Graph 5: ReductionLoopGraphInfo
opcode         name          target                                     args                                                                      kwargs
-------------  ------------  -----------------------------------------  ------------------------------------------------------------------------  --------
call_function  tmp           <function _host_tensor at 0x797d232b36a0>  ('tmp',)                                                                  {}
call_function  block_size_4  <function _get_symnode at 0x797d232b25c0>  ('block_size_4',)                                                         {}
call_function  block_size_5  <function _get_symnode at 0x797d232b25c0>  ('block_size_5',)                                                         {}
call_function  load          <function load at 0x797c7dc72b60>          (tmp, [block_size_4, block_size_5, slice(None, None, None)], None, None)  {}
call_function  sum_1         aten.sum.dim_IntList                       (load, [-1])                                                              {}
output         output        output                                     ([sum_1],)                                                                {}


=== Nodes with symbols ===
Node arg0_1 : FakeTensor(..., size=(u3, u4))
Node _new_var : FakeTensor(..., size=(u3, u4))
Node a : FakeTensor(..., size=(s97, s98))
Node sym_size_int : u3
Node block_size_3 : u8
Node load : FakeTensor(..., size=(u3, u8))
Node b : FakeTensor(..., size=(s52, s20))
Node sym_size_int_1 : u4
Node load_1 : FakeTensor(..., size=(u8, u4))
Node acc : FakeTensor(..., size=(u3, u4))
Node block_size_0 : u3
Node block_size_1 : u4
Node acc : FakeTensor(..., size=(u3, u4))
Node block_size_2 : u5
Node tile_begin : u6
Node tile_end : u7
Node _for_loop : [FakeTensor(..., size=(u3, u4))]
Node getitem : FakeTensor(..., size=(u3, u4))
Node _phi : FakeTensor(..., size=(u3, u4))
Node tile_id : u9
Node tmp : FakeTensor(..., size=(s97, s20, u0))
Node tmp : FakeTensor(..., size=(s97, s20, u0))
Node block_size_4 : u10
Node block_size_5 : u11
Node load : FakeTensor(..., size=(u10, u11, u12))
Node sum_1 : FakeTensor(..., size=(u10, u11))
Node out : FakeTensor(..., size=(s97, s20))
Node arg0_1 : FakeTensor(..., size=(u3, u4))
Node _new_var : FakeTensor(..., size=(u3, u4))
Node a : FakeTensor(..., size=(s97, s98))
Node sym_size_int : u3
Node block_size_3 : u8
Node load : FakeTensor(..., size=(u3, u8))
Node b : FakeTensor(..., size=(s52, s20))
Node sym_size_int_1 : u4
Node load_1 : FakeTensor(..., size=(u8, u4))
Node acc : FakeTensor(..., size=(u3, u4))
Node block_size_0 : u3
Node block_size_1 : u4
Node acc : FakeTensor(..., size=(u3, u4))
Node block_size_2 : u5
Node tile_begin : u6
Node tile_end : u7
Node _for_loop : [FakeTensor(..., size=(u3, u4))]
Node getitem : FakeTensor(..., size=(u3, u4))
Node _phi : FakeTensor(..., size=(u3, u4))
Node tile_id : u9
Node tmp : FakeTensor(..., size=(s97, s20, u0))
Node tmp : FakeTensor(..., size=(s97, s20, u0))
Node block_size_4 : u10
Node block_size_5 : u11
Node load : FakeTensor(..., size=(u10, u11, u12))
Node sum_1 : FakeTensor(..., size=(u10, u11))
Node block_size_4 : u10
Node block_size_5 : u11
Node out : FakeTensor(..., size=(s97, s20))
Node _get_symnode : u0
Node _for_loop : [FakeTensor(..., size=(u10, u11))]
Node getitem : FakeTensor(..., size=(u10, u11))


=== Compile Environment ===
Block Sizes (7):
  Block 0: Size=s97, Var=u3, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 1: Size=s20, Var=u4, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 2: Size=s98, Var=u5, Reduction=False, Source=FixedBlockSizeSource(value=u2)
  Block 3: Size=-u6 + u7, Var=u8, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 4: Size=s97, Var=u10, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 5: Size=s20, Var=u11, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 6: Size=u0, Var=u12, Reduction=True, Source=ReductionLoopBlockSizeSource(reduction_loop=0)
Shape Env (17):
  Var s97: 256
  Var s98: 4096
  Var s52: 4096
  Var s20: 256
  Var u0: 8192
  Var u1: 8192
  Var u2: 8192
  Var u3: 64
  Var u4: 64
  Var u5: 64
  Var u6: 8192
  Var u7: 8192
  Var u8: 64
  Var u9: 8192
  Var u10: 64
  Var u11: 64
  Var u12: 8192


=== MLIR Dump ===
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {loom.tile_k_inner = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k_outer = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
  func.func @split_k_matmul(%arg0: memref<256x4096xf32>, %arg1: memref<4096x256xf32>, %arg2: memref<256x256x8192xf32>, %arg3: memref<256x256xf32>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c4096 = arith.constant 4096 : index
    %c256 = arith.constant 256 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 256 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k_outer, upper_bound = 4096 : index} : () -> index
    %3 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k_inner, upper_bound = 64 : index} : () -> index
    %4 = arith.ceildivui %c256, %0 : index
    %5 = arith.ceildivui %c256, %1 : index
    %6 = arith.ceildivui %c4096, %2 : index
    affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (symbol(%4), symbol(%5), symbol(%6)) {
      %7 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %9 = arith.muli %arg6, %2 : index
      %10 = arith.addi %9, %2 : index
      %11 = arith.subi %10, %9 : index
      %12 = arith.ceildivui %11, %3 : index
      %13 = scf.for %arg7 = %c0 to %12 step %c1 iter_args(%arg8 = %8) -> (tensor<?x?xf32>) {
        %17 = arith.muli %arg4, %0 : index
        %18 = arith.muli %arg7, %3 : index
        %19 = arith.addi %9, %18 : index
        %20 = arith.addi %19, %3 : index
        %21 = arith.cmpi ult, %20, %10 : index
        %22 = arith.select %21, %20, %10 : index
        %23 = arith.subi %22, %19 : index
        %subview_0 = memref.subview %arg0[%17, %19] [%0, %23] [1, 1] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
        %24 = bufferization.to_tensor %subview_0 : memref<?x?xf32, strided<[4096, 1], offset: ?>> to tensor<?x?xf32>
        %25 = arith.muli %arg5, %1 : index
        %subview_1 = memref.subview %arg1[%19, %25] [%23, %1] [1, 1] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        %26 = bufferization.to_tensor %subview_1 : memref<?x?xf32, strided<[256, 1], offset: ?>> to tensor<?x?xf32>
        %27 = linalg.matmul ins(%24, %26 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %28 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg8, %27 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%7 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %29 = arith.addf %in, %in_2 : f32
          linalg.yield %29 : f32
        } -> tensor<?x?xf32>
        scf.yield %28 : tensor<?x?xf32>
      }
      %14 = arith.muli %arg4, %0 : index
      %15 = arith.muli %arg5, %1 : index
      %subview = memref.subview %arg2[%14, %15, %arg6] [%0, %1, 1] [1, 1, 1] : memref<256x256x8192xf32> to memref<?x?xf32, strided<[2097152, 8192], offset: ?>>
      %16 = bufferization.to_buffer %13 : tensor<?x?xf32> to memref<?x?xf32, strided<[2097152, 8192], offset: ?>>
      memref.copy %16, %subview : memref<?x?xf32, strided<[2097152, 8192], offset: ?>> to memref<?x?xf32, strided<[2097152, 8192], offset: ?>>
    }
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%4), symbol(%5)) {
      %7 = arith.muli %arg4, %0 : index
      %8 = arith.muli %arg5, %1 : index
      %subview = memref.subview %arg2[%7, %8, 0] [%0, %1, 8192] [1, 1, 1] : memref<256x256x8192xf32> to memref<?x?x8192xf32, strided<[2097152, 8192, 1], offset: ?>>
      %9 = bufferization.to_tensor %subview : memref<?x?x8192xf32, strided<[2097152, 8192, 1], offset: ?>> to tensor<?x?x8192xf32>
      %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %11 = linalg.fill ins(%cst : f32) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %12 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%9 : tensor<?x?x8192xf32>) outs(%11 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %14 = arith.addf %in, %out : f32
        linalg.yield %14 : f32
      } -> tensor<?x?xf32>
      %subview_0 = memref.subview %arg3[%7, %8] [%0, %1] [1, 1] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      %13 = bufferization.to_buffer %12 : tensor<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      memref.copy %13, %subview_0 : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
    }
    return
  }
}


mlir-opt validation succeeded.

