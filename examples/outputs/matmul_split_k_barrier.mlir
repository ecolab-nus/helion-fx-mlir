=== Device IR ===
Graph 0: RootGraphInfo
opcode         name          target                                     args                                                     kwargs
-------------  ------------  -----------------------------------------  -------------------------------------------------------  --------
call_function  a             <function _host_tensor at 0x71adb6d5f560>  ('a',)                                                   {}
call_function  block_size_0  <function _get_symnode at 0x71adb6d5e480>  ('block_size_0',)                                        {}
call_function  block_size_2  <function _get_symnode at 0x71adb6d5e480>  ('block_size_2',)                                        {}
call_function  load          <function load at 0x71ad1182aa20>          (a, [block_size_0, block_size_2], None, None)            {}
call_function  b             <function _host_tensor at 0x71adb6d5f560>  ('b',)                                                   {}
call_function  block_size_1  <function _get_symnode at 0x71adb6d5e480>  ('block_size_1',)                                        {}
call_function  load_1        <function load at 0x71ad1182aa20>          (b, [block_size_2, block_size_1], None, None)            {}
call_function  acc           aten.mm.default                            (load, load_1)                                           {}
call_function  tile_id       <function tile_id at 0x71ad118434c0>       (block_size_2,)                                          {}
call_function  tmp           <function _host_tensor at 0x71adb6d5f560>  ('tmp',)                                                 {}
call_function  store         <function store at 0x71ad1182a5c0>         (tmp, [block_size_0, block_size_1, tile_id], acc, None)  {}
output         output        output                                     (None,)                                                  {}
Graph 1: RootGraphInfo
opcode         name          target                                     args                                                                      kwargs
-------------  ------------  -----------------------------------------  ------------------------------------------------------------------------  --------
call_function  tmp           <function _host_tensor at 0x71adb6d5f560>  ('tmp',)                                                                  {}
call_function  block_size_3  <function _get_symnode at 0x71adb6d5e480>  ('block_size_3',)                                                         {}
call_function  block_size_4  <function _get_symnode at 0x71adb6d5e480>  ('block_size_4',)                                                         {}
call_function  load          <function load at 0x71ad1182aa20>          (tmp, [block_size_3, block_size_4, slice(None, None, None)], None, None)  {}
call_function  sum_1         aten.sum.dim_IntList                       (load, [-1])                                                              {}
call_function  out           <function _host_tensor at 0x71adb6d5f560>  ('out',)                                                                  {}
call_function  store         <function store at 0x71ad1182a5c0>         (out, [block_size_3, block_size_4], sum_1, None)                          {}
output         output        output                                     (None,)                                                                   {}
Graph 3: ReductionLoopGraphInfo
opcode         name          target                                     args                                                                      kwargs
-------------  ------------  -----------------------------------------  ------------------------------------------------------------------------  --------
call_function  tmp           <function _host_tensor at 0x71adb6d5f560>  ('tmp',)                                                                  {}
call_function  block_size_3  <function _get_symnode at 0x71adb6d5e480>  ('block_size_3',)                                                         {}
call_function  block_size_4  <function _get_symnode at 0x71adb6d5e480>  ('block_size_4',)                                                         {}
call_function  load          <function load at 0x71ad1182aa20>          (tmp, [block_size_3, block_size_4, slice(None, None, None)], None, None)  {}
call_function  sum_1         aten.sum.dim_IntList                       (load, [-1])                                                              {}
output         output        output                                     ([sum_1],)                                                                {}


=== Nodes with symbols ===
Node a : FakeTensor(..., size=(s97, s98))
Node block_size_0 : u0
Node block_size_2 : u2
Node load : FakeTensor(..., size=(u0, u2))
Node b : FakeTensor(..., size=(s52, s20))
Node block_size_1 : u1
Node load_1 : FakeTensor(..., size=(u2, u1))
Node acc : FakeTensor(..., size=(u0, u1))
Node tile_id : u3
Node tmp : FakeTensor(..., size=(s97, s20, s98))
Node tmp : FakeTensor(..., size=(s97, s20, s98))
Node block_size_3 : u4
Node block_size_4 : u5
Node load : FakeTensor(..., size=(u4, u5, u6))
Node sum_1 : FakeTensor(..., size=(u4, u5))
Node out : FakeTensor(..., size=(s97, s20))
Node a : FakeTensor(..., size=(s97, s98))
Node block_size_0 : u0
Node block_size_2 : u2
Node load : FakeTensor(..., size=(u0, u2))
Node b : FakeTensor(..., size=(s52, s20))
Node block_size_1 : u1
Node load_1 : FakeTensor(..., size=(u2, u1))
Node acc : FakeTensor(..., size=(u0, u1))
Node tile_id : u3
Node tmp : FakeTensor(..., size=(s97, s20, s98))
Node tmp : FakeTensor(..., size=(s97, s20, s98))
Node block_size_3 : u4
Node block_size_4 : u5
Node load : FakeTensor(..., size=(u4, u5, u6))
Node sum_1 : FakeTensor(..., size=(u4, u5))
Node block_size_3 : u4
Node block_size_4 : u5
Node out : FakeTensor(..., size=(s97, s20))
Node _get_symnode : s98
Node _for_loop : [FakeTensor(..., size=(u4, u5))]
Node getitem : FakeTensor(..., size=(u4, u5))


=== Compile Environment ===
Block Sizes (6):
  Block 0: Size=s97, Var=u0, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 1: Size=s20, Var=u1, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 2: Size=s98, Var=u2, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 3: Size=s97, Var=u4, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 4: Size=s20, Var=u5, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 5: Size=s98, Var=u6, Reduction=True, Source=ReductionLoopBlockSizeSource(reduction_loop=0)
Shape Env (11):
  Var s97: 256
  Var s98: 4096
  Var s52: 4096
  Var s20: 256
  Var u0: 64
  Var u1: 64
  Var u2: 64
  Var u3: 8192
  Var u4: 64
  Var u5: 64
  Var u6: 4096


=== MLIR Dump ===
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
  func.func @split_k_matmul(%arg0: memref<256x4096xf32>, %arg1: memref<4096x256xf32>, %arg2: memref<256x256x4096xf32>, %arg3: memref<256x256xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 256 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 4096 : index} : () -> index
    affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (256 ceildiv symbol(%0), 256 ceildiv symbol(%1), 4096 ceildiv symbol(%2)) {
      %3 = arith.muli %arg4, %0 : index
      %4 = arith.muli %arg6, %2 : index
      %subview = memref.subview %arg0[%3, %4] [%0, %2] [1, 1] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
      %5 = bufferization.to_tensor %subview : memref<?x?xf32, strided<[4096, 1], offset: ?>> to tensor<?x?xf32>
      %6 = arith.muli %arg5, %1 : index
      %subview_0 = memref.subview %arg1[%4, %6] [%2, %1] [1, 1] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      %7 = bufferization.to_tensor %subview_0 : memref<?x?xf32, strided<[256, 1], offset: ?>> to tensor<?x?xf32>
      %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %10 = linalg.matmul ins(%5, %7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %subview_1 = memref.subview %arg2[%3, %6, %arg6] [%0, %1, 1] [1, 1, 1] : memref<256x256x4096xf32> to memref<?x?xf32, strided<[1048576, 4096], offset: ?>>
      %11 = bufferization.to_buffer %10 : tensor<?x?xf32> to memref<?x?xf32, strided<[1048576, 4096], offset: ?>>
      memref.copy %11, %subview_1 : memref<?x?xf32, strided<[1048576, 4096], offset: ?>> to memref<?x?xf32, strided<[1048576, 4096], offset: ?>>
    }
    affine.parallel (%arg4, %arg5) = (0, 0) to (256 ceildiv symbol(%0), 256 ceildiv symbol(%1)) {
      %3 = arith.muli %arg4, %0 : index
      %4 = arith.muli %arg5, %1 : index
      %subview = memref.subview %arg2[%3, %4, 0] [%0, %1, 4096] [1, 1, 1] : memref<256x256x4096xf32> to memref<?x?x4096xf32, strided<[1048576, 4096, 1], offset: ?>>
      %5 = bufferization.to_tensor %subview : memref<?x?x4096xf32, strided<[1048576, 4096, 1], offset: ?>> to tensor<?x?x4096xf32>
      %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5 : tensor<?x?x4096xf32>) outs(%7 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %10 = arith.addf %in, %out : f32
        linalg.yield %10 : f32
      } -> tensor<?x?xf32>
      %subview_0 = memref.subview %arg3[%3, %4] [%0, %1] [1, 1] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      %9 = bufferization.to_buffer %8 : tensor<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      memref.copy %9, %subview_0 : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
    }
    return
  }
}


mlir-opt validation succeeded.

