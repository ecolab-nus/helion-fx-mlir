=== Device IR ===
Graph 0: ForLoopGraphInfo
opcode         name            target                                     args                                             kwargs
-------------  --------------  -----------------------------------------  -----------------------------------------------  --------
placeholder    arg0_1          arg0_1                                     ()                                               {}
call_function  _new_var        <function _new_var at 0x7fd2a8d92ef0>      (arg0_1,)                                        {}
call_function  x               <function _host_tensor at 0x7fd2a8d91630>  ('x',)                                           {}
call_function  sym_size_int    aten.sym_size.int                          (arg0_1, 0)                                      {}
call_function  block_size_2    <function _get_symnode at 0x7fd2a8d909d0>  ('block_size_2',)                                {}
call_function  load            <function load at 0x7fd28ff2d1b0>          (x, [sym_size_int, block_size_2], None, None)    {}
call_function  y               <function _host_tensor at 0x7fd2a8d91630>  ('y',)                                           {}
call_function  sym_size_int_1  aten.sym_size.int                          (arg0_1, 1)                                      {}
call_function  load_1          <function load at 0x7fd28ff2d1b0>          (y, [block_size_2, sym_size_int_1], None, None)  {}
call_function  acc             aten.addmm.default                         (_new_var, load, load_1)                         {}
output         output          output                                     ([acc],)                                         {}
Graph 1: RootGraphInfo
opcode         name          target                                     args                                                      kwargs
-------------  ------------  -----------------------------------------  --------------------------------------------------------  --------
call_function  block_size_0  <function _get_symnode at 0x7fd2a8d909d0>  ('block_size_0',)                                         {}
call_function  block_size_1  <function _get_symnode at 0x7fd2a8d909d0>  ('block_size_1',)                                         {}
call_function  acc           <function full at 0x7fd28ff149d0>          ([block_size_0, block_size_1], 0.0, torch.float32, None)  {}
call_function  x_size1       <function _get_symnode at 0x7fd2a8d909d0>  ('x_size1',)                                              {}
call_function  _for_loop     <function _for_loop at 0x7fd2a8d91990>     (0, [0], [x_size1], [acc])                                {}
call_function  getitem       <built-in function getitem>                (_for_loop, 0)                                            {}
call_function  _phi          <function _phi at 0x7fd2a8d91ea0>          (acc, getitem)                                            {}
call_function  out           <function _host_tensor at 0x7fd2a8d91630>  ('out',)                                                  {}
call_function  store         <function store at 0x7fd28ff2c280>         (out, [block_size_0, block_size_1], _phi, None)           {}
output         output        output                                     (None,)                                                   {}


=== Nodes with symbols ===
Node arg0_1 : FakeTensor(..., size=(u1, u2))
Node _new_var : FakeTensor(..., size=(u1, u2))
Node x : FakeTensor(..., size=(s77, s27))
Node sym_size_int : u1
Node block_size_2 : u3
Node load : FakeTensor(..., size=(u1, u3))
Node y : FakeTensor(..., size=(s17, s94))
Node sym_size_int_1 : u2
Node load_1 : FakeTensor(..., size=(u3, u2))
Node acc : FakeTensor(..., size=(u1, u2))
Node block_size_0 : u1
Node block_size_1 : u2
Node acc : FakeTensor(..., size=(u1, u2))
Node x_size1 : s27
Node _for_loop : [FakeTensor(..., size=(u1, u2))]
Node getitem : FakeTensor(..., size=(u1, u2))
Node _phi : FakeTensor(..., size=(u1, u2))
Node out : FakeTensor(..., size=(s77, s94))


=== Compile Environment ===
Block Sizes (3):
  Block 0: Size=s77, Var=u1, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 1: Size=s94, Var=u2, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 2: Size=s27, Var=u3, Reduction=False, Source=LoopSpecBlockSizeSource()
Shape Env (7):
  Var s77: 128
  Var s27: 128
  Var s17: 128
  Var s94: 256
  Var u1: 64
  Var u2: 64
  Var u3: 64


=== MLIR Dump ===
#map = affine_map<()[s0] -> (128 ceildiv s0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
  func.func @matmul(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = "loom.get_symbol"() {name = "block_size_0"} : () -> index
    %1 = "loom.get_symbol"() {name = "block_size_1"} : () -> index
    %2 = "loom.get_symbol"() {name = "block_size_2"} : () -> index
    affine.parallel (%arg3, %arg4) = (0, 0) to (128 ceildiv symbol(%0), 256 ceildiv symbol(%1)) {
      %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %5 = affine.for %arg5 = 0 to #map()[%2] iter_args(%arg6 = %4) -> (tensor<?x?xf32>) {
        %9 = arith.muli %arg3, %0 : index
        %10 = arith.muli %arg5, %2 : index
        %subview_0 = memref.subview %arg0[%9, %10] [%0, %2] [1, 1] : memref<128x128xf32> to memref<?x?xf32, strided<[128, 1], offset: ?>>
        %11 = bufferization.to_tensor %subview_0 : memref<?x?xf32, strided<[128, 1], offset: ?>> to tensor<?x?xf32>
        %12 = arith.muli %arg4, %1 : index
        %subview_1 = memref.subview %arg1[%10, %12] [%2, %1] [1, 1] : memref<128x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        %13 = bufferization.to_tensor %subview_1 : memref<?x?xf32, strided<[256, 1], offset: ?>> to tensor<?x?xf32>
        %14 = linalg.matmul ins(%11, %13 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %15 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6, %14 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %16 = arith.addf %in, %in_2 : f32
          linalg.yield %16 : f32
        } -> tensor<?x?xf32>
        affine.yield %15 : tensor<?x?xf32>
      }
      %6 = arith.muli %arg3, %0 : index
      %7 = arith.muli %arg4, %1 : index
      %subview = memref.subview %arg2[%6, %7] [%0, %1] [1, 1] : memref<128x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      %8 = bufferization.to_buffer %5 : tensor<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      memref.copy %8, %subview : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
    }
    return
  }
}


mlir-opt validation succeeded.

