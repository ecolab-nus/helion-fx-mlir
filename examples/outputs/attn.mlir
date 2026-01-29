=== Device IR ===
Graph 0: ForLoopGraphInfo
opcode         name          target                                     args                                                                         kwargs
-------------  ------------  -----------------------------------------  ---------------------------------------------------------------------------  --------
placeholder    arg0_1        arg0_1                                     ()                                                                           {}
placeholder    arg1_1        arg1_1                                     ()                                                                           {}
placeholder    arg2_1        arg2_1                                     ()                                                                           {}
placeholder    arg3_1        arg3_1                                     ()                                                                           {}
call_function  _new_var      <function _new_var at 0x7fbc0b486cb0>      (arg0_1,)                                                                    {}
call_function  _new_var_1    <function _new_var at 0x7fbc0b486cb0>      (arg1_1,)                                                                    {}
call_function  _new_var_2    <function _new_var at 0x7fbc0b486cb0>      (arg2_1,)                                                                    {}
call_function  _new_var_3    <function _new_var at 0x7fbc0b486cb0>      (arg3_1,)                                                                    {}
call_function  k_view        <function _host_tensor at 0x7fbc0b4853f0>  ('k_view',)                                                                  {}
call_function  sym_size_int  aten.sym_size.int                          (arg0_1, 0)                                                                  {}
call_function  block_size_3  <function _get_symnode at 0x7fbc0b484790>  ('block_size_3',)                                                            {}
call_function  k             <function load at 0x7fbbfc6d4f70>          (k_view, [sym_size_int, slice(None, None, None), block_size_3], None, None)  {}
call_function  qk            aten.bmm.default                           (_new_var, k)                                                                {}
call_function  _mask_to_2    <function _mask_to at 0x7fbc0b486950>      (qk, -inf)                                                                   {}
call_function  amax          aten.amax.default                          (_mask_to_2, [-1])                                                           {}
call_function  mul           aten.mul.Tensor                            (amax, 0.12751743074602467)                                                  {}
call_function  m_ij          aten.maximum.default                       (_new_var_1, mul)                                                            {}
call_function  mul_1         aten.mul.Tensor                            (qk, 0.12751743074602467)                                                    {}
call_function  subscript     <function subscript at 0x7fbbfc6f20e0>     (m_ij, [slice(None, None, None), slice(None, None, None), None])             {}
call_function  qk_1          aten.sub.Tensor                            (mul_1, subscript)                                                           {}
call_function  exp2          aten.exp2.default                          (qk_1,)                                                                      {}
call_function  _mask_to_3    <function _mask_to at 0x7fbc0b486950>      (exp2, 0)                                                                    {}
call_function  l_ij          aten.sum.dim_IntList                       (_mask_to_3, [-1])                                                           {}
call_function  sub_1         aten.sub.Tensor                            (_new_var_1, m_ij)                                                           {}
call_function  alpha         aten.exp2.default                          (sub_1,)                                                                     {}
call_function  mul_2         aten.mul.Tensor                            (_new_var_2, alpha)                                                          {}
call_function  l_i           aten.add.Tensor                            (mul_2, l_ij)                                                                {}
call_function  subscript_1   <function subscript at 0x7fbbfc6f20e0>     (alpha, [slice(None, None, None), slice(None, None, None), None])            {}
call_function  acc           aten.mul.Tensor                            (_new_var_3, subscript_1)                                                    {}
call_function  v_view        <function _host_tensor at 0x7fbc0b4853f0>  ('v_view',)                                                                  {}
call_function  v             <function load at 0x7fbbfc6d4f70>          (v_view, [sym_size_int, block_size_3, slice(None, None, None)], None, None)  {}
call_function  acc_1         aten.baddbmm.default                       (acc, _mask_to_3, v)                                                         {}
call_function  m_i           <function _new_var at 0x7fbc0b486cb0>      (m_ij,)                                                                      {}
output         output        output                                     ([m_i, l_i, acc_1],)                                                         {}
Graph 1: RootGraphInfo
opcode         name          target                                     args                                                                         kwargs
-------------  ------------  -----------------------------------------  ---------------------------------------------------------------------------  ----------------------------------------------------------------------------------------------------
call_function  block_size_0  <function _get_symnode at 0x7fbc0b484790>  ('block_size_0',)                                                            {}
call_function  block_size_1  <function _get_symnode at 0x7fbc0b484790>  ('block_size_1',)                                                            {}
call_function  m_i           <function full at 0x7fbbfc6bc790>          ([block_size_0, block_size_1], -inf, torch.float32, None)                    {}
call_function  l_i           aten.full.default                          ([block_size_0, block_size_1], 1.0)                                          {'dtype': torch.float32, 'layout': torch.strided, 'device': device(type='cpu'), 'pin_memory': False}
call_function  acc           <function full at 0x7fbbfc6bc790>          ([block_size_0, block_size_1, 128], 0.0, torch.float32, None)                {}
call_function  q_view        <function _host_tensor at 0x7fbc0b4853f0>  ('q_view',)                                                                  {}
call_function  q             <function load at 0x7fbbfc6d4f70>          (q_view, [block_size_0, block_size_1, slice(None, None, None)], None, None)  {}
call_function  k_in_size1    <function _get_symnode at 0x7fbc0b484790>  ('k_in_size1',)                                                              {}
call_function  _for_loop     <function _for_loop at 0x7fbc0b485750>     (0, [0], [k_in_size1], [q, m_i, l_i, acc])                                   {}
call_function  getitem       <built-in function getitem>                (_for_loop, 0)                                                               {}
call_function  getitem_1     <built-in function getitem>                (_for_loop, 1)                                                               {}
call_function  getitem_2     <built-in function getitem>                (_for_loop, 2)                                                               {}
call_function  _phi          <function _phi at 0x7fbc0b485c60>          (m_i, getitem)                                                               {}
call_function  _phi_1        <function _phi at 0x7fbc0b485c60>          (l_i, getitem_1)                                                             {}
call_function  _phi_2        <function _phi at 0x7fbc0b485c60>          (acc, getitem_2)                                                             {}
call_function  subscript     <function subscript at 0x7fbbfc6f20e0>     (_phi_1, [slice(None, None, None), slice(None, None, None), None])           {}
call_function  acc_1         aten.div.Tensor                            (_phi_2, subscript)                                                          {}
call_function  out           <function _host_tensor at 0x7fbc0b4853f0>  ('out',)                                                                     {}
call_function  store         <function store at 0x7fbbfc6d4040>         (out, [block_size_0, block_size_1, slice(None, None, None)], acc_1, None)    {}
output         output        output                                     (None,)                                                                      {}


=== Nodes with symbols ===
Node arg0_1 : FakeTensor(..., size=(u4, u5, 128))
Node arg1_1 : FakeTensor(..., size=(u4, u5))
Node arg2_1 : FakeTensor(..., size=(u4, u5))
Node arg3_1 : FakeTensor(..., size=(u4, u5, 128))
Node _new_var : FakeTensor(..., size=(u4, u5, 128))
Node _new_var_1 : FakeTensor(..., size=(u4, u5))
Node _new_var_2 : FakeTensor(..., size=(u4, u5))
Node _new_var_3 : FakeTensor(..., size=(u4, u5, 128))
Node k_view : FakeTensor(..., size=(s35, 128, s34))
Node sym_size_int : u4
Node block_size_3 : u7
Node k : FakeTensor(..., size=(u4, 128, u7))
Node qk : FakeTensor(..., size=(u4, u5, u7))
Node _mask_to_2 : FakeTensor(..., size=(u4, u5, u7))
Node amax : FakeTensor(..., size=(u4, u5))
Node mul : FakeTensor(..., size=(u4, u5))
Node m_ij : FakeTensor(..., size=(u4, u5))
Node mul_1 : FakeTensor(..., size=(u4, u5, u7))
Node subscript : FakeTensor(..., size=(u4, u5, 1))
Node qk_1 : FakeTensor(..., size=(u4, u5, u7))
Node exp2 : FakeTensor(..., size=(u4, u5, u7))
Node _mask_to_3 : FakeTensor(..., size=(u4, u5, u7))
Node l_ij : FakeTensor(..., size=(u4, u5))
Node sub_1 : FakeTensor(..., size=(u4, u5))
Node alpha : FakeTensor(..., size=(u4, u5))
Node mul_2 : FakeTensor(..., size=(u4, u5))
Node l_i : FakeTensor(..., size=(u4, u5))
Node subscript_1 : FakeTensor(..., size=(u4, u5, 1))
Node acc : FakeTensor(..., size=(u4, u5, 128))
Node v_view : FakeTensor(..., size=(s80, s34, 128))
Node v : FakeTensor(..., size=(u4, u7, 128))
Node acc_1 : FakeTensor(..., size=(u4, u5, 128))
Node m_i : FakeTensor(..., size=(u4, u5))
Node block_size_0 : u4
Node block_size_1 : u5
Node m_i : FakeTensor(..., size=(u4, u5))
Node l_i : FakeTensor(..., size=(u4, u5))
Node acc : FakeTensor(..., size=(u4, u5, 128))
Node q_view : FakeTensor(..., size=(s30, s48, 128))
Node q : FakeTensor(..., size=(u4, u5, 128))
Node k_in_size1 : s34
Node _for_loop : [FakeTensor(..., size=(u4, u5)), FakeTensor(..., size=(u4, u5)), FakeTensor(..., size=(u4, u5, 128))]
Node getitem : FakeTensor(..., size=(u4, u5))
Node getitem_1 : FakeTensor(..., size=(u4, u5))
Node getitem_2 : FakeTensor(..., size=(u4, u5, 128))
Node _phi : FakeTensor(..., size=(u4, u5))
Node _phi_1 : FakeTensor(..., size=(u4, u5))
Node _phi_2 : FakeTensor(..., size=(u4, u5, 128))
Node subscript : FakeTensor(..., size=(u4, u5, 1))
Node acc_1 : FakeTensor(..., size=(u4, u5, 128))
Node out : FakeTensor(..., size=(s30, s48, 128))
Node arg0_1 : FakeTensor(..., size=(u4, u5, 128))
Node arg1_1 : FakeTensor(..., size=(u4, u5))
Node arg2_1 : FakeTensor(..., size=(u4, u5))
Node arg3_1 : FakeTensor(..., size=(u4, u5, 128))
Node _new_var : FakeTensor(..., size=(u4, u5, 128))
Node _new_var_1 : FakeTensor(..., size=(u4, u5))
Node _new_var_2 : FakeTensor(..., size=(u4, u5))
Node _new_var_3 : FakeTensor(..., size=(u4, u5, 128))
Node k_view : FakeTensor(..., size=(s35, 128, s34))
Node sym_size_int : u4
Node block_size_3 : u7
Node k : FakeTensor(..., size=(u4, 128, u7))
Node qk : FakeTensor(..., size=(u4, u5, u7))
Node _mask_to_2 : FakeTensor(..., size=(u4, u5, u7))
Node amax : FakeTensor(..., size=(u4, u5))
Node mul : FakeTensor(..., size=(u4, u5))
Node m_ij : FakeTensor(..., size=(u4, u5))
Node mul_1 : FakeTensor(..., size=(u4, u5, u7))
Node subscript : FakeTensor(..., size=(u4, u5, 1))
Node qk_1 : FakeTensor(..., size=(u4, u5, u7))
Node exp2 : FakeTensor(..., size=(u4, u5, u7))
Node _mask_to_3 : FakeTensor(..., size=(u4, u5, u7))
Node l_ij : FakeTensor(..., size=(u4, u5))
Node sub_1 : FakeTensor(..., size=(u4, u5))
Node alpha : FakeTensor(..., size=(u4, u5))
Node mul_2 : FakeTensor(..., size=(u4, u5))
Node l_i : FakeTensor(..., size=(u4, u5))
Node subscript_1 : FakeTensor(..., size=(u4, u5, 1))
Node acc : FakeTensor(..., size=(u4, u5, 128))
Node v_view : FakeTensor(..., size=(s80, s34, 128))
Node v : FakeTensor(..., size=(u4, u7, 128))
Node acc_1 : FakeTensor(..., size=(u4, u5, 128))
Node m_i : FakeTensor(..., size=(u4, u5))
Node block_size_0 : u4
Node block_size_1 : u5
Node m_i : FakeTensor(..., size=(u4, u5))
Node l_i : FakeTensor(..., size=(u4, u5))
Node acc : FakeTensor(..., size=(u4, u5, 128))
Node q_view : FakeTensor(..., size=(s30, s48, 128))
Node q : FakeTensor(..., size=(u4, u5, 128))
Node k_in_size1 : s34
Node _for_loop : [FakeTensor(..., size=(u4, u5)), FakeTensor(..., size=(u4, u5)), FakeTensor(..., size=(u4, u5, 128))]
Node getitem : FakeTensor(..., size=(u4, u5))
Node getitem_1 : FakeTensor(..., size=(u4, u5))
Node getitem_2 : FakeTensor(..., size=(u4, u5, 128))
Node _phi : FakeTensor(..., size=(u4, u5))
Node _phi_1 : FakeTensor(..., size=(u4, u5))
Node _phi_2 : FakeTensor(..., size=(u4, u5, 128))
Node subscript : FakeTensor(..., size=(u4, u5, 1))
Node acc_1 : FakeTensor(..., size=(u4, u5, 128))
Node out : FakeTensor(..., size=(s30, s48, 128))


=== Compile Environment ===
Block Sizes (4):
  Block 0: Size=s30, Var=u4, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 1: Size=s48, Var=u5, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 2: Size=128, Var=128, Reduction=True, Source=ReductionLoopBlockSizeSource(reduction_loop=0)
  Block 3: Size=s34, Var=u7, Reduction=False, Source=LoopSpecBlockSizeSource()
Shape Env (13):
  Var s30: 256
  Var s48: 512
  Var s22: 128
  Var s35: 256
  Var s34: 512
  Var s4: 128
  Var s80: 256
  Var s41: 512
  Var s66: 128
  Var u4: 64
  Var u5: 64
  Var u6: 128
  Var u7: 64


=== MLIR Dump ===
#map = affine_map<()[s0] -> (512 ceildiv s0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_3 = -1 : index} {
  func.func @attention(%arg0: memref<256x128x512xf32>, %arg1: memref<256x512x128xf32>, %arg2: memref<256x512x128xf32>, %arg3: memref<256x512x128xf32>) {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 0.12751743074602467 : f64
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 0xFF800000 : f32
    %0 = "loom.get_symbol"() {name = "block_size_0"} : () -> index
    %1 = "loom.get_symbol"() {name = "block_size_1"} : () -> index
    %2 = "loom.get_symbol"() {name = "block_size_3"} : () -> index
    affine.parallel (%arg4, %arg5) = (0, 0) to (256 ceildiv symbol(%0), 512 ceildiv symbol(%1)) {
      %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %4 = linalg.fill ins(%cst_3 : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %5 = linalg.fill ins(%cst_2 : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %6 = tensor.empty(%0, %1) : tensor<?x?x128xf32>
      %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
      %8 = arith.muli %arg4, %0 : index
      %9 = arith.muli %arg5, %1 : index
      %subview = memref.subview %arg2[%8, %9, 0] [%0, %1, 128] [1, 1, 1] : memref<256x512x128xf32> to memref<?x?x128xf32, strided<[65536, 128, 1], offset: ?>>
      %10 = bufferization.to_tensor %subview : memref<?x?x128xf32, strided<[65536, 128, 1], offset: ?>> to tensor<?x?x128xf32>
      %11:3 = affine.for %arg6 = 0 to #map()[%2] iter_args(%arg7 = %4, %arg8 = %5, %arg9 = %7) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
        %14 = arith.muli %arg6, %2 : index
        %subview_5 = memref.subview %arg0[%8, 0, %14] [%0, 128, %2] [1, 1, 1] : memref<256x128x512xf32> to memref<?x128x?xf32, strided<[65536, 512, 1], offset: ?>>
        %15 = bufferization.to_tensor %subview_5 : memref<?x128x?xf32, strided<[65536, 512, 1], offset: ?>> to tensor<?x128x?xf32>
        %16 = arith.index_cast %0 : index to i64
        %17 = arith.cmpi eq, %16, %16 : i64
        cf.assert %17, "mismatching contracting dimension"
        %18 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
        %19 = linalg.fill ins(%cst_1 : f32) outs(%18 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        %20 = linalg.batch_matmul ins(%10, %15 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        %21 = tensor.empty(%0, %1) : tensor<?x?xi64>
        %22 = linalg.fill ins(%c0_i64 : i64) outs(%21 : tensor<?x?xi64>) -> tensor<?x?xi64>
        %23:2 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%20 : tensor<?x?x?xf32>) outs(%4, %22 : tensor<?x?xf32>, tensor<?x?xi64>) {
        ^bb0(%in: f32, %out: f32, %out_11: i64):
          %41 = linalg.index 2 : index
          %42 = arith.index_cast %41 : index to i64
          %43 = arith.maximumf %in, %out : f32
          %44 = arith.cmpf ogt, %in, %out : f32
          %45 = arith.select %44, %42, %out_11 : i64
          linalg.yield %43, %45 : f32, i64
        } -> (tensor<?x?xf32>, tensor<?x?xi64>)
        %24 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%23#0 : tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %41 = arith.truncf %cst_0 : f64 to f32
          %42 = arith.mulf %in, %41 : f32
          linalg.yield %42 : f32
        } -> tensor<?x?xf32>
        %25 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg7, %24 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_11: f32, %out: f32):
          %41 = arith.cmpf ogt, %in, %in_11 : f32
          %42 = arith.select %41, %in, %in_11 : f32
          linalg.yield %42 : f32
        } -> tensor<?x?xf32>
        %26 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<?x?x?xf32>) outs(%18 : tensor<?x?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %41 = arith.truncf %cst_0 : f64 to f32
          %42 = arith.mulf %in, %41 : f32
          linalg.yield %42 : f32
        } -> tensor<?x?x?xf32>
        %extracted_slice_6 = tensor.extract_slice %25[0, 0] [%0, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %expanded_7 = tensor.expand_shape %extracted_slice_6 [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf32> into tensor<?x?x1xf32>
        %27 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %expanded_7 : tensor<?x?x?xf32>, tensor<?x?x1xf32>) outs(%18 : tensor<?x?x?xf32>) {
        ^bb0(%in: f32, %in_11: f32, %out: f32):
          %41 = arith.subf %in, %in_11 : f32
          linalg.yield %41 : f32
        } -> tensor<?x?x?xf32>
        %28 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27 : tensor<?x?x?xf32>) outs(%18 : tensor<?x?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %41 = math.powf %cst, %in : f32
          linalg.yield %41 : f32
        } -> tensor<?x?x?xf32>
        %29 = linalg.fill ins(%cst_1 : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %30 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%28 : tensor<?x?x?xf32>) outs(%29 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %41 = arith.addf %in, %out : f32
          linalg.yield %41 : f32
        } -> tensor<?x?xf32>
        %31 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg7, %25 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_11: f32, %out: f32):
          %41 = arith.subf %in, %in_11 : f32
          linalg.yield %41 : f32
        } -> tensor<?x?xf32>
        %32 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%31 : tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %41 = math.powf %cst, %in : f32
          linalg.yield %41 : f32
        } -> tensor<?x?xf32>
        %33 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg8, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_11: f32, %out: f32):
          %41 = arith.mulf %in, %in_11 : f32
          linalg.yield %41 : f32
        } -> tensor<?x?xf32>
        %34 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%33, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_11: f32, %out: f32):
          %41 = arith.addf %in, %in_11 : f32
          linalg.yield %41 : f32
        } -> tensor<?x?xf32>
        %extracted_slice_8 = tensor.extract_slice %32[0, 0] [%0, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %expanded_9 = tensor.expand_shape %extracted_slice_8 [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf32> into tensor<?x?x1xf32>
        %35 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %expanded_9 : tensor<?x?x128xf32>, tensor<?x?x1xf32>) outs(%6 : tensor<?x?x128xf32>) {
        ^bb0(%in: f32, %in_11: f32, %out: f32):
          %41 = arith.mulf %in, %in_11 : f32
          linalg.yield %41 : f32
        } -> tensor<?x?x128xf32>
        %subview_10 = memref.subview %arg1[%8, %14, 0] [%0, %2, 128] [1, 1, 1] : memref<256x512x128xf32> to memref<?x?x128xf32, strided<[65536, 128, 1], offset: ?>>
        %36 = bufferization.to_tensor %subview_10 : memref<?x?x128xf32, strided<[65536, 128, 1], offset: ?>> to tensor<?x?x128xf32>
        cf.assert %17, "mismatching contracting dimension"
        %37 = arith.index_cast %2 : index to i64
        %38 = arith.cmpi eq, %37, %37 : i64
        cf.assert %38, "mismatching contracting dimension"
        %39 = linalg.batch_matmul ins(%28, %36 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%7 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
        %40 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %35 : tensor<?x?x128xf32>, tensor<?x?x128xf32>) outs(%6 : tensor<?x?x128xf32>) {
        ^bb0(%in: f32, %in_11: f32, %out: f32):
          %41 = arith.addf %in, %in_11 : f32
          linalg.yield %41 : f32
        } -> tensor<?x?x128xf32>
        affine.yield %25, %34, %40 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
      }
      %extracted_slice = tensor.extract_slice %11#1[0, 0] [%0, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf32> into tensor<?x?x1xf32>
      %12 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11#2, %expanded : tensor<?x?x128xf32>, tensor<?x?x1xf32>) outs(%6 : tensor<?x?x128xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %14 = arith.divf %in, %in_5 : f32
        linalg.yield %14 : f32
      } -> tensor<?x?x128xf32>
      %subview_4 = memref.subview %arg3[%8, %9, 0] [%0, %1, 128] [1, 1, 1] : memref<256x512x128xf32> to memref<?x?x128xf32, strided<[65536, 128, 1], offset: ?>>
      %13 = bufferization.to_buffer %12 : tensor<?x?x128xf32> to memref<?x?x128xf32, strided<[65536, 128, 1], offset: ?>>
      memref.copy %13, %subview_4 : memref<?x?x128xf32, strided<[65536, 128, 1], offset: ?>> to memref<?x?x128xf32, strided<[65536, 128, 1], offset: ?>>
    }
    return
  }
}


mlir-opt validation succeeded.

