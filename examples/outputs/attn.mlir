=== Device IR ===
Graph 0: ForLoopGraphInfo
opcode         name          target                                     args                                                                         kwargs
-------------  ------------  -----------------------------------------  ---------------------------------------------------------------------------  --------
placeholder    arg0_1        arg0_1                                     ()                                                                           {}
placeholder    arg1_1        arg1_1                                     ()                                                                           {}
placeholder    arg2_1        arg2_1                                     ()                                                                           {}
placeholder    arg3_1        arg3_1                                     ()                                                                           {}
placeholder    arg4_1        arg4_1                                     ()                                                                           {}
call_function  _new_var      <function _new_var at 0x7cdf5241d260>      (arg0_1,)                                                                    {}
call_function  _new_var_1    <function _new_var at 0x7cdf5241d260>      (arg1_1,)                                                                    {}
call_function  _new_var_2    <function _new_var at 0x7cdf5241d260>      (arg2_1,)                                                                    {}
call_function  _new_var_3    <function _new_var at 0x7cdf5241d260>      (arg3_1,)                                                                    {}
call_function  _new_var_4    <function _new_var at 0x7cdf5241d260>      (arg4_1,)                                                                    {}
call_function  k_view        <function _host_tensor at 0x7cdf523976a0>  ('k_view',)                                                                  {}
call_function  sym_size_int  aten.sym_size.int                          (arg0_1, 0)                                                                  {}
call_function  block_size_3  <function _get_symnode at 0x7cdf523965c0>  ('block_size_3',)                                                            {}
call_function  k             <function load at 0x7cdeaceceb60>          (k_view, [sym_size_int, slice(None, None, None), block_size_3], None, None)  {}
call_function  qk            aten.bmm.default                           (_new_var, k)                                                                {}
call_function  _mask_to_2    <function _mask_to at 0x7cdf5241cea0>      (qk, -inf)                                                                   {}
call_function  amax          aten.amax.default                          (_mask_to_2, [-1])                                                           {}
call_function  mul           aten.mul.Tensor                            (amax, _new_var_2)                                                           {}
call_function  m_ij          aten.maximum.default                       (_new_var_1, mul)                                                            {}
call_function  mul_1         aten.mul.Tensor                            (qk, _new_var_2)                                                             {}
call_function  subscript     <function subscript at 0x7cdeacf04180>     (m_ij, [slice(None, None, None), slice(None, None, None), None])             {}
call_function  qk_1          aten.sub.Tensor                            (mul_1, subscript)                                                           {}
call_function  exp2          aten.exp2.default                          (qk_1,)                                                                      {}
call_function  _mask_to_3    <function _mask_to at 0x7cdf5241cea0>      (exp2, 0)                                                                    {}
call_function  l_ij          aten.sum.dim_IntList                       (_mask_to_3, [-1])                                                           {}
call_function  sub_1         aten.sub.Tensor                            (_new_var_1, m_ij)                                                           {}
call_function  alpha         aten.exp2.default                          (sub_1,)                                                                     {}
call_function  mul_2         aten.mul.Tensor                            (_new_var_3, alpha)                                                          {}
call_function  l_i           aten.add.Tensor                            (mul_2, l_ij)                                                                {}
call_function  subscript_1   <function subscript at 0x7cdeacf04180>     (alpha, [slice(None, None, None), slice(None, None, None), None])            {}
call_function  acc           aten.mul.Tensor                            (_new_var_4, subscript_1)                                                    {}
call_function  v_view        <function _host_tensor at 0x7cdf523976a0>  ('v_view',)                                                                  {}
call_function  v             <function load at 0x7cdeaceceb60>          (v_view, [sym_size_int, block_size_3, slice(None, None, None)], None, None)  {}
call_function  acc_1         aten.baddbmm.default                       (acc, _mask_to_3, v)                                                         {}
call_function  m_i           <function _new_var at 0x7cdf5241d260>      (m_ij,)                                                                      {}
output         output        output                                     ([m_i, l_i, acc_1],)                                                         {}
Graph 1: RootGraphInfo
opcode         name          target                                     args                                                                         kwargs
-------------  ------------  -----------------------------------------  ---------------------------------------------------------------------------  ----------------------------------------------------------------------------------------------------
call_function  qk_scale_dev  <function full at 0x7cdead162340>          ([], 0.12751743074602467, torch.float16, None)                               {}
call_function  block_size_0  <function _get_symnode at 0x7cdf523965c0>  ('block_size_0',)                                                            {}
call_function  block_size_1  <function _get_symnode at 0x7cdf523965c0>  ('block_size_1',)                                                            {}
call_function  m_i           <function full at 0x7cdead162340>          ([block_size_0, block_size_1], -inf, torch.float16, None)                    {}
call_function  l_i           aten.full.default                          ([block_size_0, block_size_1], 1.0)                                          {'dtype': torch.float16, 'layout': torch.strided, 'device': device(type='cpu'), 'pin_memory': False}
call_function  acc           <function full at 0x7cdead162340>          ([block_size_0, block_size_1, 128], 0.0, torch.float16, None)                {}
call_function  q_view        <function _host_tensor at 0x7cdf523976a0>  ('q_view',)                                                                  {}
call_function  q             <function load at 0x7cdeaceceb60>          (q_view, [block_size_0, block_size_1, slice(None, None, None)], None, None)  {}
call_function  k_in_size1    <function _get_symnode at 0x7cdf523965c0>  ('k_in_size1',)                                                              {}
call_function  _for_loop     <function _for_loop at 0x7cdf52397a60>     (0, [0], [k_in_size1], [q, m_i, qk_scale_dev, l_i, acc])                     {}
call_function  getitem       <built-in function getitem>                (_for_loop, 0)                                                               {}
call_function  getitem_1     <built-in function getitem>                (_for_loop, 1)                                                               {}
call_function  getitem_2     <built-in function getitem>                (_for_loop, 2)                                                               {}
call_function  _phi          <function _phi at 0x7cdf5241c040>          (m_i, getitem)                                                               {}
call_function  _phi_1        <function _phi at 0x7cdf5241c040>          (l_i, getitem_1)                                                             {}
call_function  _phi_2        <function _phi at 0x7cdf5241c040>          (acc, getitem_2)                                                             {}
call_function  subscript     <function subscript at 0x7cdeacf04180>     (_phi_1, [slice(None, None, None), slice(None, None, None), None])           {}
call_function  acc_1         aten.div.Tensor                            (_phi_2, subscript)                                                          {}
call_function  out           <function _host_tensor at 0x7cdf523976a0>  ('out',)                                                                     {}
call_function  store         <function store at 0x7cdeacece700>         (out, [block_size_0, block_size_1, slice(None, None, None)], acc_1, None)    {}
output         output        output                                     (None,)                                                                      {}


=== Nodes with symbols ===
Node arg0_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node arg1_1 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node arg2_1 : FakeTensor(..., size=(), dtype=torch.float16)
Node arg3_1 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node arg4_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node _new_var_1 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node _new_var_2 : FakeTensor(..., size=(), dtype=torch.float16)
Node _new_var_3 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node _new_var_4 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node k_view : FakeTensor(..., size=(s35, 128, s34), dtype=torch.float16)
Node sym_size_int : u4
Node block_size_3 : u7
Node k : FakeTensor(..., size=(u4, 128, u7), dtype=torch.float16)
Node qk : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node _mask_to_2 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node amax : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node mul : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node m_ij : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node mul_1 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node subscript : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node qk_1 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node exp2 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node _mask_to_3 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node l_ij : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node sub_1 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node alpha : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node mul_2 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node subscript_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node v_view : FakeTensor(..., size=(s80, s34, 128), dtype=torch.float16)
Node v : FakeTensor(..., size=(u4, u7, 128), dtype=torch.float16)
Node acc_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node m_i : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node qk_scale_dev : FakeTensor(..., size=(), dtype=torch.float16)
Node block_size_0 : u4
Node block_size_1 : u5
Node m_i : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node q_view : FakeTensor(..., size=(s30, s48, 128), dtype=torch.float16)
Node q : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node k_in_size1 : s34
Node _for_loop : [FakeTensor(..., size=(u4, u5), dtype=torch.float16), FakeTensor(..., size=(u4, u5), dtype=torch.float16), FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)]
Node getitem : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node getitem_1 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node getitem_2 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node _phi : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node _phi_1 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node _phi_2 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node subscript : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node acc_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node out : FakeTensor(..., size=(s30, s48, 128), dtype=torch.float16)
Node arg0_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node arg1_1 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node arg2_1 : FakeTensor(..., size=(), dtype=torch.float16)
Node arg3_1 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node arg4_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node _new_var_1 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node _new_var_2 : FakeTensor(..., size=(), dtype=torch.float16)
Node _new_var_3 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node _new_var_4 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node k_view : FakeTensor(..., size=(s35, 128, s34), dtype=torch.float16)
Node sym_size_int : u4
Node block_size_3 : u7
Node k : FakeTensor(..., size=(u4, 128, u7), dtype=torch.float16)
Node qk : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node _mask_to_2 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node amax : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node mul : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node m_ij : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node mul_1 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node subscript : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node qk_1 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node exp2 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node _mask_to_3 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node l_ij : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node sub_1 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node alpha : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node mul_2 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node subscript_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node v_view : FakeTensor(..., size=(s80, s34, 128), dtype=torch.float16)
Node v : FakeTensor(..., size=(u4, u7, 128), dtype=torch.float16)
Node acc_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node m_i : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node qk_scale_dev : FakeTensor(..., size=(), dtype=torch.float16)
Node block_size_0 : u4
Node block_size_1 : u5
Node m_i : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node q_view : FakeTensor(..., size=(s30, s48, 128), dtype=torch.float16)
Node q : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node k_in_size1 : s34
Node _for_loop : [FakeTensor(..., size=(u4, u5), dtype=torch.float16), FakeTensor(..., size=(u4, u5), dtype=torch.float16), FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)]
Node getitem : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node getitem_1 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node getitem_2 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node _phi : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node _phi_1 : FakeTensor(..., size=(u4, u5), dtype=torch.float16)
Node _phi_2 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node subscript : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node acc_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node out : FakeTensor(..., size=(s30, s48, 128), dtype=torch.float16)


=== Compile Environment ===
Block Sizes (4):
  Block 0: Size=s30, Var=u4, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 1: Size=s48, Var=u5, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 2: Size=128, Var=128, Reduction=True, Source=ReductionLoopBlockSizeSource(reduction_loop=0)
  Block 3: Size=s34, Var=u7, Reduction=False, Source=LoopSpecBlockSizeSource()
Shape Env (13):
  Var s30: 32
  Var s48: 4096
  Var s22: 128
  Var s35: 32
  Var s34: 4096
  Var s4: 128
  Var s80: 32
  Var s41: 4096
  Var s66: 128
  Var u4: 64
  Var u5: 64
  Var u6: 128
  Var u7: 64


=== MLIR Dump ===
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @attention(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
    %cst = arith.constant 2.000000e+00 : f16
    %c0_i64 = arith.constant 0 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.000000e+00 : f16
    %cst_2 = arith.constant 0xFC00 : f16
    %cst_3 = arith.constant 1.275630e-01 : f16
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 32 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 4096 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 4096 : index} : () -> index
    %3 = arith.ceildivui %c32, %0 : index
    %4 = arith.ceildivui %c4096, %1 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %6 = linalg.fill ins(%cst_2 : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %7 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %8 = tensor.empty(%0, %1) : tensor<?x?x128xf16>
      %9 = linalg.fill ins(%cst_0 : f16) outs(%8 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %10 = arith.muli %arg4, %0 : index
      %11 = arith.muli %arg5, %1 : index
      %subview = memref.subview %arg2[%10, %11, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %12 = bufferization.to_tensor %subview : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
      %13 = arith.ceildivui %c4096, %2 : index
      %14:3 = scf.for %arg6 = %c0 to %13 step %c1 iter_args(%arg7 = %6, %arg8 = %7, %arg9 = %9) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) {
        %17 = arith.muli %arg6, %2 : index
        %subview_5 = memref.subview %arg0[%10, 0, %17] [%0, 128, %2] [1, 1, 1] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
        %18 = bufferization.to_tensor %subview_5 : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to tensor<?x128x?xf16>
        %19 = arith.index_cast %0 : index to i64
        %20 = arith.cmpi eq, %19, %19 : i64
        cf.assert %20, "mismatching contracting dimension"
        %21 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf16>
        %22 = linalg.fill ins(%cst_0 : f16) outs(%21 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %23 = linalg.batch_matmul ins(%12, %18 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%22 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %24 = tensor.empty(%0, %1) : tensor<?x?xi64>
        %25 = linalg.fill ins(%c0_i64 : i64) outs(%24 : tensor<?x?xi64>) -> tensor<?x?xi64>
        %26:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%23 : tensor<?x?x?xf16>) outs(%6, %25 : tensor<?x?xf16>, tensor<?x?xi64>) {
        ^bb0(%in: f16, %out: f16, %out_11: i64):
          %44 = linalg.index 2 : index
          %45 = arith.index_cast %44 : index to i64
          %46 = arith.maximumf %in, %out : f16
          %47 = arith.cmpf ogt, %in, %out : f16
          %48 = arith.select %47, %45, %out_11 : i64
          linalg.yield %46, %48 : f16, i64
        } -> (tensor<?x?xf16>, tensor<?x?xi64>)
        %27 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%26#0 : tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = arith.mulf %in, %cst_3 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?xf16>
        %28 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg7, %27 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.cmpf ogt, %in, %in_11 : f16
          %45 = arith.select %44, %in, %in_11 : f16
          linalg.yield %45 : f16
        } -> tensor<?x?xf16>
        %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23 : tensor<?x?x?xf16>) outs(%21 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = arith.mulf %in, %cst_3 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x?xf16>
        %extracted_slice_6 = tensor.extract_slice %28[0, 0] [%0, %1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %expanded_7 = tensor.expand_shape %extracted_slice_6 [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf16> into tensor<?x?x1xf16>
        %30 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %expanded_7 : tensor<?x?x?xf16>, tensor<?x?x1xf16>) outs(%21 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.subf %in, %in_11 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x?xf16>
        %31 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30 : tensor<?x?x?xf16>) outs(%21 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = math.powf %cst, %in : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x?xf16>
        %32 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %33 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%31 : tensor<?x?x?xf16>) outs(%32 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = arith.addf %in, %out : f16
          linalg.yield %44 : f16
        } -> tensor<?x?xf16>
        %34 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg7, %28 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.subf %in, %in_11 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?xf16>
        %35 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%34 : tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = math.powf %cst, %in : f16
          linalg.yield %44 : f16
        } -> tensor<?x?xf16>
        %36 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg8, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.mulf %in, %in_11 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?xf16>
        %37 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%36, %33 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.addf %in, %in_11 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?xf16>
        %extracted_slice_8 = tensor.extract_slice %35[0, 0] [%0, %1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %expanded_9 = tensor.expand_shape %extracted_slice_8 [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf16> into tensor<?x?x1xf16>
        %38 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %expanded_9 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%8 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.mulf %in, %in_11 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x128xf16>
        %subview_10 = memref.subview %arg1[%10, %17, 0] [%0, %2, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        %39 = bufferization.to_tensor %subview_10 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        cf.assert %20, "mismatching contracting dimension"
        %40 = arith.index_cast %2 : index to i64
        %41 = arith.cmpi eq, %40, %40 : i64
        cf.assert %41, "mismatching contracting dimension"
        %42 = linalg.batch_matmul ins(%31, %39 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%9 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %43 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %38 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%8 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.addf %in, %in_11 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x128xf16>
        scf.yield %28, %37, %43 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
      }
      %extracted_slice = tensor.extract_slice %14#1[0, 0] [%0, %1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf16> into tensor<?x?x1xf16>
      %15 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14#2, %expanded : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%8 : tensor<?x?x128xf16>) {
      ^bb0(%in: f16, %in_5: f16, %out: f16):
        %17 = arith.divf %in, %in_5 : f16
        linalg.yield %17 : f16
      } -> tensor<?x?x128xf16>
      %subview_4 = memref.subview %arg3[%10, %11, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %16 = bufferization.to_buffer %15 : tensor<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      memref.copy %16, %subview_4 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
    }
    return
  }
}


mlir-opt validation succeeded.

