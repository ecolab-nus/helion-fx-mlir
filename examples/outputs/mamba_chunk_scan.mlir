=== Device IR ===
Graph 0: ForLoopGraphInfo
opcode         name               target                                     args                                                                                  kwargs
-------------  -----------------  -----------------------------------------  ------------------------------------------------------------------------------------  --------
placeholder    arg0_1             arg0_1                                     ()                                                                                    {}
placeholder    arg1_1             arg1_1                                     ()                                                                                    {}
call_function  _new_var           <function _new_var at 0x75cce4bed300>      (arg0_1,)                                                                             {}
call_function  _new_var_1         <function _new_var at 0x75cce4bed300>      (arg1_1,)                                                                             {}
call_function  block_size_4       <function _get_symnode at 0x75cce4ef6660>  ('block_size_4',)                                                                     {}
call_function  tile_begin         <function tile_begin at 0x75cc3f69a7a0>    (block_size_4,)                                                                       {}
call_function  block_size_5       <function _get_symnode at 0x75cce4ef6660>  ('block_size_5',)                                                                     {}
call_function  tile_begin_1       <function tile_begin at 0x75cc3f69a7a0>    (block_size_5,)                                                                       {}
call_function  block_size_3       <function _get_symnode at 0x75cce4ef6660>  ('block_size_3',)                                                                     {}
call_function  tile_begin_2       <function tile_begin at 0x75cc3f69a7a0>    (block_size_3,)                                                                       {}
call_function  x_size2            <function _get_symnode at 0x75cce4ef6660>  ('x_size2',)                                                                          {}
call_function  floordiv_1         <built-in function floordiv>               (tile_begin_2, x_size2)                                                               {}
call_function  cb                 <function _host_tensor at 0x75cce4ef7740>  ('cb',)                                                                               {}
call_function  sym_size_int       aten.sym_size.int                          (arg0_1, 0)                                                                           {}
call_function  block_size_2       <function _get_symnode at 0x75cce4ef6660>  ('block_size_2',)                                                                     {}
call_function  cb_local           <function load at 0x75cc3f67ec00>          (cb, [tile_begin, tile_begin_1, floordiv_1, sym_size_int, block_size_2], None, None)  {}
call_function  dA_cumsum          <function _host_tensor at 0x75cce4ef7740>  ('dA_cumsum',)                                                                        {}
call_function  load_1             <function load at 0x75cc3f67ec00>          (dA_cumsum, [tile_begin, tile_begin_2, tile_begin_1, block_size_2], None, None)       {}
call_function  dA_cumsum_local_k  prims.convert_element_type.default         (load_1, torch.float32)                                                               {}
call_function  subscript          <function subscript at 0x75cc3f6b4220>     (_new_var, [slice(None, None, None), None])                                           {}
call_function  mul                aten.mul.Tensor                            (subscript, 1.44269504)                                                               {}
call_function  subscript_1        <function subscript at 0x75cc3f6b4220>     (dA_cumsum_local_k, [None, slice(None, None, None)])                                  {}
call_function  mul_1              aten.mul.Tensor                            (subscript_1, 1.44269504)                                                             {}
call_function  sub                aten.sub.Tensor                            (mul, mul_1)                                                                          {}
call_function  exp2               aten.exp2.default                          (sub,)                                                                                {}
call_function  cb_local_1         aten.mul.Tensor                            (cb_local, exp2)                                                                      {}
call_function  dt                 <function _host_tensor at 0x75cce4ef7740>  ('dt',)                                                                               {}
call_function  load_2             <function load at 0x75cc3f67ec00>          (dt, [tile_begin, tile_begin_2, tile_begin_1, block_size_2], None, None)              {}
call_function  dt_local           prims.convert_element_type.default         (load_2, torch.float32)                                                               {}
call_function  subscript_2        <function subscript at 0x75cc3f6b4220>     (dt_local, [None, slice(None, None, None)])                                           {}
call_function  mul_3              aten.mul.Tensor                            (cb_local_1, subscript_2)                                                             {}
call_function  cb_local_2         prims.convert_element_type.default         (mul_3, torch.float16)                                                                {}
call_function  cb_size3           <function _get_symnode at 0x75cce4ef6660>  ('cb_size3',)                                                                         {}
call_function  mul_4              <built-in function mul>                    (tile_begin_1, cb_size3)                                                              {}
call_function  tile_index         <function tile_index at 0x75cc3f69a3e0>    (block_size_2,)                                                                       {}
call_function  add                aten.add.Tensor                            (tile_index, mul_4)                                                                   {}
call_function  x                  <function _host_tensor at 0x75cce4ef7740>  ('x',)                                                                                {}
call_function  sym_size_int_1     aten.sym_size.int                          (arg1_1, 1)                                                                           {}
call_function  x_local            <function load at 0x75cc3f67ec00>          (x, [tile_begin, add, tile_begin_2, sym_size_int_1], None, None)                      {}
call_function  acc_o              <function dot at 0x75cce4ef4cc0>           (cb_local_2, x_local, _new_var_1, None)                                               {}
output         output             output                                     ([acc_o],)                                                                            {}
Graph 1: RootGraphInfo
opcode         name                    target                                     args                                                                                                        kwargs
-------------  ----------------------  -----------------------------------------  ----------------------------------------------------------------------------------------------------------  --------
call_function  block_size_0            <function _get_symnode at 0x75cce4ef6660>  ('block_size_0',)                                                                                           {}
call_function  block_size_1            <function _get_symnode at 0x75cce4ef6660>  ('block_size_1',)                                                                                           {}
call_function  acc_o                   <function full at 0x75cc3f9063e0>          ([block_size_0, block_size_1], 0.0, torch.float32, None)                                                    {}
call_function  block_size_4            <function _get_symnode at 0x75cce4ef6660>  ('block_size_4',)                                                                                           {}
call_function  tile_begin              <function tile_begin at 0x75cc3f69a7a0>    (block_size_4,)                                                                                             {}
call_function  block_size_3            <function _get_symnode at 0x75cce4ef6660>  ('block_size_3',)                                                                                           {}
call_function  tile_begin_1            <function tile_begin at 0x75cc3f69a7a0>    (block_size_3,)                                                                                             {}
call_function  block_size_5            <function _get_symnode at 0x75cce4ef6660>  ('block_size_5',)                                                                                           {}
call_function  tile_begin_2            <function tile_begin at 0x75cc3f69a7a0>    (block_size_5,)                                                                                             {}
call_function  dA_cumsum               <function _host_tensor at 0x75cce4ef7740>  ('dA_cumsum',)                                                                                              {}
call_function  load                    <function load at 0x75cc3f67ec00>          (dA_cumsum, [tile_begin, tile_begin_1, tile_begin_2, block_size_0], None, None)                             {}
call_function  dA_cumsum_local_m       prims.convert_element_type.default         (load, torch.float32)                                                                                       {}
call_function  mul                     aten.mul.Tensor                            (dA_cumsum_local_m, 1.44269504)                                                                             {}
call_function  scale_m_local           aten.exp2.default                          (mul,)                                                                                                      {}
call_function  tile_index              <function tile_index at 0x75cc3f69a3e0>    (block_size_0,)                                                                                             {}
call_function  cb_size3                <function _get_symnode at 0x75cce4ef6660>  ('cb_size3',)                                                                                               {}
call_function  mul_1                   <built-in function mul>                    (tile_begin_2, cb_size3)                                                                                    {}
call_function  add                     aten.add.Tensor                            (tile_index, mul_1)                                                                                         {}
call_function  x_size2                 <function _get_symnode at 0x75cce4ef6660>  ('x_size2',)                                                                                                {}
call_function  floordiv_1              <built-in function floordiv>               (tile_begin_1, x_size2)                                                                                     {}
call_function  C                       <function _host_tensor at 0x75cce4ef7740>  ('C',)                                                                                                      {}
call_function  C_local                 <function load at 0x75cc3f67ec00>          (C, [tile_begin, add, floordiv_1, slice(None, None, None)], None, None)                                     {}
call_function  prev_states             <function _host_tensor at 0x75cce4ef7740>  ('prev_states',)                                                                                            {}
call_function  prev_states_local       <function load at 0x75cc3f67ec00>          (prev_states, [tile_begin, tile_begin_2, tile_begin_1, block_size_1, slice(None, None, None)], None, None)  {}
call_function  permute                 aten.permute.default                       (prev_states_local, [1, 0])                                                                                 {}
call_function  acc_o_1                 <function dot at 0x75cce4ef4cc0>           (C_local, permute, acc_o, None)                                                                             {}
call_function  subscript               <function subscript at 0x75cc3f6b4220>     (scale_m_local, [slice(None, None, None), None])                                                            {}
call_function  acc_o_2                 aten.mul.Tensor                            (acc_o_1, subscript)                                                                                        {}
call_function  tile_id                 <function tile_id at 0x75cc3f69b6a0>       (block_size_0,)                                                                                             {}
call_function  add_1                   <built-in function add>                    (tile_id, 1)                                                                                                {}
call_function  mul_3                   <built-in function mul>                    (add_1, block_size_0)                                                                                       {}
call_function  _for_loop               <function _for_loop at 0x75cce4ef7b00>     (0, [0], [mul_3], [dA_cumsum_local_m, acc_o_2])                                                             {}
call_function  getitem                 <built-in function getitem>                (_for_loop, 0)                                                                                              {}
call_function  _phi                    <function _phi at 0x75cce4bec0e0>          (acc_o_2, getitem)                                                                                          {}
call_function  D                       <function _host_tensor at 0x75cce4ef7740>  ('D',)                                                                                                      {}
call_function  load_3                  <function load at 0x75cc3f67ec00>          (D, [tile_begin_1], None, None)                                                                             {}
call_function  D_local                 prims.convert_element_type.default         (load_3, torch.float32)                                                                                     {}
call_function  tile_index_1            <function tile_index at 0x75cc3f69a3e0>    (block_size_0,)                                                                                             {}
call_function  add_2                   aten.add.Tensor                            (tile_index_1, mul_1)                                                                                       {}
call_function  x                       <function _host_tensor at 0x75cce4ef7740>  ('x',)                                                                                                      {}
call_function  load_4                  <function load at 0x75cc3f67ec00>          (x, [tile_begin, add_2, tile_begin_1, block_size_1], None, None)                                            {}
call_function  x_residual              prims.convert_element_type.default         (load_4, torch.float32)                                                                                     {}
call_function  mul_5                   aten.mul.Tensor                            (x_residual, D_local)                                                                                       {}
call_function  acc_o_3                 aten.add.Tensor                            (_phi, mul_5)                                                                                               {}
call_function  convert_element_type_3  prims.convert_element_type.default         (acc_o_3, torch.float16)                                                                                    {}
call_function  tile_index_2            <function tile_index at 0x75cc3f69a3e0>    (block_size_0,)                                                                                             {}
call_function  add_4                   aten.add.Tensor                            (tile_index_2, mul_1)                                                                                       {}
call_function  out                     <function _host_tensor at 0x75cce4ef7740>  ('out',)                                                                                                    {}
call_function  store                   <function store at 0x75cc3f67e7a0>         (out, [tile_begin, add_4, tile_begin_1, block_size_1], convert_element_type_3, None)                        {}
output         output                  output                                     (None,)                                                                                                     {}


=== Nodes with symbols ===
Node arg0_1 : FakeTensor(..., size=(u1,))
Node arg1_1 : FakeTensor(..., size=(u1, u2))
Node _new_var : FakeTensor(..., size=(u1,))
Node _new_var_1 : FakeTensor(..., size=(u1, u2))
Node block_size_4 : u11
Node tile_begin : u13
Node block_size_5 : u12
Node tile_begin_1 : u15
Node block_size_3 : u10
Node tile_begin_2 : u14
Node x_size2 : s53
Node floordiv_1 : (u14//s53)
Node cb : FakeTensor(..., size=(1, s64, 1, s56, s50), dtype=torch.float16)
Node sym_size_int : u1
Node block_size_2 : u3
Node cb_local : FakeTensor(..., size=(u1, u3), dtype=torch.float16)
Node dA_cumsum : FakeTensor(..., size=(1, s96, s42, s33), dtype=torch.float16)
Node load_1 : FakeTensor(..., size=(u3,), dtype=torch.float16)
Node dA_cumsum_local_k : FakeTensor(..., size=(u3,))
Node subscript : FakeTensor(..., size=(u1, 1))
Node mul : FakeTensor(..., size=(u1, 1))
Node subscript_1 : FakeTensor(..., size=(1, u3))
Node mul_1 : FakeTensor(..., size=(1, u3))
Node sub : FakeTensor(..., size=(u1, u3))
Node exp2 : FakeTensor(..., size=(u1, u3))
Node cb_local_1 : FakeTensor(..., size=(u1, u3))
Node dt : FakeTensor(..., size=(1, s99, s89, s68), dtype=torch.float16)
Node load_2 : FakeTensor(..., size=(u3,), dtype=torch.float16)
Node dt_local : FakeTensor(..., size=(u3,))
Node subscript_2 : FakeTensor(..., size=(1, u3))
Node mul_3 : FakeTensor(..., size=(u1, u3))
Node cb_local_2 : FakeTensor(..., size=(u1, u3), dtype=torch.float16)
Node cb_size3 : s56
Node mul_4 : s56*u15
Node tile_index : FakeTensor(..., size=(u3,), dtype=torch.int32)
Node add : FakeTensor(..., size=(u3,), dtype=torch.int32)
Node x : FakeTensor(..., size=(1, s27, s53, s0), dtype=torch.float16)
Node sym_size_int_1 : u2
Node x_local : FakeTensor(..., size=(u3, u2), dtype=torch.float16)
Node acc_o : FakeTensor(..., size=(u1, u2))
Node block_size_0 : u1
Node block_size_1 : u2
Node acc_o : FakeTensor(..., size=(u1, u2))
Node block_size_4 : u11
Node tile_begin : u13
Node block_size_3 : u10
Node tile_begin_1 : u14
Node block_size_5 : u12
Node tile_begin_2 : u15
Node dA_cumsum : FakeTensor(..., size=(1, s96, s42, s33), dtype=torch.float16)
Node load : FakeTensor(..., size=(u1,), dtype=torch.float16)
Node dA_cumsum_local_m : FakeTensor(..., size=(u1,))
Node mul : FakeTensor(..., size=(u1,))
Node scale_m_local : FakeTensor(..., size=(u1,))
Node tile_index : FakeTensor(..., size=(u1,), dtype=torch.int32)
Node cb_size3 : s56
Node mul_1 : s56*u15
Node add : FakeTensor(..., size=(u1,), dtype=torch.int32)
Node x_size2 : s53
Node floordiv_1 : (u14//s53)
Node C : FakeTensor(..., size=(1, s16, 1, 16), dtype=torch.float16)
Node C_local : FakeTensor(..., size=(u1, u16), dtype=torch.float16)
Node prev_states : FakeTensor(..., size=(1, s28, s61, s46, 16), dtype=torch.float16)
Node prev_states_local : FakeTensor(..., size=(u2, u16), dtype=torch.float16)
Node permute : FakeTensor(..., size=(u16, u2), dtype=torch.float16)
Node acc_o_1 : FakeTensor(..., size=(u1, u2))
Node subscript : FakeTensor(..., size=(u1, 1))
Node acc_o_2 : FakeTensor(..., size=(u1, u2))
Node tile_id : u17
Node add_1 : u17 + 1
Node mul_3 : u1*(u17 + 1)
Node _for_loop : [FakeTensor(..., size=(u1, u2))]
Node getitem : FakeTensor(..., size=(u1, u2))
Node _phi : FakeTensor(..., size=(u1, u2))
Node D : FakeTensor(..., size=(s26,), dtype=torch.float16)
Node load_3 : FakeTensor(..., size=(), dtype=torch.float16)
Node D_local : FakeTensor(..., size=())
Node tile_index_1 : FakeTensor(..., size=(u1,), dtype=torch.int32)
Node add_2 : FakeTensor(..., size=(u1,), dtype=torch.int32)
Node x : FakeTensor(..., size=(1, s27, s53, s0), dtype=torch.float16)
Node load_4 : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node x_residual : FakeTensor(..., size=(u1, u2))
Node mul_5 : FakeTensor(..., size=(u1, u2))
Node acc_o_3 : FakeTensor(..., size=(u1, u2))
Node convert_element_type_3 : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node tile_index_2 : FakeTensor(..., size=(u1,), dtype=torch.int32)
Node add_4 : FakeTensor(..., size=(u1,), dtype=torch.int32)
Node out : FakeTensor(..., size=(1, s27, s53, s0), dtype=torch.float16)


=== Compile Environment ===
Block Sizes (7):
  Block 0: Size=s56, Var=u1, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 1: Size=s0, Var=u2, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 2: Size=u1*(u17 + 1), Var=u3, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 3: Size=s53, Var=u10, Reduction=False, Source=FixedBlockSizeSource(value=1)
  Block 4: Size=1, Var=u11, Reduction=False, Source=FixedBlockSizeSource(value=1)
  Block 5: Size=s64, Var=u12, Reduction=False, Source=FixedBlockSizeSource(value=1)
  Block 6: Size=16, Var=u16, Reduction=True, Source=ReductionLoopBlockSizeSource(reduction_loop=0)
Shape Env (30):
  Var s64: 8
  Var s56: 256
  Var s50: 256
  Var s27: 2048
  Var s53: 16
  Var s0: 64
  Var s99: 16
  Var s89: 8
  Var s68: 256
  Var s96: 16
  Var s42: 8
  Var s33: 256
  Var s16: 2048
  Var s49: 16
  Var s28: 8
  Var s61: 16
  Var s46: 64
  Var s22: 16
  Var s26: 16
  Var u1: 256
  Var u2: 64
  Var u3: 8192
  Var u10: 64
  Var u11: 64
  Var u12: 64
  Var u13: 8192
  Var u14: 8192
  Var u15: 8192
  Var u16: 16
  Var u17: 8192


=== MLIR Dump ===
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 floordiv 16)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
#map4 = affine_map<(d0)[s0] -> (d0 ceildiv s0)>
#map5 = affine_map<(d0, d1) -> (0, d1)>
#map6 = affine_map<() -> ()>
#map7 = affine_map<(d0, d1) -> ()>
module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x?xf16>, %arg5: memref<1x8x16x64x?xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.44269504 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 64 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 8192 : index} : () -> index
    %3 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_h, upper_bound = 16 : index} : () -> index
    %4 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_c, upper_bound = 8 : index} : () -> index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (16 ceildiv symbol(%3), 256 ceildiv symbol(%0), 64 ceildiv symbol(%1), 1, 8 ceildiv symbol(%4)) {
      %5 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %6 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %7 = arith.muli %arg8, %3 : index
      %8 = arith.muli %arg12, %4 : index
      %9 = arith.muli %arg9, %0 : index
      %subview = memref.subview %arg1[%arg11, %7, %8, %9] [1, 1, 1, %0] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      %10 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
      %11 = tensor.empty(%0) : tensor<?xf32>
      %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%10 : tensor<?xf16>) outs(%11 : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %37 = arith.extf %in : f16 to f32
        linalg.yield %37 : f32
      } -> tensor<?xf32>
      %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%12 : tensor<?xf32>) outs(%11 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %37 = arith.truncf %cst_0 : f64 to f32
        %38 = arith.mulf %in, %37 : f32
        linalg.yield %38 : f32
      } -> tensor<?xf32>
      %14 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%13 : tensor<?xf32>) outs(%11 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %37 = math.powf %cst, %in : f32
        linalg.yield %37 : f32
      } -> tensor<?xf32>
      %15 = arith.muli %8, %c256 : index
      %16 = affine.apply #map1(%7)
      %dim = memref.dim %arg4, %c3 : memref<1x2048x1x?xf16>
      %subview_2 = memref.subview %arg4[%arg11, %15, %16, 0] [1, %0, 1, %dim] [1, 1, 1, 1] : memref<1x2048x1x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %17 = bufferization.to_tensor %subview_2 : memref<?x?xf16, strided<[?, 1], offset: ?>> to tensor<?x?xf16>
      %18 = arith.muli %arg10, %1 : index
      %dim_3 = memref.dim %arg5, %c4 : memref<1x8x16x64x?xf16>
      %subview_4 = memref.subview %arg5[%arg11, %8, %7, %18, 0] [1, 1, 1, %1, %dim_3] [1, 1, 1, 1, 1] : memref<1x8x16x64x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %19 = bufferization.to_tensor %subview_4 : memref<?x?xf16, strided<[?, 1], offset: ?>> to tensor<?x?xf16>
      %20 = tensor.empty(%dim_3, %1) : tensor<?x?xf16>
      %transposed = linalg.transpose ins(%19 : tensor<?x?xf16>) outs(%20 : tensor<?x?xf16>) permutation = [1, 0] 
      %21 = linalg.matmul ins(%17, %transposed : tensor<?x?xf16>, tensor<?x?xf16>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %extracted_slice = tensor.extract_slice %14[0] [%0] [1] : tensor<?xf32> to tensor<?xf32>
      %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [%0, 1] : tensor<?xf32> into tensor<?x1xf32>
      %22 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%21, %expanded : tensor<?x?xf32>, tensor<?x1xf32>) outs(%5 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %37 = arith.mulf %in, %in_8 : f32
        linalg.yield %37 : f32
      } -> tensor<?x?xf32>
      %23 = arith.addi %arg9, %c1 : index
      %24 = arith.muli %23, %0 : index
      %25 = affine.apply #map4(%24)[%2]
      %26 = scf.for %arg13 = %c0 to %25 step %c1 iter_args(%arg14 = %22) -> (tensor<?x?xf32>) {
        %37 = arith.muli %arg13, %2 : index
        %subview_8 = memref.subview %arg0[%arg11, %8, %16, %9, %37] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %38 = bufferization.to_tensor %subview_8 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %subview_9 = memref.subview %arg1[%arg11, %7, %8, %37] [1, 1, 1, %2] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %39 = bufferization.to_tensor %subview_9 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %40 = tensor.empty(%2) : tensor<?xf32>
        %41 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%39 : tensor<?xf16>) outs(%40 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %57 = arith.extf %in : f16 to f32
          linalg.yield %57 : f32
        } -> tensor<?xf32>
        %extracted_slice_10 = tensor.extract_slice %12[0] [%0] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_11 = tensor.expand_shape %extracted_slice_10 [[0, 1]] output_shape [%0, 1] : tensor<?xf32> into tensor<?x1xf32>
        %42 = tensor.empty(%0) : tensor<?x1xf32>
        %43 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%expanded_11 : tensor<?x1xf32>) outs(%42 : tensor<?x1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %57 = arith.truncf %cst_0 : f64 to f32
          %58 = arith.mulf %in, %57 : f32
          linalg.yield %58 : f32
        } -> tensor<?x1xf32>
        %extracted_slice_12 = tensor.extract_slice %41[0] [%2] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_13 = tensor.expand_shape %extracted_slice_12 [[0, 1]] output_shape [1, %2] : tensor<?xf32> into tensor<1x?xf32>
        %44 = tensor.empty(%2) : tensor<1x?xf32>
        %45 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%expanded_13 : tensor<1x?xf32>) outs(%44 : tensor<1x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %57 = arith.truncf %cst_0 : f64 to f32
          %58 = arith.mulf %in, %57 : f32
          linalg.yield %58 : f32
        } -> tensor<1x?xf32>
        %46 = tensor.empty(%0, %2) : tensor<?x?xf32>
        %47 = linalg.generic {indexing_maps = [#map3, #map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%43, %45 : tensor<?x1xf32>, tensor<1x?xf32>) outs(%46 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_18: f32, %out: f32):
          %57 = arith.subf %in, %in_18 : f32
          linalg.yield %57 : f32
        } -> tensor<?x?xf32>
        %48 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%47 : tensor<?x?xf32>) outs(%46 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %57 = math.powf %cst, %in : f32
          linalg.yield %57 : f32
        } -> tensor<?x?xf32>
        %49 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%38, %48 : tensor<?x?xf16>, tensor<?x?xf32>) outs(%46 : tensor<?x?xf32>) {
        ^bb0(%in: f16, %in_18: f32, %out: f32):
          %57 = arith.extf %in : f16 to f32
          %58 = arith.mulf %57, %in_18 : f32
          linalg.yield %58 : f32
        } -> tensor<?x?xf32>
        %subview_14 = memref.subview %arg2[%arg11, %7, %8, %37] [1, 1, 1, %2] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %50 = bufferization.to_tensor %subview_14 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %51 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%50 : tensor<?xf16>) outs(%40 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %57 = arith.extf %in : f16 to f32
          linalg.yield %57 : f32
        } -> tensor<?xf32>
        %extracted_slice_15 = tensor.extract_slice %51[0] [%2] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_16 = tensor.expand_shape %extracted_slice_15 [[0, 1]] output_shape [1, %2] : tensor<?xf32> into tensor<1x?xf32>
        %52 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%49, %expanded_16 : tensor<?x?xf32>, tensor<1x?xf32>) outs(%46 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_18: f32, %out: f32):
          %57 = arith.mulf %in, %in_18 : f32
          linalg.yield %57 : f32
        } -> tensor<?x?xf32>
        %53 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %54 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%52 : tensor<?x?xf32>) outs(%53 : tensor<?x?xf16>) {
        ^bb0(%in: f32, %out: f16):
          %57 = arith.truncf %in : f32 to f16
          linalg.yield %57 : f16
        } -> tensor<?x?xf16>
        %subview_17 = memref.subview %arg3[%arg11, %15, %7, %18] [1, %2, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %55 = bufferization.to_tensor %subview_17 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
        %56 = linalg.matmul ins(%54, %55 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        scf.yield %56 : tensor<?x?xf32>
      }
      %subview_5 = memref.subview %arg6[%7] [1] [1] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
      %27 = bufferization.to_tensor %subview_5 : memref<f16, strided<[], offset: ?>> to tensor<f16>
      %28 = tensor.empty() : tensor<f32>
      %29 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = []} ins(%27 : tensor<f16>) outs(%28 : tensor<f32>) {
      ^bb0(%in: f16, %out: f32):
        %37 = arith.extf %in : f16 to f32
        linalg.yield %37 : f32
      } -> tensor<f32>
      %subview_6 = memref.subview %arg3[%arg11, %15, %7, %18] [1, %0, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %30 = bufferization.to_tensor %subview_6 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
      %31 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%30 : tensor<?x?xf16>) outs(%5 : tensor<?x?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %37 = arith.extf %in : f16 to f32
        linalg.yield %37 : f32
      } -> tensor<?x?xf32>
      %32 = linalg.generic {indexing_maps = [#map2, #map7, #map2], iterator_types = ["parallel", "parallel"]} ins(%31, %29 : tensor<?x?xf32>, tensor<f32>) outs(%5 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %37 = arith.mulf %in, %in_8 : f32
        linalg.yield %37 : f32
      } -> tensor<?x?xf32>
      %33 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%26, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %37 = arith.addf %in, %in_8 : f32
        linalg.yield %37 : f32
      } -> tensor<?x?xf32>
      %34 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %35 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%33 : tensor<?x?xf32>) outs(%34 : tensor<?x?xf16>) {
      ^bb0(%in: f32, %out: f16):
        %37 = arith.truncf %in : f32 to f16
        linalg.yield %37 : f16
      } -> tensor<?x?xf16>
      %subview_7 = memref.subview %arg7[%arg11, %15, %7, %18] [1, %0, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %36 = bufferization.to_buffer %35 : tensor<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      memref.copy %36, %subview_7 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
    }
    return
  }
}


mlir-opt validation succeeded.

