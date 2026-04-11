=== Device IR ===
Graph 0: ForLoopGraphInfo
opcode         name               target                                     args                                                                                  kwargs
-------------  -----------------  -----------------------------------------  ------------------------------------------------------------------------------------  --------
placeholder    arg0_1             arg0_1                                     ()                                                                                    {}
placeholder    arg1_1             arg1_1                                     ()                                                                                    {}
call_function  _new_var           <function _new_var at 0x771dfc5f5300>      (arg0_1,)                                                                             {}
call_function  _new_var_1         <function _new_var at 0x771dfc5f5300>      (arg1_1,)                                                                             {}
call_function  block_size_4       <function _get_symnode at 0x771dfc572660>  ('block_size_4',)                                                                     {}
call_function  tile_begin         <function tile_begin at 0x771d570b67a0>    (block_size_4,)                                                                       {}
call_function  block_size_5       <function _get_symnode at 0x771dfc572660>  ('block_size_5',)                                                                     {}
call_function  tile_begin_1       <function tile_begin at 0x771d570b67a0>    (block_size_5,)                                                                       {}
call_function  block_size_3       <function _get_symnode at 0x771dfc572660>  ('block_size_3',)                                                                     {}
call_function  tile_begin_2       <function tile_begin at 0x771d570b67a0>    (block_size_3,)                                                                       {}
call_function  x_size2            <function _get_symnode at 0x771dfc572660>  ('x_size2',)                                                                          {}
call_function  floordiv_1         <built-in function floordiv>               (tile_begin_2, x_size2)                                                               {}
call_function  cb                 <function _host_tensor at 0x771dfc573740>  ('cb',)                                                                               {}
call_function  sym_size_int       aten.sym_size.int                          (arg0_1, 0)                                                                           {}
call_function  block_size_2       <function _get_symnode at 0x771dfc572660>  ('block_size_2',)                                                                     {}
call_function  cb_local           <function load at 0x771d5709ac00>          (cb, [tile_begin, tile_begin_1, floordiv_1, sym_size_int, block_size_2], None, None)  {}
call_function  dA_cumsum          <function _host_tensor at 0x771dfc573740>  ('dA_cumsum',)                                                                        {}
call_function  load_1             <function load at 0x771d5709ac00>          (dA_cumsum, [tile_begin, tile_begin_2, tile_begin_1, block_size_2], None, None)       {}
call_function  dA_cumsum_local_k  prims.convert_element_type.default         (load_1, torch.float32)                                                               {}
call_function  subscript          <function subscript at 0x771d570d0220>     (_new_var, [slice(None, None, None), None])                                           {}
call_function  mul                aten.mul.Tensor                            (subscript, 1.44269504)                                                               {}
call_function  subscript_1        <function subscript at 0x771d570d0220>     (dA_cumsum_local_k, [None, slice(None, None, None)])                                  {}
call_function  mul_1              aten.mul.Tensor                            (subscript_1, 1.44269504)                                                             {}
call_function  sub                aten.sub.Tensor                            (mul, mul_1)                                                                          {}
call_function  exp2               aten.exp2.default                          (sub,)                                                                                {}
call_function  cb_local_1         aten.mul.Tensor                            (cb_local, exp2)                                                                      {}
call_function  dt                 <function _host_tensor at 0x771dfc573740>  ('dt',)                                                                               {}
call_function  load_2             <function load at 0x771d5709ac00>          (dt, [tile_begin, tile_begin_2, tile_begin_1, block_size_2], None, None)              {}
call_function  dt_local           prims.convert_element_type.default         (load_2, torch.float32)                                                               {}
call_function  subscript_2        <function subscript at 0x771d570d0220>     (dt_local, [None, slice(None, None, None)])                                           {}
call_function  mul_3              aten.mul.Tensor                            (cb_local_1, subscript_2)                                                             {}
call_function  cb_local_2         prims.convert_element_type.default         (mul_3, torch.float16)                                                                {}
call_function  cb_size3           <function _get_symnode at 0x771dfc572660>  ('cb_size3',)                                                                         {}
call_function  mul_4              <built-in function mul>                    (tile_begin_1, cb_size3)                                                              {}
call_function  tile_index         <function tile_index at 0x771d570b63e0>    (block_size_2,)                                                                       {}
call_function  add                aten.add.Tensor                            (tile_index, mul_4)                                                                   {}
call_function  x                  <function _host_tensor at 0x771dfc573740>  ('x',)                                                                                {}
call_function  sym_size_int_1     aten.sym_size.int                          (arg1_1, 1)                                                                           {}
call_function  x_local            <function load at 0x771d5709ac00>          (x, [tile_begin, add, tile_begin_2, sym_size_int_1], None, None)                      {}
call_function  acc_o              <function dot at 0x771dfc570cc0>           (cb_local_2, x_local, _new_var_1, None)                                               {}
output         output             output                                     ([acc_o],)                                                                            {}
Graph 1: RootGraphInfo
opcode         name                    target                                     args                                                                                                        kwargs
-------------  ----------------------  -----------------------------------------  ----------------------------------------------------------------------------------------------------------  --------
call_function  block_size_0            <function _get_symnode at 0x771dfc572660>  ('block_size_0',)                                                                                           {}
call_function  block_size_1            <function _get_symnode at 0x771dfc572660>  ('block_size_1',)                                                                                           {}
call_function  acc_o                   <function full at 0x771d5733a3e0>          ([block_size_0, block_size_1], 0.0, torch.float32, None)                                                    {}
call_function  block_size_4            <function _get_symnode at 0x771dfc572660>  ('block_size_4',)                                                                                           {}
call_function  tile_begin              <function tile_begin at 0x771d570b67a0>    (block_size_4,)                                                                                             {}
call_function  block_size_3            <function _get_symnode at 0x771dfc572660>  ('block_size_3',)                                                                                           {}
call_function  tile_begin_1            <function tile_begin at 0x771d570b67a0>    (block_size_3,)                                                                                             {}
call_function  block_size_5            <function _get_symnode at 0x771dfc572660>  ('block_size_5',)                                                                                           {}
call_function  tile_begin_2            <function tile_begin at 0x771d570b67a0>    (block_size_5,)                                                                                             {}
call_function  dA_cumsum               <function _host_tensor at 0x771dfc573740>  ('dA_cumsum',)                                                                                              {}
call_function  load                    <function load at 0x771d5709ac00>          (dA_cumsum, [tile_begin, tile_begin_1, tile_begin_2, block_size_0], None, None)                             {}
call_function  dA_cumsum_local_m       prims.convert_element_type.default         (load, torch.float32)                                                                                       {}
call_function  mul                     aten.mul.Tensor                            (dA_cumsum_local_m, 1.44269504)                                                                             {}
call_function  scale_m_local           aten.exp2.default                          (mul,)                                                                                                      {}
call_function  tile_index              <function tile_index at 0x771d570b63e0>    (block_size_0,)                                                                                             {}
call_function  cb_size3                <function _get_symnode at 0x771dfc572660>  ('cb_size3',)                                                                                               {}
call_function  mul_1                   <built-in function mul>                    (tile_begin_2, cb_size3)                                                                                    {}
call_function  add                     aten.add.Tensor                            (tile_index, mul_1)                                                                                         {}
call_function  x_size2                 <function _get_symnode at 0x771dfc572660>  ('x_size2',)                                                                                                {}
call_function  floordiv_1              <built-in function floordiv>               (tile_begin_1, x_size2)                                                                                     {}
call_function  C                       <function _host_tensor at 0x771dfc573740>  ('C',)                                                                                                      {}
call_function  C_local                 <function load at 0x771d5709ac00>          (C, [tile_begin, add, floordiv_1, slice(None, None, None)], None, None)                                     {}
call_function  prev_states             <function _host_tensor at 0x771dfc573740>  ('prev_states',)                                                                                            {}
call_function  prev_states_local       <function load at 0x771d5709ac00>          (prev_states, [tile_begin, tile_begin_2, tile_begin_1, block_size_1, slice(None, None, None)], None, None)  {}
call_function  permute                 aten.permute.default                       (prev_states_local, [1, 0])                                                                                 {}
call_function  acc_o_1                 <function dot at 0x771dfc570cc0>           (C_local, permute, acc_o, None)                                                                             {}
call_function  subscript               <function subscript at 0x771d570d0220>     (scale_m_local, [slice(None, None, None), None])                                                            {}
call_function  acc_o_2                 aten.mul.Tensor                            (acc_o_1, subscript)                                                                                        {}
call_function  tile_id                 <function tile_id at 0x771d570b76a0>       (block_size_0,)                                                                                             {}
call_function  add_1                   <built-in function add>                    (tile_id, 1)                                                                                                {}
call_function  mul_3                   <built-in function mul>                    (add_1, block_size_0)                                                                                       {}
call_function  _for_loop               <function _for_loop at 0x771dfc573b00>     (0, [0], [mul_3], [dA_cumsum_local_m, acc_o_2])                                                             {}
call_function  getitem                 <built-in function getitem>                (_for_loop, 0)                                                                                              {}
call_function  _phi                    <function _phi at 0x771dfc5f40e0>          (acc_o_2, getitem)                                                                                          {}
call_function  D                       <function _host_tensor at 0x771dfc573740>  ('D',)                                                                                                      {}
call_function  load_3                  <function load at 0x771d5709ac00>          (D, [tile_begin_1], None, None)                                                                             {}
call_function  D_local                 prims.convert_element_type.default         (load_3, torch.float32)                                                                                     {}
call_function  tile_index_1            <function tile_index at 0x771d570b63e0>    (block_size_0,)                                                                                             {}
call_function  add_2                   aten.add.Tensor                            (tile_index_1, mul_1)                                                                                       {}
call_function  x                       <function _host_tensor at 0x771dfc573740>  ('x',)                                                                                                      {}
call_function  load_4                  <function load at 0x771d5709ac00>          (x, [tile_begin, add_2, tile_begin_1, block_size_1], None, None)                                            {}
call_function  x_residual              prims.convert_element_type.default         (load_4, torch.float32)                                                                                     {}
call_function  mul_5                   aten.mul.Tensor                            (x_residual, D_local)                                                                                       {}
call_function  acc_o_3                 aten.add.Tensor                            (_phi, mul_5)                                                                                               {}
call_function  convert_element_type_3  prims.convert_element_type.default         (acc_o_3, torch.float16)                                                                                    {}
call_function  tile_index_2            <function tile_index at 0x771d570b63e0>    (block_size_0,)                                                                                             {}
call_function  add_4                   aten.add.Tensor                            (tile_index_2, mul_1)                                                                                       {}
call_function  out                     <function _host_tensor at 0x771dfc573740>  ('out',)                                                                                                    {}
call_function  store                   <function store at 0x771d5709a7a0>         (out, [tile_begin, add_4, tile_begin_1, block_size_1], convert_element_type_3, None)                        {}
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
Node C : FakeTensor(..., size=(1, s16, 1, s22), dtype=torch.float16)
Node C_local : FakeTensor(..., size=(u1, u16), dtype=torch.float16)
Node prev_states : FakeTensor(..., size=(1, s28, s61, s46, s22), dtype=torch.float16)
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
  Block 6: Size=s22, Var=u16, Reduction=True, Source=ReductionLoopBlockSizeSource(reduction_loop=0)
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
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> (0, d1)>
#map4 = affine_map<() -> ()>
#map5 = affine_map<(d0, d1) -> ()>
module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.44269504 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 64 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 8192 : index} : () -> index
    %3 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_h, upper_bound = 16 : index} : () -> index
    %4 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_c, upper_bound = 8 : index} : () -> index
    %5 = arith.ceildivui %c16, %3 : index
    %6 = arith.ceildivui %c256, %0 : index
    %7 = arith.ceildivui %c64, %1 : index
    %8 = arith.ceildivui %c8, %4 : index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (symbol(%5), symbol(%6), symbol(%7), 1, symbol(%8)) {
      %9 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %10 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %11 = arith.muli %arg8, %3 : index
      %12 = arith.muli %arg12, %4 : index
      %13 = arith.muli %arg9, %0 : index
      %subview = memref.subview %arg1[%arg11, %11, %12, %13] [1, 1, 1, %0] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      %14 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
      %15 = tensor.empty(%0) : tensor<?xf32>
      %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%14 : tensor<?xf16>) outs(%15 : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %41 = arith.extf %in : f16 to f32
        linalg.yield %41 : f32
      } -> tensor<?xf32>
      %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%16 : tensor<?xf32>) outs(%15 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %41 = arith.truncf %cst_0 : f64 to f32
        %42 = arith.mulf %in, %41 : f32
        linalg.yield %42 : f32
      } -> tensor<?xf32>
      %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%17 : tensor<?xf32>) outs(%15 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %41 = math.powf %cst, %in : f32
        linalg.yield %41 : f32
      } -> tensor<?xf32>
      %19 = arith.muli %12, %c256 : index
      %20 = arith.divui %11, %c16 : index
      %subview_2 = memref.subview %arg4[%arg11, %19, %20, 0] [1, %0, 1, 16] [1, 1, 1, 1] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %21 = bufferization.to_tensor %subview_2 : memref<?x16xf16, strided<[16, 1], offset: ?>> to tensor<?x16xf16>
      %22 = arith.muli %arg10, %1 : index
      %subview_3 = memref.subview %arg5[%arg11, %12, %11, %22, 0] [1, 1, 1, %1, 16] [1, 1, 1, 1, 1] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %23 = bufferization.to_tensor %subview_3 : memref<?x16xf16, strided<[16, 1], offset: ?>> to tensor<?x16xf16>
      %24 = tensor.empty(%1) : tensor<16x?xf16>
      %transposed = linalg.transpose ins(%23 : tensor<?x16xf16>) outs(%24 : tensor<16x?xf16>) permutation = [1, 0] 
      %25 = linalg.matmul ins(%21, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %extracted_slice = tensor.extract_slice %18[0] [%0] [1] : tensor<?xf32> to tensor<?xf32>
      %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [%0, 1] : tensor<?xf32> into tensor<?x1xf32>
      %26 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%25, %expanded : tensor<?x?xf32>, tensor<?x1xf32>) outs(%9 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %41 = arith.mulf %in, %in_7 : f32
        linalg.yield %41 : f32
      } -> tensor<?x?xf32>
      %27 = arith.addi %arg9, %c1 : index
      %28 = arith.muli %27, %0 : index
      %29 = arith.ceildivui %28, %2 : index
      %30 = scf.for %arg13 = %c0 to %29 step %c1 iter_args(%arg14 = %26) -> (tensor<?x?xf32>) {
        %41 = arith.muli %arg13, %2 : index
        %subview_7 = memref.subview %arg0[%arg11, %12, %20, %13, %41] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %42 = bufferization.to_tensor %subview_7 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %subview_8 = memref.subview %arg1[%arg11, %11, %12, %41] [1, 1, 1, %2] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %43 = bufferization.to_tensor %subview_8 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %44 = tensor.empty(%2) : tensor<?xf32>
        %45 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%43 : tensor<?xf16>) outs(%44 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %61 = arith.extf %in : f16 to f32
          linalg.yield %61 : f32
        } -> tensor<?xf32>
        %extracted_slice_9 = tensor.extract_slice %16[0] [%0] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_10 = tensor.expand_shape %extracted_slice_9 [[0, 1]] output_shape [%0, 1] : tensor<?xf32> into tensor<?x1xf32>
        %46 = tensor.empty(%0) : tensor<?x1xf32>
        %47 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_10 : tensor<?x1xf32>) outs(%46 : tensor<?x1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %61 = arith.truncf %cst_0 : f64 to f32
          %62 = arith.mulf %in, %61 : f32
          linalg.yield %62 : f32
        } -> tensor<?x1xf32>
        %extracted_slice_11 = tensor.extract_slice %45[0] [%2] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_12 = tensor.expand_shape %extracted_slice_11 [[0, 1]] output_shape [1, %2] : tensor<?xf32> into tensor<1x?xf32>
        %48 = tensor.empty(%2) : tensor<1x?xf32>
        %49 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_12 : tensor<1x?xf32>) outs(%48 : tensor<1x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %61 = arith.truncf %cst_0 : f64 to f32
          %62 = arith.mulf %in, %61 : f32
          linalg.yield %62 : f32
        } -> tensor<1x?xf32>
        %50 = tensor.empty(%0, %2) : tensor<?x?xf32>
        %51 = linalg.generic {indexing_maps = [#map2, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%47, %49 : tensor<?x1xf32>, tensor<1x?xf32>) outs(%50 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_17: f32, %out: f32):
          %61 = arith.subf %in, %in_17 : f32
          linalg.yield %61 : f32
        } -> tensor<?x?xf32>
        %52 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%51 : tensor<?x?xf32>) outs(%50 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %61 = math.powf %cst, %in : f32
          linalg.yield %61 : f32
        } -> tensor<?x?xf32>
        %53 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%42, %52 : tensor<?x?xf16>, tensor<?x?xf32>) outs(%50 : tensor<?x?xf32>) {
        ^bb0(%in: f16, %in_17: f32, %out: f32):
          %61 = arith.extf %in : f16 to f32
          %62 = arith.mulf %61, %in_17 : f32
          linalg.yield %62 : f32
        } -> tensor<?x?xf32>
        %subview_13 = memref.subview %arg2[%arg11, %11, %12, %41] [1, 1, 1, %2] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %54 = bufferization.to_tensor %subview_13 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %55 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%54 : tensor<?xf16>) outs(%44 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %61 = arith.extf %in : f16 to f32
          linalg.yield %61 : f32
        } -> tensor<?xf32>
        %extracted_slice_14 = tensor.extract_slice %55[0] [%2] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_15 = tensor.expand_shape %extracted_slice_14 [[0, 1]] output_shape [1, %2] : tensor<?xf32> into tensor<1x?xf32>
        %56 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%53, %expanded_15 : tensor<?x?xf32>, tensor<1x?xf32>) outs(%50 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_17: f32, %out: f32):
          %61 = arith.mulf %in, %in_17 : f32
          linalg.yield %61 : f32
        } -> tensor<?x?xf32>
        %57 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %58 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%56 : tensor<?x?xf32>) outs(%57 : tensor<?x?xf16>) {
        ^bb0(%in: f32, %out: f16):
          %61 = arith.truncf %in : f32 to f16
          linalg.yield %61 : f16
        } -> tensor<?x?xf16>
        %subview_16 = memref.subview %arg3[%arg11, %19, %11, %22] [1, %2, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %59 = bufferization.to_tensor %subview_16 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
        %60 = linalg.matmul ins(%58, %59 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        scf.yield %60 : tensor<?x?xf32>
      }
      %subview_4 = memref.subview %arg6[%11] [1] [1] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
      %31 = bufferization.to_tensor %subview_4 : memref<f16, strided<[], offset: ?>> to tensor<f16>
      %32 = tensor.empty() : tensor<f32>
      %33 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = []} ins(%31 : tensor<f16>) outs(%32 : tensor<f32>) {
      ^bb0(%in: f16, %out: f32):
        %41 = arith.extf %in : f16 to f32
        linalg.yield %41 : f32
      } -> tensor<f32>
      %subview_5 = memref.subview %arg3[%arg11, %19, %11, %22] [1, %0, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %34 = bufferization.to_tensor %subview_5 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
      %35 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%34 : tensor<?x?xf16>) outs(%9 : tensor<?x?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %41 = arith.extf %in : f16 to f32
        linalg.yield %41 : f32
      } -> tensor<?x?xf32>
      %36 = linalg.generic {indexing_maps = [#map1, #map5, #map1], iterator_types = ["parallel", "parallel"]} ins(%35, %33 : tensor<?x?xf32>, tensor<f32>) outs(%9 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %41 = arith.mulf %in, %in_7 : f32
        linalg.yield %41 : f32
      } -> tensor<?x?xf32>
      %37 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%30, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %41 = arith.addf %in, %in_7 : f32
        linalg.yield %41 : f32
      } -> tensor<?x?xf32>
      %38 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %39 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%37 : tensor<?x?xf32>) outs(%38 : tensor<?x?xf16>) {
      ^bb0(%in: f32, %out: f16):
        %41 = arith.truncf %in : f32 to f16
        linalg.yield %41 : f16
      } -> tensor<?x?xf16>
      %subview_6 = memref.subview %arg7[%arg11, %19, %11, %22] [1, %0, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %40 = bufferization.to_buffer %39 : tensor<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      memref.copy %40, %subview_6 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
    }
    return
  }
}


mlir-opt validation succeeded.

