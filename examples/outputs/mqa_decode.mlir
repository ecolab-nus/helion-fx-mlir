=== Device IR ===
Graph 0: ForLoopGraphInfo
opcode         name             target                                     args                                                                         kwargs
-------------  ---------------  -----------------------------------------  ---------------------------------------------------------------------------  --------
placeholder    arg0_1           arg0_1                                     ()                                                                           {}
placeholder    arg1_1           arg1_1                                     ()                                                                           {}
placeholder    arg2_1           arg2_1                                     ()                                                                           {}
placeholder    arg3_1           arg3_1                                     ()                                                                           {}
placeholder    arg4_1           arg4_1                                     ()                                                                           {}
call_function  _new_var         <function _new_var at 0x7f77f4f81360>      (arg0_1,)                                                                    {}
call_function  _new_var_1       <function _new_var at 0x7f77f4f81360>      (arg1_1,)                                                                    {}
call_function  _new_var_2       <function _new_var at 0x7f77f4f81360>      (arg2_1,)                                                                    {}
call_function  _new_var_3       <function _new_var at 0x7f77f4f81360>      (arg3_1,)                                                                    {}
call_function  _new_var_4       <function _new_var at 0x7f77f4f81360>      (arg4_1,)                                                                    {}
call_function  k_view           <function _host_tensor at 0x7f77f4f47a30>  ('k_view',)                                                                  {}
call_function  sym_size_int     aten.sym_size.int                          (arg0_1, 0)                                                                  {}
call_function  block_size_4     <function _get_symnode at 0x7f77f4f46dd0>  ('block_size_4',)                                                            {}
call_function  k                <function load at 0x7f77dc4735b0>          (k_view, [sym_size_int, slice(None, None, None), block_size_4], None, None)  {}
call_function  qk               aten.bmm.default                           (_new_var, k)                                                                {}
call_function  _mask_to_2       <function _mask_to at 0x7f77f4f81000>      (qk, -inf)                                                                   {}
call_function  m_qk             aten.amax.default                          (_mask_to_2, [-1], True)                                                     {}
call_function  mul              aten.mul.Tensor                            (m_qk, _new_var_2)                                                           {}
call_function  m_ij             aten.maximum.default                       (_new_var_1, mul)                                                            {}
call_function  m_ij_broadcast   <function broadcast at 0x7f77dc2115a0>     (m_ij, 2, [sym_size_int, 32, block_size_4])                                  {}
call_function  mul_1            aten.mul.Tensor                            (qk, _new_var_2)                                                             {}
call_function  qk_1             aten.sub.Tensor                            (mul_1, m_ij_broadcast)                                                      {}
call_function  exp              aten.exp.default                           (qk_1,)                                                                      {}
call_function  _mask_to_3       <function _mask_to at 0x7f77f4f81000>      (exp, 0)                                                                     {}
call_function  l_ij             aten.sum.dim_IntList                       (_mask_to_3, [-1], True)                                                     {}
call_function  sub_1            aten.sub.Tensor                            (_new_var_1, m_ij)                                                           {}
call_function  alpha            aten.exp.default                           (sub_1,)                                                                     {}
call_function  mul_2            aten.mul.Tensor                            (_new_var_3, alpha)                                                          {}
call_function  l_i              aten.add.Tensor                            (mul_2, l_ij)                                                                {}
call_function  alpha_broadcast  <function broadcast at 0x7f77dc2115a0>     (alpha, 2, [sym_size_int, 32, 128])                                          {}
call_function  acc              aten.mul.Tensor                            (_new_var_4, alpha_broadcast)                                                {}
call_function  v_view           <function _host_tensor at 0x7f77f4f47a30>  ('v_view',)                                                                  {}
call_function  v                <function load at 0x7f77dc4735b0>          (v_view, [sym_size_int, block_size_4, slice(None, None, None)], None, None)  {}
call_function  acc_1            aten.baddbmm.default                       (acc, _mask_to_3, v)                                                         {}
call_function  m_i              <function _new_var at 0x7f77f4f81360>      (m_ij,)                                                                      {}
output         output           output                                     ([m_i, l_i, acc_1],)                                                         {}
Graph 1: IfGraphInfo
opcode         name            target                                     args                                                                                           kwargs
-------------  --------------  -----------------------------------------  ---------------------------------------------------------------------------------------------  --------
placeholder    arg0_1          arg0_1                                     ()                                                                                             {}
placeholder    arg1_1          arg1_1                                     ()                                                                                             {}
call_function  _new_var        <function _new_var at 0x7f77f4f81360>      (arg0_1,)                                                                                      {}
call_function  _new_var_1      <function _new_var at 0x7f77f4f81360>      (arg1_1,)                                                                                      {}
call_function  _mask_to        <function _mask_to at 0x7f77f4f81000>      (_new_var, -inf)                                                                               {}
call_function  max_lse         aten.amax.default                          (_mask_to, [0])                                                                                {}
call_function  sub             aten.sub.Tensor                            (_new_var, max_lse)                                                                            {}
call_function  weights         aten.exp.default                           (sub,)                                                                                         {}
call_function  _mask_to_1      <function _mask_to at 0x7f77f4f81000>      (weights, 0)                                                                                   {}
call_function  lse_sum         aten.sum.dim_IntList                       (_mask_to_1, [0])                                                                              {}
call_function  norm_scale      aten.div.Tensor                            (weights, lse_sum)                                                                             {}
call_function  sym_size_int    aten.sym_size.int                          (arg0_1, 0)                                                                                    {}
call_function  sym_size_int_1  aten.sym_size.int                          (arg0_1, 1)                                                                                    {}
call_function  norm_scale_1    <function broadcast at 0x7f77dc2115a0>     (norm_scale, 3, [sym_size_int, sym_size_int_1, 32, 128])                                       {}
call_function  mul             aten.mul.Tensor                            (_new_var_1, norm_scale_1)                                                                     {}
call_function  _mask_to_2      <function _mask_to at 0x7f77f4f81000>      (mul, 0)                                                                                       {}
call_function  weighted_acc    aten.sum.dim_IntList                       (_mask_to_2, [0])                                                                              {}
call_function  out             <function _host_tensor at 0x7f77f4f47a30>  ('out',)                                                                                       {}
call_function  store           <function store at 0x7f77dc471990>         (out, [sym_size_int_1, slice(None, None, None), slice(None, None, None)], weighted_acc, None)  {}
output         output          output                                     ([],)                                                                                          {}
Graph 2: RootGraphInfo
opcode         name          target                                     args                                                                                    kwargs
-------------  ------------  -----------------------------------------  --------------------------------------------------------------------------------------  --------
call_function  qk_scale      <function full at 0x7f77dc462dd0>          ([], 0.08838834764831843, torch.float16, None)                                          {}
call_function  block_size_0  <function _get_symnode at 0x7f77f4f46dd0>  ('block_size_0',)                                                                       {}
call_function  m_i           <function full at 0x7f77dc462dd0>          ([block_size_0, 32, 1], -inf, torch.float16, None)                                      {}
call_function  l_i           <function full at 0x7f77dc462dd0>          ([block_size_0, 32, 1], 1.0, torch.float16, None)                                       {}
call_function  acc           <function full at 0x7f77dc462dd0>          ([block_size_0, 32, 128], 0.0, torch.float16, None)                                     {}
call_function  q_view        <function _host_tensor at 0x7f77f4f47a30>  ('q_view',)                                                                             {}
call_function  q             <function load at 0x7f77dc4735b0>          (q_view, [block_size_0, slice(None, None, None), slice(None, None, None)], None, None)  {}
call_function  block_size_1  <function _get_symnode at 0x7f77f4f46dd0>  ('block_size_1',)                                                                       {}
call_function  tile_begin    <function tile_begin at 0x7f77dc493130>    (block_size_1,)                                                                         {}
call_function  tile_end      <function tile_end at 0x7f77dc493520>      (block_size_1,)                                                                         {}
call_function  _for_loop     <function _for_loop at 0x7f77f4f47d90>     (0, [tile_begin], [tile_end], [q, m_i, qk_scale, l_i, acc])                             {}
call_function  getitem       <built-in function getitem>                (_for_loop, 0)                                                                          {}
call_function  getitem_1     <built-in function getitem>                (_for_loop, 1)                                                                          {}
call_function  getitem_2     <built-in function getitem>                (_for_loop, 2)                                                                          {}
call_function  _phi          <function _phi at 0x7f77f4f80310>          (m_i, getitem)                                                                          {}
call_function  _phi_1        <function _phi at 0x7f77f4f80310>          (l_i, getitem_1)                                                                        {}
call_function  _phi_2        <function _phi at 0x7f77f4f80310>          (acc, getitem_2)                                                                        {}
call_function  log           aten.log.default                           (_phi_1,)                                                                               {}
call_function  split_lse     aten.add.Tensor                            (log, _phi)                                                                             {}
call_function  gathered_lse  <function gather at 0x7f77dc211870>        (block_size_1, split_lse)                                                               {}
call_function  gathered_acc  <function gather at 0x7f77dc211870>        (block_size_1, _phi_2)                                                                  {}
call_function  tile_id       <function tile_id at 0x7f77dc493a30>       (block_size_1,)                                                                         {}
call_function  eq            <built-in function eq>                     (tile_id, 0)                                                                            {}
call_function  _if           <function _if at 0x7f77f4f80160>           (eq, 1, [gathered_lse, gathered_acc])                                                   {}
output         output        output                                     (None,)                                                                                 {}


=== Nodes with symbols ===
Node arg0_1 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node arg1_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node arg2_1 : FakeTensor(..., size=(), dtype=torch.float16)
Node arg3_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node arg4_1 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node _new_var_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _new_var_2 : FakeTensor(..., size=(), dtype=torch.float16)
Node _new_var_3 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _new_var_4 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node k_view : FakeTensor(..., size=(s30, 128, s4), dtype=torch.float16)
Node sym_size_int : u0
Node block_size_4 : u6
Node k : FakeTensor(..., size=(u0, 128, u6), dtype=torch.float16)
Node qk : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node _mask_to_2 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node m_qk : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node mul : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node m_ij : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node m_ij_broadcast : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node mul_1 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node qk_1 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node exp : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node _mask_to_3 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node l_ij : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node sub_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node alpha : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node mul_2 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node alpha_broadcast : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node v_view : FakeTensor(..., size=(s30, s4, 128), dtype=torch.float16)
Node v : FakeTensor(..., size=(u0, u6, 128), dtype=torch.float16)
Node acc_1 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node m_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node arg0_1 : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node arg1_1 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node _new_var_1 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node _mask_to : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node max_lse : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node sub : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node weights : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node _mask_to_1 : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node lse_sum : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node norm_scale : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node sym_size_int : u1
Node sym_size_int_1 : u0
Node norm_scale_1 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node mul : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node _mask_to_2 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node weighted_acc : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node out : FakeTensor(..., size=(s30, 32, 128), dtype=torch.float16)
Node qk_scale : FakeTensor(..., size=(), dtype=torch.float16)
Node block_size_0 : u0
Node m_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node q_view : FakeTensor(..., size=(s30, 32, 128), dtype=torch.float16)
Node q : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node block_size_1 : u1
Node tile_begin : u4
Node tile_end : u5
Node _for_loop : [FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16), FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16), FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)]
Node getitem : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node getitem_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node getitem_2 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node _phi : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _phi_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _phi_2 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node log : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node split_lse : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node gathered_lse : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node gathered_acc : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node tile_id : u7
Node eq : Eq(u7, 0)
Node _if : []
Node arg0_1 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node arg1_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node arg2_1 : FakeTensor(..., size=(), dtype=torch.float16)
Node arg3_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node arg4_1 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node _new_var_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _new_var_2 : FakeTensor(..., size=(), dtype=torch.float16)
Node _new_var_3 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _new_var_4 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node k_view : FakeTensor(..., size=(s30, 128, s4), dtype=torch.float16)
Node sym_size_int : u0
Node block_size_4 : u6
Node k : FakeTensor(..., size=(u0, 128, u6), dtype=torch.float16)
Node qk : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node _mask_to_2 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node m_qk : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node mul : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node m_ij : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node m_ij_broadcast : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node mul_1 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node qk_1 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node exp : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node _mask_to_3 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node l_ij : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node sub_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node alpha : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node mul_2 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node alpha_broadcast : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node v_view : FakeTensor(..., size=(s30, s4, 128), dtype=torch.float16)
Node v : FakeTensor(..., size=(u0, u6, 128), dtype=torch.float16)
Node acc_1 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node m_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node arg0_1 : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node arg1_1 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node _new_var_1 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node _mask_to : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node max_lse : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node sub : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node weights : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node _mask_to_1 : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node lse_sum : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node norm_scale : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node sym_size_int : u1
Node sym_size_int_1 : u0
Node norm_scale_1 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node mul : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node _mask_to_2 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node weighted_acc : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node out : FakeTensor(..., size=(s30, 32, 128), dtype=torch.float16)
Node qk_scale : FakeTensor(..., size=(), dtype=torch.float16)
Node block_size_0 : u0
Node m_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node q_view : FakeTensor(..., size=(s30, 32, 128), dtype=torch.float16)
Node q : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node block_size_1 : u1
Node tile_begin : u4
Node tile_end : u5
Node _for_loop : [FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16), FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16), FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)]
Node getitem : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node getitem_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node getitem_2 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node _phi : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _phi_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _phi_2 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node log : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node split_lse : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node gathered_lse : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node gathered_acc : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node tile_id : u7
Node eq : Eq(u7, 0)
Node _if : []
Node arg0_1 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node arg1_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node arg2_1 : FakeTensor(..., size=(), dtype=torch.float16)
Node arg3_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node arg4_1 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node _new_var_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _new_var_2 : FakeTensor(..., size=(), dtype=torch.float16)
Node _new_var_3 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _new_var_4 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node k_view : FakeTensor(..., size=(s30, 128, s4), dtype=torch.float16)
Node sym_size_int : u0
Node block_size_4 : u6
Node k : FakeTensor(..., size=(u0, 128, u6), dtype=torch.float16)
Node qk : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node _mask_to_2 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node m_qk : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node mul : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node m_ij : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node m_ij_broadcast : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node mul_1 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node qk_1 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node exp : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node _mask_to_3 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node l_ij : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node sub_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node alpha : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node mul_2 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node alpha_broadcast : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node v_view : FakeTensor(..., size=(s30, s4, 128), dtype=torch.float16)
Node v : FakeTensor(..., size=(u0, u6, 128), dtype=torch.float16)
Node acc_1 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node m_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node arg0_1 : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node arg1_1 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node _new_var_1 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node _mask_to : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node max_lse : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node sub : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node weights : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node _mask_to_1 : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node lse_sum : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node norm_scale : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node sym_size_int : u1
Node sym_size_int_1 : u0
Node norm_scale_1 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node mul : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node _mask_to_2 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node weighted_acc : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node out : FakeTensor(..., size=(s30, 32, 128), dtype=torch.float16)
Node qk_scale : FakeTensor(..., size=(), dtype=torch.float16)
Node block_size_0 : u0
Node m_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node q_view : FakeTensor(..., size=(s30, 32, 128), dtype=torch.float16)
Node q : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node block_size_1 : u1
Node tile_begin : u4
Node tile_end : u5
Node _for_loop : [FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16), FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16), FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)]
Node getitem : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node getitem_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node getitem_2 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node _phi : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _phi_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _phi_2 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node log : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node split_lse : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node gathered_lse : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node gathered_acc : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node tile_id : u7
Node eq : Eq(u7, 0)
Node _if : []
Node arg0_1 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node arg1_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node arg2_1 : FakeTensor(..., size=(), dtype=torch.float16)
Node arg3_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node arg4_1 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node _new_var_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _new_var_2 : FakeTensor(..., size=(), dtype=torch.float16)
Node _new_var_3 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _new_var_4 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node k_view : FakeTensor(..., size=(s30, 128, s4), dtype=torch.float16)
Node sym_size_int : u0
Node block_size_4 : u6
Node k : FakeTensor(..., size=(u0, 128, u6), dtype=torch.float16)
Node qk : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node _mask_to_2 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node m_qk : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node mul : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node m_ij : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node m_ij_broadcast : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node mul_1 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node qk_1 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node exp : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node _mask_to_3 : FakeTensor(..., size=(u0, 32, u6), dtype=torch.float16)
Node l_ij : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node sub_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node alpha : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node mul_2 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node alpha_broadcast : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node v_view : FakeTensor(..., size=(s30, s4, 128), dtype=torch.float16)
Node v : FakeTensor(..., size=(u0, u6, 128), dtype=torch.float16)
Node acc_1 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node m_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node arg0_1 : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node arg1_1 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node _new_var_1 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node _mask_to : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node max_lse : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node sub : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node weights : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node _mask_to_1 : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node lse_sum : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node norm_scale : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node sym_size_int : u1
Node sym_size_int_1 : u0
Node norm_scale_1 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node mul : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node _mask_to_2 : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node weighted_acc : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node out : FakeTensor(..., size=(s30, 32, 128), dtype=torch.float16)
Node qk_scale : FakeTensor(..., size=(), dtype=torch.float16)
Node block_size_0 : u0
Node m_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node q_view : FakeTensor(..., size=(s30, 32, 128), dtype=torch.float16)
Node q : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node block_size_1 : u1
Node tile_begin : u4
Node tile_end : u5
Node _for_loop : [FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16), FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16), FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)]
Node getitem : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node getitem_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node getitem_2 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node _phi : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _phi_1 : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node _phi_2 : FakeTensor(..., size=(u0, 32, 128), dtype=torch.float16)
Node log : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node split_lse : FakeTensor(..., size=(u0, 32, 1), dtype=torch.float16)
Node gathered_lse : FakeTensor(..., size=(u1, u0, 32, 1), dtype=torch.float16)
Node gathered_acc : FakeTensor(..., size=(u1, u0, 32, 128), dtype=torch.float16)
Node tile_id : u7
Node eq : Eq(u7, 0)
Node _if : []


=== Compile Environment ===
Block Sizes (5):
  Block 0: Size=s30, Var=u0, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 1: Size=s4, Var=u1, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 2: Size=32, Var=32, Reduction=True, Source=ReductionLoopBlockSizeSource(reduction_loop=0)
  Block 3: Size=128, Var=128, Reduction=True, Source=ReductionLoopBlockSizeSource(reduction_loop=1)
  Block 4: Size=-u4 + u5, Var=u6, Reduction=False, Source=LoopSpecBlockSizeSource()
Shape Env (17):
  Var s30: 16
  Var s22: 32
  Var s19: 128
  Var s35: 16
  Var s4: 8192
  Var s23: 128
  Var s80: 16
  Var s66: 8192
  Var s41: 128
  Var u0: 64
  Var u1: 64
  Var u2: 32
  Var u3: 128
  Var u4: 8192
  Var u5: 8192
  Var u6: 64
  Var u7: 8192


=== MLIR Dump ===
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  func.func @flash_decode(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %cst_1 = arith.constant 0xFC00 : f16
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 8.837890e-02 : f16
    %c8192 = arith.constant 8192 : index
    %c16 = arith.constant 16 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 16 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_s, upper_bound = 8192 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 64 : index} : () -> index
    %3 = arith.ceildivui %c16, %0 : index
    %4 = arith.ceildivui %c8192, %1 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0) : tensor<?x32x1xf16>
      %6 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %7 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %8 = tensor.empty(%0) : tensor<?x32x128xf16>
      %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %10 = arith.muli %arg4, %0 : index
      %subview = memref.subview %arg3[%10, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      %11 = bufferization.to_tensor %subview : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to tensor<?x32x128xf16>
      %12 = arith.muli %arg5, %1 : index
      %13 = arith.addi %12, %1 : index
      %14 = arith.subi %13, %12 : index
      %15 = arith.ceildivui %14, %2 : index
      %16:3 = scf.for %arg6 = %c0 to %15 step %c1 iter_args(%arg7 = %6, %arg8 = %7, %arg9 = %9) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
        %24 = arith.muli %arg6, %2 : index
        %25 = arith.addi %12, %24 : index
        %subview_3 = memref.subview %arg0[%10, 0, %25] [%0, 128, %2] [1, 1, 1] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        %26 = bufferization.to_tensor %subview_3 : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to tensor<?x128x?xf16>
        %27 = arith.index_cast %0 : index to i64
        %28 = arith.cmpi eq, %27, %27 : i64
        cf.assert %28, "mismatching contracting dimension"
        %29 = tensor.empty(%0, %2) : tensor<?x32x?xf16>
        %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %31 = linalg.batch_matmul ins(%11, %26 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%30 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %32 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%31 : tensor<?x32x?xf16>) outs(%6 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = arith.maximumf %in, %out : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %33 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32 : tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = arith.mulf %in, %cst_2 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %34 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %33 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.cmpf ogt, %in, %in_5 : f16
          %54 = arith.select %53, %in, %in_5 : f16
          linalg.yield %54 : f16
        } -> tensor<?x32x1xf16>
        %35 = tensor.empty(%0) : tensor<?x32x32xf16>
        %36 = "loom.broadcast"(%34, %35) {dim = 2 : i64} : (tensor<?x32x1xf16>, tensor<?x32x32xf16>) -> tensor<?x32x?xf16>
        %37 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31 : tensor<?x32x?xf16>) outs(%29 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = arith.mulf %in, %cst_2 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x?xf16>
        %38 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %36 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%29 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.subf %in, %in_5 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x?xf16>
        %39 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38 : tensor<?x32x?xf16>) outs(%29 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = math.exp %in : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x?xf16>
        %40 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %41 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%39 : tensor<?x32x?xf16>) outs(%40 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = arith.addf %in, %out : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %42 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %34 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.subf %in, %in_5 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %43 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42 : tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = math.exp %in : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %44 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %43 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.mulf %in, %in_5 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %45 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %41 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.addf %in, %in_5 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %46 = "loom.broadcast"(%43, %35) {dim = 2 : i64} : (tensor<?x32x1xf16>, tensor<?x32x32xf16>) -> tensor<?x32x128xf16>
        %47 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %46 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%8 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.mulf %in, %in_5 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x128xf16>
        %subview_4 = memref.subview %arg1[%10, %25, 0] [%0, %2, 128] [1, 1, 1] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        %48 = bufferization.to_tensor %subview_4 : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        cf.assert %28, "mismatching contracting dimension"
        %49 = arith.index_cast %2 : index to i64
        %50 = arith.cmpi eq, %49, %49 : i64
        cf.assert %50, "mismatching contracting dimension"
        %51 = linalg.batch_matmul ins(%39, %48 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%9 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %52 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%51, %47 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%8 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.addf %in, %in_5 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x128xf16>
        scf.yield %34, %45, %52 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
      }
      %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16#1 : tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %out: f16):
        %24 = math.log %in : f16
        linalg.yield %24 : f16
      } -> tensor<?x32x1xf16>
      %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %16#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %24 = arith.addf %in, %in_3 : f16
        linalg.yield %24 : f16
      } -> tensor<?x32x1xf16>
      %19 = tensor.empty(%4, %0) : tensor<?x?x32x1xf16>
      %20 = "loom.gather"(%18, %19, %arg5) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : (tensor<?x32x1xf16>, tensor<?x?x32x1xf16>, index) -> tensor<?x?x32x1xf16>
      %21 = tensor.empty(%4, %0) : tensor<?x?x32x128xf16>
      %22 = "loom.gather"(%16#2, %21, %arg5) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : (tensor<?x32x128xf16>, tensor<?x?x32x128xf16>, index) -> tensor<?x?x32x128xf16>
      %23 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %23 {
        %24 = tensor.empty(%0) : tensor<?x32x1xi64>
        %25 = linalg.fill ins(%c0_i64 : i64) outs(%24 : tensor<?x32x1xi64>) -> tensor<?x32x1xi64>
        %26:2 = linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%20 : tensor<?x?x32x1xf16>) outs(%6, %25 : tensor<?x32x1xf16>, tensor<?x32x1xi64>) {
        ^bb0(%in: f16, %out: f16, %out_4: i64):
          %38 = linalg.index 0 : index
          %39 = arith.index_cast %38 : index to i64
          %40 = arith.maximumf %in, %out : f16
          %41 = arith.cmpf ogt, %in, %out : f16
          %42 = arith.select %41, %39, %out_4 : i64
          linalg.yield %40, %42 : f16, i64
        } -> (tensor<?x32x1xf16>, tensor<?x32x1xi64>)
        %27 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %26#0 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%19 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %38 = arith.subf %in, %in_4 : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x32x1xf16>
        %28 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27 : tensor<?x?x32x1xf16>) outs(%19 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %38 = math.exp %in : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x32x1xf16>
        %29 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %30 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%28 : tensor<?x?x32x1xf16>) outs(%29 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %38 = arith.addf %in, %out : f16
          linalg.yield %38 : f16
        } -> tensor<?x32x1xf16>
        %31 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %30 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%19 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %38 = arith.divf %in, %in_4 : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x32x1xf16>
        %32 = tensor.empty(%1, %0) : tensor<?x?x32x32xf16>
        %33 = "loom.broadcast"(%31, %32) {dim = 3 : i64} : (tensor<?x?x32x1xf16>, tensor<?x?x32x32xf16>) -> tensor<?x?x32x128xf16>
        %34 = arith.cmpi eq, %4, %1 : index
        cf.assert %34, "mismatched size for broadcast"
        %35 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %33 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%21 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %38 = arith.mulf %in, %in_4 : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x32x128xf16>
        %36 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%35 : tensor<?x?x32x128xf16>) outs(%9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %38 = arith.addf %in, %out : f16
          linalg.yield %38 : f16
        } -> tensor<?x32x128xf16>
        %subview_3 = memref.subview %arg2[%10, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %37 = bufferization.to_buffer %36 : tensor<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        memref.copy %37, %subview_3 : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      }
    }
    return
  }
}


mlir-opt validation succeeded.

