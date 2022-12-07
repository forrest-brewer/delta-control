// --------------------------------------------------------------------
// Copyright 2022 qaztronic
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
//
// Licensed under the Solderpad Hardware License v 2.1 (the "License");
// you may not use this file except in compliance with the License, or,
// at your option, the Apache License version 2.0. You may obtain a copy
// of the License at
//
// https://solderpad.org/licenses/SHL-2.1/
//
// Unless required by applicable law or agreed to in writing, any work
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// --------------------------------------------------------------------

import sd_filter_pkg::*;

module sd_filter
( input   clk
, input   reset
, input   enb
, input   input_rsvd
, input   feedback
, output  signed [46:0] output_rsvd  // sfix47_En22
);
  // -------------------------------------------------------------
  wire signed [24:0] node_5_out1;  // sfix25
  wire signed [32:0] node_4_out1;  // sfix33_En8
  wire signed [32:0] node_3_out1;  // sfix33_En8
  wire signed [46:0] Sum8_stage2_add_cast;  // sfix47_En22
  wire signed [47:0] Sum8_op_stage1;  // sfix48_En22
  wire signed [32:0] node_2_out1;  // sfix33_En8
  wire signed [65:0] Gain3_out1;  // sfix66_En63
  wire signed [46:0] Sum8_stage3_add_cast;  // sfix47_En22
  wire signed [46:0] Sum8_stage3_add_cast_1;  // sfix47_En22

  // --------------------------------------------------------------------
  localparam sd_filter_cfg_t SDF_NODE_5_CFG =
  '{ IN_W : 16
   , IN_N : 0
   , S    : 1
   , C_W  : 16
   , C_N  : 0
   , OUT_W: 25
   , OUT_N: 0
   };

  wire signed [SDF_NODE_5_CFG.IN_W-1:0] node_in_5 = 'sb0;

  sd_filter_node #(11199.0, 11.0, SDF_NODE_5_CFG)
    u_node_5
    ( .clk(clk)
    , .reset(reset)
    , .enb(enb)
    , .input_rsvd(input_rsvd)
    , .node_in(node_in_5)  // sfix16
    , .feedback(feedback)
    , .node_out(node_5_out1)  // sfix25
    );

  // --------------------------------------------------------------------
  localparam sd_filter_cfg_t SDF_NODE_4_CFG =
  '{ IN_W : 25
   , IN_N : 0
   , S    : 11
   , C_W  : 21
   , C_N  : 8
   , OUT_W: 33
   , OUT_N: 8
   };

  sd_filter_node #(1874.234375, 34.68359375, SDF_NODE_4_CFG)
    u_node_4
    ( .clk(clk)
    , .reset(reset)
    , .enb(enb)
    , .input_rsvd(input_rsvd)
    , .node_in(node_5_out1)  // sfix25
    , .feedback(feedback)
    , .node_out(node_4_out1)  // sfix33_En8
    );

  // --------------------------------------------------------------------
  localparam sd_filter_cfg_t SDF_NODE_3_CFG =
  '{ IN_W : 33
   , IN_N : 8
   , S    : 11
   , C_W  : 25
   , C_N  : 8
   , OUT_W: 33
   , OUT_N: 8
   };

  sd_filter_node #(19312.5078125, 173.80078125, SDF_NODE_3_CFG)
    u_node_3
    ( .clk(clk)
    , .reset(reset)
    , .enb(enb)
    , .input_rsvd(input_rsvd)
    , .node_in(node_4_out1)  // sfix33_En8
    , .feedback(feedback)
    , .node_out(node_3_out1)  // sfix33_En8
    );

  // --------------------------------------------------------------------
  wire signed [23:0] beta;  // sfix24_En22

  scaler_mux #(24, 22, 0.0009999275207519531)
    beta_mux_i(input_rsvd, beta);

  assign Sum8_stage2_add_cast = {{23{beta[23]}}, beta};
  assign Sum8_op_stage1 = {Sum8_stage2_add_cast[46], Sum8_stage2_add_cast};

  // --------------------------------------------------------------------
  localparam sd_filter_cfg_t SDF_NODE_2_CFG =
  '{ IN_W : 33
   , IN_N : 8
   , S    : 11
   , C_W  : 21
   , C_N  : 8
   , OUT_W: 33
   , OUT_N: 8
   };

  sd_filter_node #(1603.01953125, 0.046875, SDF_NODE_2_CFG)
    u_node_2_i
    ( .clk(clk)
    , .reset(reset)
    , .enb(enb)
    , .input_rsvd(input_rsvd)
    , .node_in(node_3_out1)  // sfix33_En8
    , .feedback(feedback)
    , .node_out(node_2_out1)  // sfix33_En8
    );

  // --------------------------------------------------------------------
  assign Gain3_out1 = {{2{node_2_out1[32]}}, {node_2_out1, 31'b0000000000000000000000000000000}};
  assign Sum8_stage3_add_cast = Sum8_op_stage1[46:0];
  assign Sum8_stage3_add_cast_1 = {{22{Gain3_out1[65]}}, Gain3_out1[65:41]};
  assign output_rsvd = Sum8_stage3_add_cast + Sum8_stage3_add_cast_1;

// --------------------------------------------------------------------
endmodule
