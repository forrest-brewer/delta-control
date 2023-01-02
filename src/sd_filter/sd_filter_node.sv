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

module sd_filter_node #(real ALPHA, real BETA, sd_filter_cfg_t CFG)
( input   clk
, input   reset
, input   enb
, input   input_rsvd
, input   signed [CFG.IN_W-1:0] node_in
, input   feedback
, output  signed [CFG.OUT_W-1:0] node_out
);

  // -------------------------------------------------------------
  localparam IN_W  = CFG.IN_W         ;  // previous node width
  localparam IN_N  = CFG.IN_N         ;  // previous node fractional width
  localparam OUT_W = CFG.OUT_W        ;  // accumulator width
  localparam OUT_N = CFG.OUT_N        ;  // accumulator fractional width
  localparam S     = CFG.S            ;  // right shift
  localparam C_W   = CFG.C_W          ;  // coefficient width
  localparam C_N   = CFG.C_N          ;  // coefficient fractional width
  localparam IN_M  = IN_W - IN_N - 1  ;
  localparam C_M   = C_W - C_N - 1    ;
  localparam OUT_M = OUT_W - OUT_N - 1;

  // -------------------------------------------------------------
  // Qm.n | w = 1 + m + n
  localparam K_M  = IN_W - IN_N - S - 1;
  localparam K_N  = IN_N + S;
  localparam K_W  = IN_W;
  localparam M  = C_M > K_M ? C_M : K_M;
  localparam N  = C_N;
  localparam W  = 1 + M + N;

// --------------------------------------------------------------------
// synthesis translate_off
  initial
  begin
    a_k_n_c_n: assert(K_N > C_N) else $fatal;
  end
// synthesis translate_on
// --------------------------------------------------------------------

  initial
  begin
    $display("%m | IN   | Q%2d.%2d | sfix%0d_En%0d", IN_M, IN_N, IN_W, IN_N);
    $display("%m | K    | Q%2d.%2d | sfix%0d_En%0d", K_M, K_N, K_W, K_N);
    $display("%m | C    | Q%2d.%2d | sfix%0d_En%0d", C_M,  C_N, C_W, C_N);
    $display("%m | node | Q%0d.%0d | sfix%0d_En%0d", M, N, W, N);
    $display("%m | OUT  | Q%2d.%2d | sfix%0d_En%0d", OUT_M,  OUT_N, OUT_W, OUT_N);
  end

  // -------------------------------------------------------------
  wire signed [W-1:0] k_ts_node;

  generate
    if(K_M > C_M)
      assign k_ts_node = node_in[IN_W - 1:K_N - C_N];
    else
      assign k_ts_node = { {(C_M - K_M){node_in[IN_W-1]}}, node_in[IN_W - 1:K_N - C_N]};
  endgenerate

  // -------------------------------------------------------------
  wire signed [C_W-1:0] beta;

  scaler_mux #(C_W, C_N, BETA)
    beta_mux_i(input_rsvd, beta);

  // -------------------------------------------------------------
  wire signed [C_W-1:0] alpha;

  scaler_mux #(C_W, C_N, ALPHA)
    alpha_mux_i(feedback, alpha);

  // -------------------------------------------------------------
  wire signed [W:0] k_ts_sum = k_ts_node + beta - alpha;  // grow by 1 bit for addition
  reg  signed [OUT_W-1:0] z;
  wire signed [OUT_W-1:0] acc_sum = z + {{(OUT_W - W - 1){k_ts_sum[W]}}, k_ts_sum};

  always @(posedge clk or posedge reset)
    if (reset == 1'b1)
      z <= 'sh0;
    else if (enb)
      z <= acc_sum;

  assign node_out = z;

// -------------------------------------------------------------
endmodule
