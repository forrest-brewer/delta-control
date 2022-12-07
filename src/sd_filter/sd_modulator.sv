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

module sd_modulator  #(W, Q)
( input                 clk
, input                 reset
, input                 enb
, input  signed [W-1:0] in
, output                out
);
  // -------------------------------------------------------------
  reg signed  [W-1:0] z[2];
  wire signed [W-1:0] sum[2];

  generate
    for (genvar i = 0; i < 2; i++)
      always_ff @(posedge clk or posedge reset)
      if(reset)
        z[i] <= 'sb0;
      else if(enb)
        z[i] <= sum[i];
  endgenerate

  // -------------------------------------------------------------
  wire signed [W-1:0] quantizer_out;

  scaler_mux #(W, Q, 1.0)
    quantizer_i (out, quantizer_out);

  // -------------------------------------------------------------
  assign sum[0] = in     - quantizer_out + z[0];
  assign sum[1] = sum[0] - quantizer_out + z[1];
  assign out    = z[1] > 'sb0 ? 1'b1 : 1'b0;

// -------------------------------------------------------------
endmodule
