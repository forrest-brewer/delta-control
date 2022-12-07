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

module scaler_mux #(int W, int Q, real V)
( input                  in
, output  signed [W-1:0] out
);
  // -------------------------------------------------------------
  localparam logic signed [W-1:0] V_NEG = ~($rtoi(V * 2**Q) - 1);
  localparam logic signed [W-1:0] V_POS =   $rtoi(V * 2**Q);

  initial
    $display("%m | %b | %b |", V_NEG , V_POS);

  // -------------------------------------------------------------
  assign out = in > 1'b0 ? V_POS : V_NEG;

endmodule
