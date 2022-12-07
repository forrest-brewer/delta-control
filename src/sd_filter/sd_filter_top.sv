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

module sd_filter_top
( input   clk
, input   reset
, input   clk_enable
, input   signed [15:0] input_rsvd  // sfix16_En7
, output  ce_out
, output  output_rsvd
);
  // -------------------------------------------------------------
  wire SD_2nd_Order_Modulator1_out1;
  wire SD_2nd_Order_Modulator_out1;
  wire signed [46:0] SD_Filter_out1;  // sfix47_En22

  // -------------------------------------------------------------
  sd_modulator #(16, 7)
    sd_modulator_in
    ( .clk(clk)
    , .reset(reset)
    , .enb(clk_enable)
    , .in(input_rsvd)
    , .out(SD_2nd_Order_Modulator1_out1)
    );

  // -------------------------------------------------------------
  sd_filter u_SD_Filter (.clk(clk),
                         .reset(reset),
                         .enb(clk_enable),
                         .input_rsvd(SD_2nd_Order_Modulator1_out1),
                         .feedback(SD_2nd_Order_Modulator_out1),
                         .output_rsvd(SD_Filter_out1)  // sfix47_En22
                         );

  // -------------------------------------------------------------
  sd_modulator #(47, 22)
    sd_modulator_out
    ( .clk(clk)
    , .reset(reset)
    , .enb(clk_enable)
    , .in(SD_Filter_out1)
    , .out(SD_2nd_Order_Modulator_out1)
    );

  // -------------------------------------------------------------
  assign output_rsvd = SD_2nd_Order_Modulator_out1;
  assign ce_out = clk_enable;

// -------------------------------------------------------------
endmodule
