// --------------------------------------------------------------------
// Copyright 2020 qaztronic
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
//
// Licensed under the Solderpad Hardware License v 2.1 (the “License”);
// you may not use this file except in compliance with the License, or,
// at your option, the Apache License version 2.0. You may obtain a copy
// of the License at
//
// https://solderpad.org/licenses/SHL-2.1/
//
// Unless required by applicable law or agreed to in writing, any work
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// --------------------------------------------------------------------

module tb_base #(N, realtime PERIODS[N])
( output bit tb_clk[N]
, output bit tb_aresetn
, output bit tb_reset[N]
);
timeunit 1ns;
timeprecision 100ps;

  // --------------------------------------------------------------------
  function void assert_reset(realtime reset_assert);
    fork
      begin
      tb_aresetn = 0;
      #reset_assert;
      tb_aresetn = 1;
      end
    join_none
  endfunction

  // --------------------------------------------------------------------
  bit disable_clks[N];

  generate
    for(genvar j = 0; j < N; j++) begin : g_clk
      always
        if(disable_clks[j])
          tb_clk[j] = 0;
        else
          #(PERIODS[j]/2) tb_clk[j] = ~tb_clk[j];
    end
  endgenerate

  // --------------------------------------------------------------------
  generate
    for(genvar j = 0; j < N; j++) begin : g_reset
      bit reset = 1;
      assign tb_reset[j] = reset;

      always @(posedge tb_clk[j] or negedge tb_aresetn)
        if(~tb_aresetn)
          reset = 1;
        else
          reset = 0;
    end
  endgenerate

  // --------------------------------------------------------------------
  initial
    assert_reset((PERIODS[0] * 5) + (PERIODS[0] / 3));

// --------------------------------------------------------------------
endmodule
