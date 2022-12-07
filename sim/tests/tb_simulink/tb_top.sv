// --------------------------------------------------------------------
// Copyright 2020 qaztronic
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

module tb_top;
  timeunit 1ns;
  timeprecision 100ps;
  import tb_top_pkg::*;

  // --------------------------------------------------------------------
  localparam real OSR = 256;      // oversample ratio
  localparam real FB  = 22050;    // nyquist
  localparam real FS  = OSR*2*FB; // sampling frequency
  localparam real TS  = (1/FS)*1s;    // sampling period

  // --------------------------------------------------------------------
  localparam realtime PERIODS[1] = '{88.577ns};
  localparam CLOCK_COUNT = $size(PERIODS);

  // --------------------------------------------------------------------
  bit tb_clk[CLOCK_COUNT];
  wire tb_aresetn;
  bit tb_reset[CLOCK_COUNT];

  tb_base #(.N(CLOCK_COUNT), .PERIODS(PERIODS)) tb(.*);

  // --------------------------------------------------------------------
  wire clk = tb_clk[0];
  wire reset = tb_reset[0];

  // --------------------------------------------------------------------
  logic clk_enable = 0;
  logic signed [15:0] input_rsvd = 0;  // sfix16_En7
  wire  ce_out;
  wire  output_rsvd;

  SD_2nd_Order_Modulator
    dut(.*);

  // --------------------------------------------------------------------
  int tx_signal[];

  task send_signal();
    fork
    begin
      foreach(tx_signal[i])
      begin
        @(posedge clk);
        input_rsvd = tx_signal[i];
      end
    end
    join_none
  endtask

  // --------------------------------------------------------------------
  logic rx_signal[];

  task get_signal();
    fork
    begin
      foreach(rx_signal[i])
      begin
        @(posedge clk);
        rx_signal[i] = output_rsvd;
      end
    end
    join_none
  endtask

  // --------------------------------------------------------------------
  initial
  begin
    wait(~tb_reset[0]);
    repeat(32) @(posedge tb_clk[0]);
    clk_enable = 1;
    repeat(32) @(posedge tb_clk[0]);

    // $readmemh("../../python/chirp_hex.txt", tx_signal);
    $readmemh("../../python/noise_hex.txt", tx_signal);
    rx_signal = new[tx_signal.size()];
    send_signal();
    get_signal();

    wait fork;
    $writememh("../../python/rx_tb_simulink.txt", rx_signal);
    $display("Test Done!");
    do_stop();
  end

// --------------------------------------------------------------------
endmodule
