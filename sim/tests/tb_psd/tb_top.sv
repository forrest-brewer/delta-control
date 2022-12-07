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
  // import uvm_pkg::*;
  import tb_top_pkg::*;
  // import axis_pkg::*;
  // import fifo_pkg::*;
  // `include "uvm_macros.svh"

  // --------------------------------------------------------------------
  localparam real OSR = 256;      // oversample ratio
  localparam real FB  = 22050;    // nyquist
  localparam real FS  = OSR*2*FB; // sampling frequency
  localparam real TS  = (1/FS)*1s;    // sampling period

  // --------------------------------------------------------------------
  // localparam realtime PERIODS[1] = '{TS/100ps};
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

  // // --------------------------------------------------------------------
  // fifo_if #(.W(W), .D(D)) dut_if(.*);

  // sync_fifo #(.W(W), .D(D))
    // dut
    // (
      // .wr_full(dut_if.wr_full),
      // .wr_data(dut_if.wr_data),
      // .wr_en(dut_if.wr_en),
      // .rd_empty(dut_if.rd_empty),
      // .rd_data(dut_if.rd_data),
      // .rd_en(dut_if.rd_en),
      // .count(dut_if.count),
      // .clk(dut_if.clk),
      // .reset(dut_if.reset)
    // );


  // --------------------------------------------------------------------
  // Call it with the parameters and the time and it returns the signal
  // value at that time. The w1, w2 are angular frequencies,
  // so 2πf1, 2πf2, M is the chirp duration, A is the peak amplitude.

  localparam pi = 3.1415926535897931;
  logic [15:0] data = 'hffff;
  // real signal;

  function real Chirp(input real w1, w2, A, M, t);
    Chirp = A*$cos(2*pi*w1*t+(2*pi*w2-2*pi*w1)*t*t/(2*M));
  endfunction

// double Chirp(double w1, double w2, double A, double M, double time)
   // {
   // double res;
   // res=A*cos(w1*time+(w2-w1)*time*time/(2*M));
   // return res;
   // }

  // --------------------------------------------------------------------
  real tscale_unit;
  realtime t_edge1;
  realtime t_edge2;
  realtime t_event;
  realtime prev_time;
  real clk_freq;

  realtime tp_scale_list[] = '{1fs, 10fs, 100fs, 1ps, 10ps, 100ps, 1ns, 10ns, 100ns};
  realtime time_scaler_list[] = '{1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7};
  realtime time_scaler;
  realtime tp_scale;
  realtime tp_prev;
  realtime tp;
  realtime tu;

  initial
  begin

    foreach(tp_scale_list[i])
    begin
      tp_prev = $realtime;
      #(tp_scale_list[i]);
      if($realtime - tp_prev != 0)
      begin
        tp = tp_scale_list[i];
        time_scaler = time_scaler_list[i];
        $display("timeprecision is %d| %g | %g", i, tp, time_scaler);
        break;
      end
    end

    foreach(tp_scale_list[i])
    begin
      tp_prev = $realtime;
      #1;
      // $display("*** %d| %g | %g", i, ($realtime - tp_prev), tp_scale_list[i]);

      if(($realtime - tp_prev) == tp_scale_list[i])
      begin
        tu = tp_scale_list[i];
        $display("timeunit is %d| %g", i, tu);
        break;
      end
    end

  end

  initial
  begin
    wait(~tb_reset[0]);

    t_edge1 = $realtime;
    #1;  // Single unit time delay
    $display("#1 | %g |", $realtime - t_edge1);
    tscale_unit = ($realtime - t_edge1) / 1fs;  //  Normalise the timescale into picoseconds (1*10^-12)

    t_edge1 = $realtime;
    #1ns;  // Single unit time delay
    $display("#1ns | %g |", $realtime - t_edge1);

    t_edge1 = $realtime;
    #1ps;  // Single unit time delay
    $display("#1ps | %g |", $realtime - t_edge1);


    @(posedge clk);
    t_edge2 = $realtime;
    @(posedge clk);
    t_edge1 = $realtime;
    clk_freq = 1.0s/((t_edge1 - t_edge2) * tscale_unit * 1fs);
    $display("tscale_unit | %g |", tscale_unit);
    $display("clk_freq    | %g |", clk_freq * tu  );
  end

  // --------------------------------------------------------------------
  // task tx_signal(input event e, input int signal[]);
    // fork
    // begin
      // foreach(signal[i])
      // begin
        // wait(e.triggered);
        // $display("%d | %h |", i, signal[i]);
      // end
    // end
    // join_none
  // endtask

  task tx_signal(input int signal[]);
    fork
    begin
      foreach(signal[i])
      begin
        @(posedge clk);
        // $display("%d | %h |", i, signal[i]);
        data = signal[i];
      end
    end
    join
  endtask


  // --------------------------------------------------------------------
  // tb_dut_config cfg_h = new(dut_if);
  // 8.8577097505668934240362811791383e-8
  // 0.000000088577
  // 0.0000008858

  int signal[];

  initial
  begin
    wait(~tb_reset[0]);
    repeat(32) @(posedge tb_clk[0]);

    // repeat(32) @(posedge tb_clk[0])
    // begin
      // // data = $rtoi(Chirp(1,FB,2**16-1,0.1,$realtime * time_scaler));
      // $display("%g | %g | %d"
              // , $realtime * time_scaler
              // , ($realtime - prev_time) * time_scaler
              // , data
              // );
      // prev_time = $realtime;
    // end

    $display("(1/FS)*1s  | %g |", (1/FS)*1s  );
    $display("TS/100ps   | %g |", TS/100ps   );
    $display("88.58ns    | %g |", 88.58ns    );
    $display("100ps / 1ns| %g |", 100ps / 1ns);
    $display("1ns / 1s   | %g |", 1ns / 1s   );

    $readmemh("../../python/chirp_hex.txt", signal);
    tx_signal(signal);

    wait fork;
    $display("Test Done!");
    do_stop();


    // uvm_config_db #(tb_dut_config)::set(null, "*", "tb_dut_config", cfg_h);
    // run_test("t_debug");
  end

  // // --------------------------------------------------------------------
  // always @(posedge tb_clk[0])
    // data = $rtoi(Chirp(1,FB,2**16-1,0.1,$realtime * time_scaler));



// --------------------------------------------------------------------
endmodule
