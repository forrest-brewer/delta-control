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

package tb_top_pkg;
  // import fifo_pkg::*;
  // import uvm_pkg::*;
  // `include "uvm_macros.svh"

  // --------------------------------------------------------------------
  task do_stop;
    $stop;
  endtask
  
  // // --------------------------------------------------------------------
  // localparam fifo_cfg_t FIFO_CFG =
  // '{  W: 16
   // ,  D: 8
   // ,  UB: $clog2(FIFO_CFG.D)
  // };
  
  // // --------------------------------------------------------------------
  // localparam W = 16;
  // localparam D = 8;
  // localparam UB = $clog2(D);

  // // --------------------------------------------------------------------
  // `include "tb_dut_config.svh"
  // `include "tb_env.svh"
  // `include "s_debug.svh"
  // `include "t_top_base.svh"
  // `include "t_debug.svh"

// --------------------------------------------------------------------
endpackage
