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

// --------------------------------------------------------------------
 typedef enum
 { NONE
 , SPORADIC
 // , BURSTY
 , REGULAR
 } random_delay_type_e;

// --------------------------------------------------------------------
class random_delay;

  rand int unsigned delay = 0;
  time wait_timescale = 1ns;

  // --------------------------------------------------------------------
  virtual function void set_delay(random_delay_type_e kind = REGULAR);
    case(kind)
      NONE:     delay = 0;
      SPORADIC: assert(this.randomize() with{delay dist {0 := 96, [1:3] := 3, [4:7] := 1};});
      REGULAR:  assert(this.randomize() with{delay dist {0 := 60, [1:3] := 30, [4:7] := 10};});
      default:  delay = 0;
    endcase
  endfunction: set_delay

  // --------------------------------------------------------------------
  virtual task next(random_delay_type_e kind = REGULAR);
    set_delay(kind);
    #(delay * wait_timescale);
  endtask: next

  // --------------------------------------------------------------------
  virtual function int unsigned get(random_delay_type_e kind = REGULAR);
    set_delay(kind);
    return(delay);
  endfunction: get

// --------------------------------------------------------------------
endclass
