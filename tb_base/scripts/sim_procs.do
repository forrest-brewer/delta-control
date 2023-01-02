# --------------------------------------------------------------------
# Copyright 2020 qaztronic
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# Licensed under the Solderpad Hardware License v 2.1 (the “License”);
# you may not use this file except in compliance with the License, or,
# at your option, the Apache License version 2.0. You may obtain a copy
# of the License at
#
# https://solderpad.org/licenses/SHL-2.1/
#
# Unless required by applicable law or agreed to in writing, any work
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# --------------------------------------------------------------------

# ------------------------------------
proc make_lib { lib {rebuild "false"} } {
  set fbasename [file rootname [file tail $lib]]

  if {[file isdirectory  $lib]} {
    echo "INFO: Simulation library $lib already exists"

    if { $rebuild == "rebuild" } {
      echo "INFO: Rebuilding library. Deleting and recompiling $lib"
      quit -sim
      file delete -force ./$lib
      vlib $lib
      vmap $fbasename $lib
    }

  } else {
      echo "INFO: Creating Simulation library $lib"
      vlib $lib
      vmap $fbasename $lib
  }
}

# ------------------------------------
proc sim_compile_lib {lib {rebuild "false"} } {
  global env

  dict with lib {
    if { !([file isdirectory $dir/sim/libs]) } {
      file mkdir $dir/sim/libs
    }

    if {([file isdirectory $dir/sim/libs/$name]) && ($rebuild == "false")} {
      echo "INFO: Simulation library $dir/sim/libs/$name already exists"
      return
    }

    foreach src [glob -nocomplain -type d $dir/src/*] {
      echo "INFO: compiling $src to $name"
      make_lib $dir/sim/libs/$name

      if {[file exists $src/files.f]} {
        vlog -work $name -f $src/files.f
      }
    }

    foreach sim [glob -nocomplain -type d $dir/sim/src/*] {
      echo "INFO: compiling $sim to $name"
      make_lib $dir/sim/libs/$name

      if {[file exists $sim/files.f]} {
        vlog -work $name -f $sim/files.f
      }
    }
  }
}

# ------------------------------------
proc sim_restart {  } {
  global env

  if { [file exists ./wip.do] } {
    echo "INFO: found ./wip.do"
    do ./wip.do
  }

  if { [file exists ./pre_sim.do] } {
    echo "INFO: found ./pre_sim.do"
    do ./pre_sim.do
  }

  if { [string equal nodesign [runStatus]] } {
    sim_run_sim
  } else {
    restart -force
  }

  run -all
  echo "INFO: run -all done."

  if { [file exists ./post_sim.do] } {
    echo "INFO: found ./post_sim.do"
    do ./post_sim.do
  }
}

# ------------------------------------
proc sim_run_sim {  } {

  if { [file exists ./pre_sim.do] } {
    echo "INFO: found ./pre_sim.do"
    do ./pre_sim.do
  }

  if {[file exists ./sim.do]} {
    do ./sim.do
  } else {
    do ../../libs/sim.do
  }

  if { [file exists ./wave.do] } {
    do ./wave.do
  }

  if { [file exists ./post_sim.do] } {
    echo "INFO: found ./post_sim.do"
    do ./post_sim.do
  }

  run -all
}
