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

global env

set env(ROOT_DIR) ../../../..
set env(PROJECT_DIR) ../../..
set env(TB_BASE_DIR) "$env(ROOT_DIR)/tb_base"

dict set lib_tb_base name "tb_base"
dict set lib_tb_base dir $env(TB_BASE_DIR)
set env(LIB_TB_BASE) $lib_tb_base

do $env(TB_BASE_DIR)/scripts/sim_procs.do
