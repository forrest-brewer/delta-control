#
#

# +UVM_VERBOSITY=UVM_DEBUG
# +UVM_VERBOSITY=UVM_HIGH
# +UVM_TESTNAME=t_debug

-L tb_base
# -L AXI4_LIB

-voptargs=+acc=npr+/tb_top
-voptargs=+acc=npr+/tb_top/dut
