# delta-control
python and Verilog implementations of $\Sigma\Delta$ control filters.
Start [here](https://github.com/forrest-brewer/delta-control/blob/main/notebooks/0_top.ipynb) for the background information.

## Directory Structure

### SD_Filter_Design
Original Matlab code from Joseph Poverelli

### examples
Various examples using the Python port of $\Sigma\Delta$ Stream Computation

### notebooks
Jupyter notebooks demonstrating the Python port of $\Sigma\Delta$ Stream Computation

### sd_filter_fp
Simulink fixed point implementation

### sdfpy
Python code used in $\Sigma\Delta$ Stream Computation

### sim
SystemVerilog simulation files
- [Test Bench](https://github.com/forrest-brewer/delta-control/blob/main/sim/tests/tb_simulink) for the Simulink generated Verilog files
- [Test Bench](https://github.com/forrest-brewer/delta-control/blob/main/sim/tests/tb_sd_filter) for the Synthesizable and parameterizable SystemVeriog code

### src
Synthesizable SystemVerilog implementations

### tb_base
SystemVerilog simulation library



