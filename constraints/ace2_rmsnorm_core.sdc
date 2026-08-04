create_clock -name clk_i -period 10.000 [get_ports clk_i]
set_input_delay 0.200 -clock clk_i [get_ports -filter {direction == input && name != clk_i} *]
set_output_delay 0.200 -clock clk_i [all_outputs]
set_clock_uncertainty 0.100 [get_clocks clk_i]
