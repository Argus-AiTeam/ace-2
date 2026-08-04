read_liberty /OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog build/sky130_rmsnorm/ace2_shell_mapped_sta.v
link_design ace2_shell
read_sdc constraints/ace2_rmsnorm_core.sdc
read_vcd -scope ace2_shell_power_dump_tb/tb/dut ppa/raw/latest/ace2_shell_lm_head_power.vcd
report_power -digits 6
