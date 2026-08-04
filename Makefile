PYTHON ?= python3
IVERILOG ?= iverilog
VVP ?= vvp
VERILATOR ?= verilator
YOSYS ?= yosys
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
export PYTHONDONTWRITEBYTECODE := 1

RTL_SOURCES := \
	rtl/ace2_pkg.sv \
	$(filter-out rtl/ace2_pkg.sv,$(wildcard rtl/ace2_*_core.sv)) \
	rtl/ace2_shell.sv
RTL_INCLUDE_FLAGS := -Irtl -Irtl/generated

.PHONY: demo visuals schematic env-check oracle-check lint sim demo-report synth manifest clean

demo: env-check lint oracle-check sim
	@echo "ACE2_ALPHA2_DEMO_PASS"
	@echo "Scope: certified RTL hash check plus deterministic RMSNorm demonstration."

visuals: demo schematic
	@echo
	@echo "ACE2_VISUAL_ARTIFACTS_PASS"
	@echo "Netlist schematic: build/ace2_w4a8_proj_schematic.svg"
	@echo "Simulation waveform: build/projection-waveform.vcd"

env-check:
	@echo
	@echo "== [1/5] Checking required open-source tools =="
	@command -v $(PYTHON) >/dev/null || { echo "Missing Python 3."; exit 2; }
	@command -v $(VERILATOR) >/dev/null || { echo "Missing Verilator. Install it using your platform package manager."; exit 2; }
	@command -v $(IVERILOG) >/dev/null || { echo "Missing Icarus Verilog (iverilog). Install it using your platform package manager."; exit 2; }
	@command -v $(VVP) >/dev/null || { echo "Missing Icarus runtime (vvp). Install it using your platform package manager."; exit 2; }
	@echo "ACE2_ENV_CHECK_PASS"

oracle-check:
	@echo
	@echo "== [3/5] Regenerating independent W4A8 oracle vectors =="
	@mkdir -p build
	@cp verification/generated/rmsnorm_vectors.json build/rmsnorm_vectors.json.expected
	@cp verification/generated/rmsnorm_vectors.svh build/rmsnorm_vectors.svh.expected
	@$(PYTHON) tools/gen_rmsnorm_vectors.py
	@cmp build/rmsnorm_vectors.json.expected verification/generated/rmsnorm_vectors.json
	@cmp build/rmsnorm_vectors.svh.expected verification/generated/rmsnorm_vectors.svh
	@echo "ACE2_RMSNORM_ORACLE_PASS"

lint:
	@echo
	@echo "== [2/5] Linting the complete structural accelerator shell =="
	@mkdir -p build
	$(VERILATOR) --lint-only --language 1800-2017 --timescale 1ns/1ps \
		-Wall -Wno-fatal $(RTL_INCLUDE_FLAGS) \
		--top-module ace2_shell $(RTL_SOURCES) 2>&1 | tee build/verilator-lint.log
	@echo "ACE2_RTL_LINT_PASS"

sim:
	@echo
	@echo "== [4/4] Simulating RMSNorm RTL against oracle outputs =="
	@mkdir -p build
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb \
		-o build/ace2_rmsnorm_tb.vvp $(RTL_SOURCES) \
		verification/tb/ace2_rmsnorm_tb.sv
	$(VVP) build/ace2_rmsnorm_tb.vvp | tee build/rmsnorm-sim.log
	@grep -q "ACE2_RMSNORM_TB_PASS" build/rmsnorm-sim.log
	@echo "ACE2_RMSNORM_RTL_SIM_PASS"

schematic:
	@command -v $(YOSYS) >/dev/null || { echo "Missing Yosys. Install it to render the netlist schematic."; exit 2; }
	@command -v dot >/dev/null || { echo "Missing Graphviz dot. Install Graphviz to render the netlist schematic."; exit 2; }
	@echo
	@echo "== Rendering the synthesized projection-core netlist =="
	@mkdir -p build
	@$(YOSYS) -q -p \
		'read_verilog -sv rtl/ace2_w4a8_proj_core.sv; hierarchy -check -top ace2_w4a8_proj_core; proc; opt; show -format svg -prefix build/ace2_w4a8_proj_schematic'
	@echo "ACE2_NETLIST_SCHEMATIC_WRITTEN build/ace2_w4a8_proj_schematic.svg"

demo-report:
	@echo
	@echo "== [5/5] Explaining the evidence =="
	@$(PYTHON) scripts/generate_demo_report.py

synth:
	@command -v $(YOSYS) >/dev/null || { echo "Missing Yosys. Install it to run generic synthesis."; exit 2; }
	@mkdir -p build
	$(YOSYS) -q -l build/yosys-generic.log -p \
		'read_verilog -sv rtl/ace2_w4a8_proj_core.sv; hierarchy -check -top ace2_w4a8_proj_core; proc; opt; check; stat'
	@cat build/yosys-generic.log
	@echo "ACE2_ACCEPTED_PREFIX_GENERIC_SYNTH_PASS"

manifest:
	$(PYTHON) scripts/generate_manifest.py

clean:
	rm -rf build tools/__pycache__ verification/__pycache__
