PYTHON ?= python3
SEED ?=
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

OPERATOR_DEMOS := input-rmsnorm q-proj k-proj v-proj rope-q rope-k kv-write \
	attention-score softmax attention-value o-proj attention-residual \
	post-attention-rmsnorm mlp-gate mlp-up silu mlp-down mlp-residual
IP_PACKAGES := w4a8_projection rmsnorm rope kv_cache attention softmax \
	silu_swiglu mlp qwen25_transformer_layer

.PHONY: demo demo-extended demo-operator demo-operators $(addprefix demo-,$(OPERATOR_DEMOS)) ip-list ip-validate ip-docs ip-demo ip-demo-all $(addprefix ip-,$(IP_PACKAGES)) visuals schematic certified-rtl-check env-check oracle-check lint sim local-challenge challenge-sim negative-control operator-suite full-shell-sim demo-report synth manifest clean

demo: certified-rtl-check env-check lint oracle-check sim local-challenge challenge-sim negative-control operator-suite demo-report
	@echo "ACE2_LOCAL_RTL_DEMO_PASS"
	@echo "Scope: certified RTL identity, fresh RMSNorm challenge, Transformer operators, and selected shell integration."
	@echo "Local challenge: build/demo_challenge/challenge.json"
	@echo "Operator suite: build/operator_demo/operator-suite.json"
	@echo "Waveform: build/demo_challenge/rmsnorm-waveform.vcd"
	@echo "Visual report: build/DEMO_REPORT.html"
	@echo "Text report: build/DEMO_REPORT.md"

demo-extended: demo full-shell-sim
	@echo "ACE2_EXTENDED_SHELL_DEMO_PASS"
	@echo "Full shell log: build/operator_demo/full-shell.log"
	@echo "MLP-up proof log: build/operator_demo/mlp-up.log"

demo-operator:
	@test -n "$(OP)" || { echo "Usage: make demo-operator OP=<operator>"; exit 2; }
	@$(PYTHON) scripts/run_single_operator_demo.py "$(OP)"

demo-operators:
	@printf '%s\n' $(OPERATOR_DEMOS)

$(addprefix demo-,$(OPERATOR_DEMOS)):
	@$(PYTHON) scripts/run_single_operator_demo.py "$(@:demo-%=%)"

ip-list:
	@$(PYTHON) scripts/run_ip_demo.py --list

ip-validate:
	@$(PYTHON) scripts/validate_ip_catalog.py

ip-docs:
	@$(PYTHON) scripts/generate_ip_package_docs.py

ip-demo:
	@test -n "$(IP)" || { echo "Usage: make ip-demo IP=<package>"; exit 2; }
	@$(PYTHON) scripts/run_ip_demo.py "$(IP)"

ip-demo-all:
	@$(PYTHON) scripts/run_ip_demo.py --all

$(addprefix ip-,$(IP_PACKAGES)):
	@$(PYTHON) scripts/run_ip_demo.py "$(@:ip-%=%)"

visuals: demo schematic
	@echo
	@echo "ACE2_VISUAL_ARTIFACTS_PASS"
	@echo "Netlist schematic: build/ace2_w4a8_proj_schematic.svg"

certified-rtl-check:
	@echo
	@echo "== [1/6] Verifying the certified RTL source identity =="
	@sha256sum -c CERTIFIED_RTL.sha256
	@echo "ACE2_CERTIFIED_RTL_HASH_PASS"

env-check:
	@echo
	@echo "== [2/6] Checking required open-source tools =="
	@command -v $(PYTHON) >/dev/null || { echo "Missing Python 3."; exit 2; }
	@command -v $(VERILATOR) >/dev/null || { echo "Missing Verilator. Install it using your platform package manager."; exit 2; }
	@command -v $(IVERILOG) >/dev/null || { echo "Missing Icarus Verilog (iverilog). Install it using your platform package manager."; exit 2; }
	@command -v $(VVP) >/dev/null || { echo "Missing Icarus runtime (vvp). Install it using your platform package manager."; exit 2; }
	@echo "ACE2_ENV_CHECK_PASS"

oracle-check:
	@echo
	@echo "== [4/6] Regenerating independent RMSNorm oracle vectors =="
	@mkdir -p build
	@cp verification/generated/rmsnorm_vectors.json build/rmsnorm_vectors.json.expected
	@cp verification/generated/rmsnorm_vectors.svh build/rmsnorm_vectors.svh.expected
	@$(PYTHON) tools/gen_rmsnorm_vectors.py
	@cmp build/rmsnorm_vectors.json.expected verification/generated/rmsnorm_vectors.json
	@cmp build/rmsnorm_vectors.svh.expected verification/generated/rmsnorm_vectors.svh
	@echo "ACE2_RMSNORM_ORACLE_PASS"

lint:
	@echo
	@echo "== [3/6] Linting the complete structural accelerator shell =="
	@mkdir -p build
	$(VERILATOR) --lint-only --language 1800-2017 --timescale 1ns/1ps \
		-Wall -Wno-fatal $(RTL_INCLUDE_FLAGS) \
		--top-module ace2_shell $(RTL_SOURCES) 2>&1 | tee build/verilator-lint.log
	@echo "ACE2_RTL_LINT_PASS"

sim:
	@echo
	@echo "== [5/6] Simulating RMSNorm RTL against oracle outputs =="
	@mkdir -p build
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb \
		-o build/ace2_rmsnorm_tb.vvp $(RTL_SOURCES) \
		verification/tb/ace2_rmsnorm_tb.sv
	$(VVP) build/ace2_rmsnorm_tb.vvp | tee build/rmsnorm-sim.log
	@grep -q "ACE2_RMSNORM_TB_PASS" build/rmsnorm-sim.log
	@echo "ACE2_RMSNORM_RTL_SIM_PASS"

local-challenge:
	@echo
	@echo "== [6/9] Creating a fresh machine-local challenge =="
	@rm -rf build/demo_challenge
	@$(PYTHON) scripts/create_demo_challenge.py

challenge-sim:
	@echo
	@echo "== [7/9] Compiling and simulating the fresh challenge =="
	$(IVERILOG) -g2012 -DACE2_DEMO_CHALLENGE $(RTL_INCLUDE_FLAGS) -Iverification/tb \
		-o build/demo_challenge/ace2_rmsnorm_challenge.vvp $(RTL_SOURCES) \
		verification/tb/ace2_rmsnorm_tb.sv
	$(VVP) build/demo_challenge/ace2_rmsnorm_challenge.vvp | tee build/demo_challenge/challenge-sim.log
	@grep -q "ACE2_RMSNORM_TB_PASS" build/demo_challenge/challenge-sim.log
	@test -s build/demo_challenge/rmsnorm-waveform.vcd
	@$(PYTHON) scripts/render_demo_waveform.py
	@test -s build/demo_challenge/rmsnorm-waveform.svg
	@echo "ACE2_LOCAL_CHALLENGE_RTL_PASS"

negative-control:
	@echo
	@echo "== [8/9] Proving the checker rejects a corrupted expectation =="
	@$(PYTHON) scripts/create_negative_control.py
	@$(IVERILOG) -g2012 -DACE2_DEMO_NEGATIVE $(RTL_INCLUDE_FLAGS) -Iverification/tb \
		-o build/demo_challenge/ace2_rmsnorm_negative.vvp $(RTL_SOURCES) \
		verification/tb/ace2_rmsnorm_tb.sv
	@set +e; \
	$(VVP) build/demo_challenge/ace2_rmsnorm_negative.vvp > build/demo_challenge/negative-control.log 2>&1; \
	rc=$$?; set -e; \
	if [[ $$rc -eq 0 ]]; then \
		echo "Negative control unexpectedly passed"; \
		cat build/demo_challenge/negative-control.log; \
		exit 1; \
	fi
	@grep -q "OUTPUT_MISMATCH" build/demo_challenge/negative-control.log
	@echo "ACE2_NEGATIVE_CONTROL_PASS expected_corruption_was_rejected" | tee -a build/demo_challenge/negative-control.log

operator-suite:
	@echo
	@echo "== [9/10] Generating random Python oracles and running Transformer operators =="
	@SEED="$(SEED)" $(PYTHON) scripts/run_operator_demo.py

full-shell-sim:
	@echo
	@echo "== Running the complete public ace2_shell regression (slow) =="
	@mkdir -p build/operator_demo
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb \
		-o build/operator_demo/ace2_shell_full.vvp $(RTL_SOURCES) \
		verification/tb/ace2_shell_tb.sv
	$(VVP) build/operator_demo/ace2_shell_full.vvp | tee build/operator_demo/full-shell.log
	@grep -q "ACE2_SHELL_TB_PASS" build/operator_demo/full-shell.log
	@echo
	@echo "== Proving the MLP-up path omitted by the default shell schedule =="
	$(VVP) build/operator_demo/ace2_shell_full.vvp +MLP_UP_ONLY | tee build/operator_demo/mlp-up.log
	@grep -q "ACE2_SHELL_MLP_UP_TB_PASS" build/operator_demo/mlp-up.log
	@$(PYTHON) scripts/generate_demo_report.py

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
	@echo "== [10/10] Generating the local visual evidence dashboard =="
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
