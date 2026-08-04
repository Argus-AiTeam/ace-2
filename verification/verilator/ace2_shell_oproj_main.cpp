#include <cstdint>
#include <iomanip>
#include <iostream>

#include "Vace2_shell_oproj_harness.h"
#include "verilated.h"

static void tick(Vace2_shell_oproj_harness& top) {
    top.clk_i = 0;
    top.eval();
    top.clk_i = 1;
    top.eval();
    top.clk_i = 0;
    top.eval();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vace2_shell_oproj_harness top;

    top.rst_ni = 0;
    for (int cycle = 0; cycle < 4; ++cycle) {
        tick(top);
    }
    top.rst_ni = 1;

    constexpr std::uint64_t kTimeoutCycles = 10'000'000;
    std::uint64_t cycles = 0;
    while (!top.done_o && cycles < kTimeoutCycles) {
        tick(top);
        ++cycles;
    }

    if (!top.done_o) {
        std::cerr << "ACE2_SHELL_VERILATOR_TB_FAIL timeout_cycles="
                  << cycles << '\n';
        return 1;
    }

    std::cout << "ACE2_SHELL_OPCODE_RESULT opcode=01 vectors="
              << top.vectors_o << " writes=" << top.writes_o
              << " signature=" << std::hex << std::setw(16)
              << std::setfill('0') << top.signature_o << std::dec
              << " total_cycles=" << top.total_cycles_o
              << " max_cycles=" << top.max_cycles_o << '\n';

    if (top.failures_o != 0) {
        std::cerr << "ACE2_SHELL_VERILATOR_TB_FAIL failures="
                  << top.failures_o << '\n';
        return 1;
    }

    std::cout << "ACE2_SHELL_VERILATOR_TB_PASS opcode=o_proj\n";
    return 0;
}
