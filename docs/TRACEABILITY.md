# Public Traceability

| Claim | Included evidence |
|---|---|
| Synthesizable structural shell exists | `rtl/ace2_shell.sv` and listed cores |
| Full 434-item framework is represented | Shell/control descriptor framework |
| Accepted projection-prefix arithmetic is reproducible | Python oracle, deterministic vectors, projection testbench |
| Accepted numerical support reaches `layer_0.v_proj` | `make oracle-check` and `make sim` demonstrate the projection operator class |
| Full-model quality is accepted | **No; explicitly rejected** |
| `layer_0.rope_q` is supported end to end | **No; first unsupported operator** |
| Projection-shadow candidate is accepted | **No; bounded no-go** |

The default demo intentionally does not run unsupported downstream numerical
claims.
