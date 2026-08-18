# ACE-2 开放 IP 库

[English](IP_LIBRARY.md) | [机器可读目录](ip/catalog.json)

开放 IP 库把现有、已验证的 ACE-2 Qwen2.5-0.5B W4A8 RTL 整理为便于发现和复用的
功能包。所有包直接引用 `rtl/` 中的规范源码，不复制或分叉已认证实现。

## 目录

| 功能包 | 复用边界 | 公开证明 |
|---|---|---|
| [`w4a8_projection`](ip/w4a8_projection/) | 独立投影 core；矩阵角色由 shell 调度 | Q 投影 shell 模式 |
| [`rmsnorm`](ip/rmsnorm/) | 独立流式 core | 独立 RTL testbench |
| [`rope`](ip/rope/) | 独立 core | Q/K 成对 shell 模式 |
| [`kv_cache`](ip/kv_cache/) | 共享 shell 写入路径，不是独立 cache core | KV-write shell 模式 |
| [`attention`](ip/attention/) | 独立 score/compose core 加 shell 集成 | Score 与 Value 模式 |
| [`softmax`](ip/softmax/) | 独立的有限上下文 core | Softmax shell 模式 |
| [`silu_swiglu`](ip/silu_swiglu/) | 独立激活/门控 core | SiLU shell 模式 |
| [`mlp`](ip/mlp/) | 集成包，不是单一独立 core | Gate/Up/SiLU/Down/Residual |
| [`qwen25_transformer_layer`](ip/qwen25_transformer_layer/) | shell 集成包，不是即插即用的 layer core | 选定的 Residual/Norm 集成模式 |

现有 18 个算子 Demo 证明了 ACE-2 数据路径的支持范围，但多个名称会共享 core 或 shell
证明。因此 manifest 会明确标注“独立 core”“共享 shell 路径”或“集成包”。

## 查看与运行

```sh
make ip-list
make ip-validate
make ip-demo IP=rmsnorm
make ip-softmax                 # 快捷命令
make ip-demo-all                # 包含耗时较长的 MLP-up 证明
```

结构化结果写入 `build/ip_library/<package>/result.json`。只有所有映射的底层 RTL
证明都成功并出现所需 PASS marker，功能包才会报告 PASS。

## 集成示例

请直接使用规范源码。例如实例化投影 core：

```systemverilog
ace2_w4a8_proj_core #(
    .K_SIZE(896), .MAC_LANES(4), .ACT_WIDTH(8),
    .WGT_WIDTH(4), .ACC_WIDTH(32)
) u_q_projection (/* 连接 README 所述 ready/valid 端口 */);
```

从仓库根目录编译 `rtl/ace2_w4a8_proj_core.sv`，不要把源码复制到 `ip/` 包内。矩阵遍历、
存储器寻址、权重与 scale 供给，以及 Q/K/V/O/MLP 角色调度仍属于集成职责，目前由
`ace2_shell` 实现。Shell 现还提供 Fused `0x0b` 指令：缓存一个 Activation Tile，
并通过共享 Projection Engine 按顺序执行 Q/K/V。它属于集成 Bundle，不代表三套
物理并行的 Projection Core。

## 能力边界

本库兼容已演示的 Qwen2.5-0.5B W4A8 定点路径（hidden size 896、intermediate size
4864、head dimension 64）。它不声明任意 Transformer 支持、稳定的完整模型推理 API、
任意文本对话、模型权重或 FPGA 部署。

## 贡献

欢迎贡献可移植 wrapper、独立 oracle、可再分发测试向量、文档和集成，但必须保持证据
边界。新增包应明确规范源码、接口、依赖、证明映射、限制和 Apache-2.0 许可证继承。
详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
