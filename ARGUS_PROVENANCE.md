# Argus Design Provenance

## Attribution

**ACE** means **Argus Compute Engine**. ACE-2 was designed and iterated by
**Argus — Autonomous Research Generation & Understanding System** under a
human-owned mission and explicit approval gates.

Public Argus resources:

- Argus AI Team projects: <https://github.com/Argus-AiTeam>
- Official website: <https://argusbot.cn/>
- Results: <https://argusbot.cn/results.html>
- Research: <https://argusbot.cn/research.html>

“Designed by Argus” means that Argus performed the iterative engineering work:
contract analysis, architecture proposals, RTL generation, reference-model
construction, test generation, verification, quality localization, evidence
binding, independent review handoffs, and rollback/reroute decisions. The human
operator retained ownership of objectives, budgets, authorization, release,
and external actions.

ACE-2 follows the same public presentation principle as other Argus AI Team
projects: lead with the strongest measured result, link it to reproducible
artifacts, and state the claim boundary beside the result. Autonomous execution
is part of the provenance; it does not weaken the evidence requirement.

## Engineering workflow

Argus used a staged, evidence-driven loop:

```text
human mission and immutable targets
  -> architecture contract
  -> environment/tool qualification
  -> bounded RTL implementation
  -> independent reference and focused RTL checks
  -> full-model quality discriminator
  -> reviewer decision
  -> accept, repair, or rollback
```

The harness deliberately distinguishes:

- **structural coverage:** source, interfaces, control, and executable RTL exist;
- **local correctness:** an isolated block agrees with its oracle;
- **end-to-end acceptance:** the composed model remains within the quality gate.

This distinction is why the Alpha can contain a broad RTL framework while its
accepted numerical frontier remains narrower.

## What Argus produced for the Alpha

- accelerator architecture and interface specifications;
- synthesizable compute and shell RTL;
- independent fixed-point reference implementations;
- deterministic test-vector generation;
- RTL testbenches, lint, simulation, synthesis, and PPA workflows;
- traceability and evidence artifacts;
- benchmark-driven failure localization;
- bounded negative results for rejected numerical candidates.

## Evidence-driven rejection

Argus did not convert a clean standalone test into a false full-model success.
When full-model benchmarking contradicted earlier structural evidence, it
rolled the accepted frontier back to the earliest unsupported operator. Later
candidate mechanisms were accepted only for bounded implementation and were
sealed as no-go when they failed quality gates.

The current Alpha therefore states both:

1. the engineering framework is substantial and reproducible; and
2. usable end-to-end Qwen inference is not yet demonstrated.

## Human and agent responsibilities

| Responsibility | Owner |
|---|---|
| Mission and product goal | Human operator |
| Immutable quality, area, and frequency targets | Human-approved contract |
| Architecture/RTL/test iteration | Argus |
| Independent evidence review | Argus reviewer role |
| Ordinary reversible execution | Argus under authorization |
| Credentials, publication, and external release | Human operator |
| Final project license | Human/project owner |

## Reproducibility

The release package excludes Argus runtime state and private infrastructure.
Its public demo does not require the Argus harness. This separation lets others
reproduce the packaged RTL evidence while still preserving attribution to the
system that designed and iterated it.
