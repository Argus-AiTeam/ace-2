# ACE-2 Host-Trust Recovery Status

This page is a public-safe summary of the execution-admission work performed
after the Alpha 3 release. It does not distribute the recovery package,
migration payloads, receipts, protected evidence, local paths, or privileged
instructions.

## Latest accepted result

The V8 host-trust recovery package received Fresh-L2 acceptance with:

| Field | Public result |
|---|---|
| Package status | `BUILD_READY_EXTERNAL_ROOT_REQUIRED` |
| Verifier checks | 58 |
| Reported issues | 0 |
| Content SHA-256 | `07663099352edfad32eb39919ad9475f1f887328ebb549bdb9cae1c48f5ccad1` |
| Independent review | Accepted |
| Installed tree verified | No |
| Privileged execution performed | No |
| Stage 1 completed | No |

## Trust boundary

A package cannot establish independent trust by executing package-controlled
code before its own identity is authenticated. V8 therefore has no
package-self-authorizing entrypoint. The remaining prerequisite is an
independent external-root channel that supplies both:

1. the exact accepted package identity; and
2. the exact root invocation through a channel outside the package being
   authenticated.

The evaluated account must not be able to invoke, replace, or impersonate that
channel. Until this prerequisite exists, ACE-2 remains fail-closed.

## What this result does not claim

V8 acceptance is a build-readiness and trust-boundary result. It does not
claim:

- installation or host activation;
- privileged execution;
- completion of the arbitrary-text W4A8 chat milestone;
- new RTL certification beyond the preserved Alpha 2 baseline;
- XRT integration, bitstream generation, or U280 board execution.

The ordered product path remains Stage 1 local accelerator-facing chat before
Stage 2 U280 deployment.
