# Alpha 2 Verification Report

Release preparation verifies:

1. every file in `CERTIFIED_RTL.sha256` matches the certified RTL source hash;
2. the deterministic RMSNorm oracle and RTL demo pass;
3. the release manifest regenerates and verifies;
4. no credential-like file names, private-key material, GitHub tokens,
   machine-local home-directory paths, or owner email addresses occur in
   tracked release content;
5. model weights, raw private evidence, build outputs, and active chat/U280
   development are absent.

Full model and PPA results are summarized in
[CERTIFICATION.md](CERTIFICATION.md). Their sealed raw packets are retained in
the controlled internal archive rather than duplicated into this repository.
