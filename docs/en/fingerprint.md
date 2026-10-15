# Fingerprint

A task vector stores the fingerprint of the base it was extracted against
(tensor names, shapes, dtypes, and a hash of an anchor tensor). At merge time
this is recomputed and compared, so a TV extracted against a different base
cannot be mixed in by accident.

Module: `goatmerge/fingerprint.py` — `compute_base_fingerprint`,
`model_manifest`, `tensor_content_hash`.

- The anchor tensor is hashed with row-chunked streaming (O(chunk)).
- `merge` compares each TV's stored fingerprint with the current base and
  refuses on mismatch (exit code 3). Bypass with `--skip-fingerprint-check`
  (CLI) / `skip_fingerprint_check` (YAML) only when you mean it.
- `verify_against_base` is the gate used by `cli.cmd_merge`.

Covered by `tests/test_fingerprint.py` (determinism, sensitivity) and the
fingerprint-gate / extract round-trip tests in
`tests/test_consensus_merge.py`.
