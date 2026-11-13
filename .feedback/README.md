# Feedback task starting environment

Upstream: https://github.com/fox-it/dissect.cstruct
Upstream commit: `65eeecb47a4c88d43217ef8f68631e8ac3322446`
Prepared on macOS with the Python version recorded in the environment plan. No external database, queue, or cloud account is required.

The dependency file pins the installed runtime and test packages for the selected baseline. It is not a claim that all optional upstream test suites are installed.

From the repository root:

```sh
export NO_INTERNET=1 TZ=UTC UV_NO_CONFIG=1

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r .feedback/requirements.txt -e .
.venv/bin/python -m pytest -q -o addopts= tests/test_compiler.py tests/test_types_pointer.py tests/test_types_structure.py tests/test_expression.py tests/test_parser.py tests/test_align.py tests/test_bitfield.py
```

Scope note: The scope is the command below, not all optional upstream integrations.

Observed baseline (2026-11-13): `416 passed in 0.47s`
Whole-command wall time: 0.604 seconds, exit code 0. This is one local measurement, not a runtime guarantee. The first install is excluded.

This seed adds preparation metadata only. Application sources and upstream tests are unchanged; there is no task solution in this commit.
