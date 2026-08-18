# Agent Note: Project .env Bootstrap-Only Variables Warn Instead of Crash

Status: implemented

English | [中文](2026-08-18-project-env-bootstrap-vars-warn-skip.zh.md)

## Problem

`loadLayeredEnv` reads the invoking directory's `.env` (the project layer) and
the Harness home's `.env` (the user layer) as credential sources (the [user
environment layer decision](../architecture/2026-08-04-credentials-yaml-and-user-environment-layer.md)),
rejecting any bootstrap-only variable (`PATH`, `PYTHONPATH`, `NODE_OPTIONS`,
`DSH_*`, …) with a hard error before applying anything. The project layer is an
arbitrary user project's own file, however: a Python project legitimately sets
`PYTHONPATH=./` for its own tooling, so launching `dsh` from such a project
aborts with `… .env sets "PYTHONPATH", which only the launching environment may
set …` even though dsh never needed that variable.

## Decision

The project layer drops bootstrap-only names with a warning instead of throwing;
the user layer (`$DSH_HOME/.env`), which is dsh's own configuration, keeps the
hard error. A bootstrap-only value is never applied from either file — only the
inherited environment may supply it — so the security property (a discovered
`.env` cannot hijack process, runtime, VCS, or network bootstrap) is unchanged.
The project layer's remaining values are still applied as before.

`readEnvLayer` gains a `reject` parameter (default `true`); `loadLayeredEnv`
passes `false` for the project layer. Both layers are still parsed before either
is applied, so a user-layer rejection never leaves the project layer applied.

## Verification

`loadLayeredEnv` unit tests: the hard-error cases now exercise the user layer
(`$DSH_HOME/.env`) and still assert the throw and that nothing is applied; a new
case asserts a project `.env` containing `DSH_PERMISSION_MODE` and `PYTHONPATH`
warns, applies the remaining value, and never materializes the bootstrap names.

## Alternatives considered

**Remove `PYTHONPATH` from every project `.env`.** The dsh error message
suggests exporting the variable in the shell, but a project's own `.env`
legitimately carries such variables for its own tooling; this would be a
whack-a-mole fix that still crashes in the next project with a bootstrap-only
variable.

**Warn and skip in both layers.** The user layer is dsh-owned configuration; a
bootstrap-only variable there is a genuine dsh misconfiguration that should
still fail loud.

## Consequences

Launching dsh inside an arbitrary project no longer aborts on that project's
`.env`; bootstrap-only names there are ignored with a single-line warning.
`$DSH_HOME/.env` misconfiguration still fails loud. Bootstrap-only variables
remain never-applied from any discovered file.
