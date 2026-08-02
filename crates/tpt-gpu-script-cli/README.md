# tptb-cli

Command-line interface for the TPT Script compiler — the `tpt` binary.

## Overview

`tptb-cli` exposes `tptb-core`'s full pipeline (lex → parse → type-check → codegen) as a command-line tool. It compiles `.tpts` source files to TPTIR or Rust output.

## Installation

```sh
cargo install tpt-gpu-script-cli
```

## Usage

```sh
tpt build my_kernel.tpts        # compile to TPTIR
tpt check my_kernel.tpts        # type-check only
tpt fmt my_kernel.tpts          # format in-place
```

## License

Apache-2.0 — see the [repository](https://github.com/tpt-solutions/tpt-gpu) for details.
