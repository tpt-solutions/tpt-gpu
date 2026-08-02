# tpt-gpu-script-format

TPT Script formatter and linter — canonical pretty-printer and style enforcer for `.tpts` source files.

## Overview

`tpt-gpu-script-format` consumes a `tpt-gpu-script-core` AST and re-emits normalized TPT Script source. It is the backend for the `tpt fmt` subcommand and the LSP formatting provider.

## Usage

```toml
[dependencies]
tpt-gpu-script-format = "1.0"
```

## License

Apache-2.0 — see the [repository](https://github.com/tpt-solutions/tpt-gpu) for details.
