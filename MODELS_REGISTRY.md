# TPT Shared Model Registry

All tools in the TPT compute suite (tpt-gpu, tpt-spark, tpt-crucible) share a
single GGUF model directory so that models are downloaded once and never
duplicated.

---

## Canonical location

```
~/.tpt/models/
├── models.json          # manifest (see schema below)
├── llama-3-8b-q4.gguf
├── mistral-7b-q4.gguf
└── ...
```

On Windows the home directory is `%USERPROFILE%`, so the full path is
`%USERPROFILE%\.tpt\models\`.

---

## Manifest format — `models.json`

```json
{
  "version": "1",
  "models": [
    {
      "name":     "llama-3-8b-q4",
      "file":     "llama-3-8b-q4.gguf",
      "arch":     "llama3",
      "size_gb":  4.7,
      "sha256":   "abc123...",
      "source":   "https://huggingface.co/..."
    }
  ]
}
```

### Field definitions

| Field     | Required | Description |
|-----------|----------|-------------|
| `name`    | yes      | Human-readable, URL-safe identifier (used as lookup key) |
| `file`    | yes      | Filename relative to `~/.tpt/models/` |
| `arch`    | yes      | Model architecture tag: `llama3`, `mistral`, `phi3`, `gemma2`, etc. |
| `size_gb` | yes      | Approximate on-disk size in GiB |
| `sha256`  | no       | SHA-256 of the GGUF file for integrity verification |
| `source`  | no       | Original download URL |

---

## Tool responsibilities

### tpt-gpu (this repo)

- Provides the `tpt-gpu-model-registry` crate (`crates/tpt-gpu-model-registry/`) with:
  - `ModelRegistry::open()` — loads or creates `~/.tpt/models/models.json`
  - `ModelRegistry::register()` — adds or updates a model entry
  - `ModelRegistry::find_by_name()` — looks up a model
  - `ModelRegistry::download()` — downloads a GGUF from a URL and registers it
- The HuggingFace download helper (`crates/tpt-gpu-model-registry/src/hf.rs`) writes
  directly into `~/.tpt/models/` and updates the manifest on success.

### tpt-spark

- Reads `~/.tpt/models/models.json` to discover available models instead of
  maintaining its own directory.
- Passes the resolved `file` path to its `WgpuEngine` or `TptGpuEngine`.

### tpt-crucible

- Catalyst ingestion reads GGUF files from `~/.tpt/models/` using the manifest
  as the source of truth for architecture and quantisation metadata.

---

## CLI walkthrough — `tpt-gpu-models`

```bash
# Build the CLI (once)
cargo build --release -p tpt-gpu-model-registry

# Show the registry directory
./target/release/tpt-gpu-models dir
# → /home/user/.tpt/models

# List registered models (empty at first)
./target/release/tpt-gpu-models list
# → No models registered.

# Download a GGUF from HuggingFace and register it
./target/release/tpt-gpu-models fetch \
    --url  https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf \
    --name llama-2-7b-q4 \
    --arch llama \
    --size-gb 3.8
# → Downloading 'llama-2-7b-q4' from https://…
# → Downloaded and registered 'llama-2-7b-q4' at /home/user/.tpt/models/llama-2-7b.Q4_K_M.gguf.

# Register a file that is already on disk
./target/release/tpt-gpu-models add \
    --name  my-model \
    --file  my-model.gguf \
    --arch  mistral \
    --size-gb 4.1

# List after adding
./target/release/tpt-gpu-models list
# NAME                           ARCH       SIZE_GB  FILE
# llama-2-7b-q4                  llama      3.8      llama-2-7b.Q4_K_M.gguf
# my-model                       mistral    4.1      my-model.gguf

# Remove an entry from the manifest (file stays on disk)
./target/release/tpt-gpu-models remove my-model
# → Removed 'my-model'.
```

---

## End-to-end workflow — download, convert, and optimise

The complete path from a fresh HuggingFace download to an optimised model
ready for inference:

```bash
# 1. Download a GGUF from HuggingFace into the shared registry
./target/release/tpt-gpu-models fetch \
    --url  https://huggingface.co/google/gemma-2-2b-it-gguf/resolve/main/gemma-2-2b-it-q4_k_m.gguf \
    --name gemma-2-2b-it-q4 \
    --arch gemma2 \
    --size-gb 1.6
# Downloads to ~/.tpt/models/gemma-2-2b-it-q4_k_m.gguf and registers the entry.

# 2. Convert GGUF → TPTF (imports header metadata and raw tensor bytes)
tpt-gpu-model-optimizer convert \
    --input  ~/.tpt/models/gemma-2-2b-it-q4_k_m.gguf \
    --output gemma2.tptf

# 3. Optimise: sensitivity analysis → mixed-precision quantisation → TPTF output
tpt-gpu-model-optimizer optimize \
    gemma2.tptf \
    --output gemma2-opt.tptf

# 4. (Optional) Re-export to GGUF for llama.cpp / other tools
tpt-gpu-model-optimizer export \
    --format gguf \
    gemma2-opt.tptf \
    --output gemma2-opt.gguf
```

The `convert` step (step 2) is provided by
`crates/tpt-gpu-model-optimizer/src/import/gguf.rs` (`GgufImporter`).  It
reads the GGUF v2/v3 binary header, extracts all KV metadata (arch,
context length, hidden dim, layer count, head counts, FFN dim, vocab size),
maps per-tensor `ggml_type` codes to TPTF bit depths
(`F32→32`, `F16→16`, `Q8_0→8`, `Q6_K→6`, `Q4_K→4`, `Q2_K→2`),
and writes a self-contained `.tptf` file that the `optimize` step can
consume without needing access to the original GGUF.

---

## Versioning

The manifest `version` field is currently `"1"`. A breaking change to the
manifest schema will increment this number. Tools should refuse to parse
manifests with an unrecognised version and prompt the user to update.
