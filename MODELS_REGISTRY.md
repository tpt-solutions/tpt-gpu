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
    --arch llama2 \
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
# llama-2-7b-q4                  llama2     3.8      llama-2-7b.Q4_K_M.gguf
# my-model                       mistral    4.1      my-model.gguf

# Remove an entry from the manifest (file stays on disk)
./target/release/tpt-gpu-models remove my-model
# → Removed 'my-model'.
```

---

## Versioning

The manifest `version` field is currently `"1"`. A breaking change to the
manifest schema will increment this number. Tools should refuse to parse
manifests with an unrecognised version and prompt the user to update.
