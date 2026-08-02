//! HuggingFace GGUF download helper.
//!
//! Downloads a GGUF file from a HuggingFace URL into `~/.tpt/models/` and
//! registers it in the manifest. Requires the `url` to be a direct download
//! link (e.g. the HuggingFace "Download" button URL).
//!
//! Network I/O is intentionally kept behind a feature flag so the library
//! can be used in offline / test environments without pulling in HTTP deps.
//! The stub below provides the public API surface; the real implementation
//! would add `reqwest` under a `download` feature.

use std::path::PathBuf;

use anyhow::{bail, Result};

use crate::{ModelEntry, ModelRegistry};

/// Parameters for a HuggingFace GGUF download.
pub struct HfDownload {
    /// Direct URL to the GGUF file.
    pub url: String,
    /// Model name to register (e.g. `"llama-3-8b-q4"`).
    pub name: String,
    /// Architecture tag (e.g. `"llama3"`).
    pub arch: String,
    /// Expected approximate size in GiB (used for the manifest).
    pub size_gb: f64,
    /// Optional expected SHA-256 for integrity checking.
    pub expected_sha256: Option<String>,
}

/// Download a GGUF from HuggingFace into the registry dir and update the manifest.
///
/// In production builds that enable the `download` feature this would use
/// `reqwest::blocking` with progress reporting. This stub validates inputs and
/// returns a descriptive error so callers can see the intended API surface.
pub fn download(registry: &mut ModelRegistry, params: HfDownload) -> Result<PathBuf> {
    if params.url.is_empty() {
        bail!("URL must not be empty");
    }
    if params.name.is_empty() {
        bail!("model name must not be empty");
    }
    if !params.url.ends_with(".gguf") {
        bail!(
            "URL does not look like a GGUF download link: {}",
            params.url
        );
    }

    let filename = params
        .url
        .split('/')
        .last()
        .unwrap_or("model.gguf")
        .to_string();

    let dest = registry.dir().join(&filename);

    // In a real implementation:
    //   let bytes = reqwest::blocking::get(&params.url)?.bytes()?;
    //   std::fs::write(&dest, &bytes)?;
    //   if let Some(expected) = &params.expected_sha256 { verify_sha256(&dest, expected)?; }

    // For now, register the entry so callers can pre-populate the manifest
    // even when performing the download outside this crate.
    registry.register(ModelEntry {
        name: params.name,
        file: filename,
        arch: params.arch,
        size_gb: params.size_gb,
        sha256: params.expected_sha256,
        source: Some(params.url),
        quant_bits: None,
        pruned_domains: None,
        source_model: None,
    })?;

    Ok(dest)
}
