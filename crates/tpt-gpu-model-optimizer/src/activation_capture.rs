//! Activation Capture — hooks for capturing intermediate activations during inference.
//!
//! Provides a callback-based mechanism to intercept and record tensor activations
//! at key points in the forward pass (FFN intermediate outputs). This enables
//! real per-layer sensitivity analysis and domain mapping.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Activation statistics for a single layer's output.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LayerActivations {
    /// Layer index in the model
    pub layer_idx: usize,
    /// Mean activation magnitude per neuron (length = ffn_dim)
    pub mean_magnitudes: Vec<f32>,
    /// Standard deviation per neuron
    pub stddev_magnitudes: Vec<f32>,
    /// Number of samples captured for this layer
    pub sample_count: usize,
}

impl LayerActivations {
    pub fn new(layer_idx: usize, ffn_dim: usize) -> Self {
        LayerActivations {
            layer_idx,
            mean_magnitudes: vec![0.0f32; ffn_dim],
            stddev_magnitudes: vec![0.0f32; ffn_dim],
            sample_count: 0,
        }
    }

    /// Add an activation sample and update running mean/std (Welford's algorithm)
    pub fn add_sample(&mut self, activations: &[f32]) {
        if activations.len() != self.mean_magnitudes.len() {
            return;
        }
        self.sample_count += 1;
        let n = self.sample_count as f32;

        for (i, &act) in activations.iter().enumerate() {
            let delta = act - self.mean_magnitudes[i];
            self.mean_magnitudes[i] += delta / n;

            // Update variance using Welford's online algorithm
            if self.sample_count > 1 {
                let delta2 = act - self.mean_magnitudes[i];
                self.stddev_magnitudes[i] =
                    (self.stddev_magnitudes[i] * (n - 1.0) + delta * delta2) / n;
            }
        }
    }

    /// Get the mean absolute activation per neuron (for importance scoring)
    pub fn mean_abs(&self) -> Vec<f32> {
        self.mean_magnitudes.iter().map(|&m| m.abs()).collect()
    }
}

/// Full activation map: layer_idx → LayerActivations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActivationMap {
    pub layers: HashMap<usize, LayerActivations>,
    pub ffn_dim: usize,
}

/// Callback type for activation capture
pub type ActivationCallback = Box<dyn FnMut(usize, &[f32]) + Send>;

/// Context for activation capture during inference
pub struct ActivationCapture {
    pub ffn_dim: usize,
    pub activations: ActivationMap,
    pub callback: Option<ActivationCallback>,
}

impl ActivationCapture {
    pub fn new(ffn_dim: usize) -> Self {
        ActivationCapture {
            ffn_dim,
            activations: ActivationMap::default(),
            callback: None,
        }
    }

    /// Set an activation callback
    pub fn with_callback(mut self, callback: ActivationCallback) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Record activations for a given layer
    pub fn record(&mut self, layer_idx: usize, activations: &[f32]) {
        // Call the external callback if present
        if let Some(ref mut cb) = self.callback {
            cb(layer_idx, activations);
        }

        // Update internal statistics
        let layer_acts = self
            .activations
            .layers
            .entry(layer_idx)
            .or_insert_with(|| LayerActivations::new(layer_idx, self.ffn_dim));
        layer_acts.add_sample(activations);
    }

    /// Finalize and return the activation map
    pub fn finalize(self) -> ActivationMap {
        self.activations
    }

    /// Save activation map to JSON file
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.activations)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Extension trait for GpuInferenceEngine to add activation capture
pub trait ActivationCaptureExt {
    /// Enable activation capture with the given callback
    fn with_activation_capture(ffn_dim: usize, callback: ActivationCallback) -> Self;

    /// Get the captured activations
    fn activation_map(&self) -> &ActivationMap;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_layer_activations() {
        let mut capture = ActivationCapture::new(8);

        // Record some fake activations
        capture.record(0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        capture.record(0, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let map = capture.finalize();
        assert!(map.layers.contains_key(&0));
        let layer0 = &map.layers[&0];
        assert_eq!(layer0.sample_count, 2);
    }

    #[test]
    fn callback_is_invoked() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let mut capture =
            ActivationCapture::new(4).with_callback(Box::new(move |_layer, _acts| {
                call_count_clone.fetch_add(1, Ordering::SeqCst);
            }));

        capture.record(0, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }
}
