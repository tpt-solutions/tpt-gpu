//! Ollama (Local) AI Provider
//!
//! Uses the local Ollama server API. Requires a running Ollama instance
//! (default at http://localhost:11434). No API key needed.

use super::AiProvider;
use crate::error::{AiError, AiResult};
use crate::request::{AiMessage, AiRequest, Role};
use crate::response::{AiChoice, AiResponse, FinishReason, Usage};

/// Ollama local model provider
pub struct OllamaProvider {
    base_url: String,
    default_model: String,
    client: ureq::Agent,
    timeout_secs: u64,
}

impl OllamaProvider {
    /// Create with default localhost URL
    pub fn new() -> Self {
        OllamaProvider {
            base_url: "http://localhost:11434".to_string(),
            default_model: "llama3.1".to_string(),
            client: ureq::Agent::new(),
            timeout_secs: 60,
        }
    }

    /// Set custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set default model
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Set request timeout
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Check if Ollama server is reachable
    pub fn is_server_running(&self) -> bool {
        match self
            .client
            .get(&format!("{}/api/tags", self.base_url))
            .call()
        {
            Ok(resp) => resp.status() == 200,
            Err(_) => false,
        }
    }

    /// List locally available models
    pub fn list_local_models(&self) -> AiResult<Vec<String>> {
        let resp = self
            .client
            .get(&format!("{}/api/tags", self.base_url))
            .call()
            .map_err(|e| AiError::provider_unavailable("ollama", e.to_string()))?;
        let body: serde_json::Value = resp
            .into_json()
            .map_err(|e| AiError::serialization(e.to_string()))?;
        let models: Vec<String> = body
            .pointer("/models")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.pointer("/name").and_then(|v| v.as_str()))
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();
        Ok(models)
    }

    fn map_error(&self, status: u16, body: &str) -> AiError {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
            let msg = json
                .pointer("/error")
                .and_then(|v| v.as_str())
                .unwrap_or(body)
                .to_string();
            match status {
                400 => AiError::InvalidRequest {
                    message: msg,
                    code: None,
                },
                404 => AiError::ModelNotFound {
                    model: self.default_model.clone(),
                },
                500 => AiError::InternalError {
                    message: msg,
                    status_code: Some(500),
                },
                _ => AiError::Unknown { message: msg },
            }
        } else {
            AiError::InternalError {
                message: format!("HTTP {}: {}", status, body),
                status_code: Some(status),
            }
        }
    }

    fn parse_response(&self, resp: ureq::Response) -> AiResult<AiResponse> {
        let status = resp.status();
        let body = resp
            .into_string()
            .map_err(|e| AiError::serialization(e.to_string()))?;
        if status != 200 {
            return Err(self.map_error(status, &body));
        }
        let ollama_resp: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| AiError::serialization(e.to_string()))?;
        let model = ollama_resp
            .pointer("/model")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.default_model)
            .to_string();
        let created = ollama_resp
            .pointer("/created_at")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<u64>().ok());
        let prompt_eval_count = ollama_resp
            .pointer("/prompt_eval_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let eval_count = ollama_resp
            .pointer("/eval_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let usage = Usage {
            prompt_tokens: prompt_eval_count,
            completion_tokens: eval_count,
            total_tokens: prompt_eval_count + eval_count,
        };
        let content = ollama_resp
            .pointer("/message/content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let done = ollama_resp
            .pointer("/done")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let finish_reason = if done {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };
        let choices = vec![AiChoice {
            index: 0,
            message: AiMessage {
                role: Role::Assistant,
                content: content.to_string(),
            },
            finish_reason,
        }];
        Ok(AiResponse {
            id: None,
            model,
            choices,
            usage,
            created,
        })
    }
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AiProvider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }
    fn is_available(&self) -> bool {
        self.is_server_running()
    }
    fn default_model(&self) -> &str {
        &self.default_model
    }

    fn list_models(&self) -> Vec<String> {
        if let Ok(models) = self.list_local_models() {
            if !models.is_empty() {
                return models;
            }
        }
        vec![
            "llama3.1".to_string(),
            "llama3.1:8b".to_string(),
            "llama3.1:70b".to_string(),
            "codellama".to_string(),
            "mistral".to_string(),
            "mixtral".to_string(),
            "qwen2.5".to_string(),
            "phi4".to_string(),
            "gemma2".to_string(),
        ]
    }

    fn complete(&self, request: &AiRequest) -> AiResult<AiResponse> {
        let url = format!("{}/api/chat", self.base_url);
        let body = serde_json::json!({"model": request.config.model, "messages": request.messages.iter().map(|m| serde_json::json!({"role": m.role, "content": m.content})).collect::<Vec<_>>(), "stream": false, "options": {"temperature": request.config.temperature.unwrap_or(0.7), "num_predict": request.config.max_tokens.unwrap_or(4096)}});
        let response = self
            .client
            .post(&url)
            .set("content-type", "application/json")
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .send_json(body);
        match response {
            Ok(resp) => self.parse_response(resp),
            Err(ureq::Error::Status(status, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                Err(self.map_error(status, &body))
            }
            Err(ureq::Error::Transport(e)) => {
                Err(AiError::provider_unavailable("ollama", e.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_provider_name() {
        let provider = OllamaProvider::new();
        assert_eq!(provider.name(), "ollama");
    }

    #[test]
    fn test_ollama_default_model() {
        let provider = OllamaProvider::new();
        assert_eq!(provider.default_model(), "llama3.1");
    }

    #[test]
    fn test_ollama_list_models_fallback() {
        let provider = OllamaProvider::new();
        let models = provider.list_models();
        assert!(models.len() >= 5);
        assert!(models.iter().any(|m| m.contains("llama")));
    }

    #[test]
    fn test_ollama_with_custom_model() {
        let provider = OllamaProvider::new().with_default_model("mistral");
        assert_eq!(provider.default_model(), "mistral");
    }
}
