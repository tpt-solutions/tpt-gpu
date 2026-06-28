mod provider;
pub use provider::{AiProvider, AiError, ClaudeProvider, OpenRouterProvider, OllamaProvider};

/// Select a provider from environment variables.
///
/// Priority: ANTHROPIC_API_KEY → OPENROUTER_API_KEY → Ollama (local, no key needed).
pub fn provider_from_env() -> Box<dyn AiProvider> {
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        Box::new(ClaudeProvider::new(key))
    } else if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
        Box::new(OpenRouterProvider::new(key))
    } else {
        Box::new(OllamaProvider::default())
    }
}
