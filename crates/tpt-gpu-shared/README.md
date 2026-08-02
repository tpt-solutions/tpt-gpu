# TPT AI � Multi-Provider AI Abstraction

A unified Rust library for LLM inference across multiple backends: **Claude** (Anthropic), **OpenRouter**, and **Ollama**.

## Features

- ?? **Single `AiProvider` trait** � Switch providers with zero code changes
- ?? **Three providers** � Claude, OpenRouter (100+ models), and local Ollama
- ?? **Factory pattern** � Create providers from config or environment variables
- ?? **Type-safe requests/responses** � Structured messages, model configs, and usage stats
- ?? **Multi-turn conversations** � Support for system prompts and conversation history
- ?? **Token tracking** � Automatic usage counting and reporting
- ? **Async-ready** � Providers are `Send + Sync` for multi-threaded contexts
- ??? **Error handling** � Comprehensive error types with retry guidance

## Quick Start

### Add to Cargo.toml

```toml
[dependencies]
tpt-ai = { path = "../shared" }
```

### Basic Usage

```rust
use tpt_ai::{AiProvider, AiRequest, ClaudeProvider, ProviderFactory};

// Direct provider creation
let provider = ClaudeProvider::new("sk-ant-...")?;

// Or use the factory
let provider = ProviderFactory::create("claude", Some("sk-ant-..."))?;

// Or auto-detect from environment variables
let provider = ProviderFactory::from_env()?;

// Build a request
let request = AiRequest::with_system(
    "claude-sonnet-4-20250514",
    "You are a GPU kernel generator.",
    "Generate a GEMM kernel",
)
.with_max_tokens(4096)
.with_temperature(0.7);

// Get response
let response = provider.complete(&request)?;
println!("{}", response.text().unwrap_or(""));
```

## Providers

### Claude (Anthropic)

Uses the Anthropic Messages API. Requires an API key.

```rust
use tpt_ai::ClaudeProvider;

// From API key
let provider = ClaudeProvider::new("sk-ant-...");

// From environment variable (ANTHROPIC_API_KEY)
let provider = ClaudeProvider::from_env()?;

// Custom model
let provider = ClaudeProvider::new("sk-ant-...")
    .with_default_model("claude-opus-4-20250514");
```

**Environment Variables:**
- `ANTHROPIC_API_KEY` � Your Anthropic API key

**Default Model:** `claude-sonnet-4-20250514`

### OpenRouter

Aggregates 100+ models from multiple providers. Requires an API key.

```rust
use tpt_ai::OpenRouterProvider;

let provider = OpenRouterProvider::new("sk-or-...");
let provider = OpenRouterProvider::from_env()?;
```

**Environment Variables:**
- `OPENROUTER_API_KEY` � Your OpenRouter API key

**Default Model:** `google/gemini-2.0-flash-001`

### Ollama (Local)

Uses a local Ollama server. No API key required.

```rust
use tpt_ai::OllamaProvider;

let provider = OllamaProvider::new();
let provider = OllamaProvider::new()
    .with_base_url("http://localhost:11434")
    .with_default_model("llama3.1");
```

**Default URL:** `http://localhost:11434`

**Default Model:** `llama3.1`

## Factory Pattern

The `ProviderFactory` makes it easy to create providers dynamically:

```rust
use tpt_ai::ProviderFactory;

// Create by name
let provider = ProviderFactory::create("claude", Some("sk-ant-..."))?;

// Auto-detect from environment
let provider = ProviderFactory::from_env()?;

// Type-specific constructors
let claude = ProviderFactory::claude("sk-ant-...");
let openrouter = ProviderFactory::openrouter("sk-or-...");
let ollama = ProviderFactory::ollama();
```

## Request Building

```rust
use tpt_ai::{AiRequest, AiMessage, Role};

// Simple request
let request = AiRequest::new("claude-sonnet-4-20250514", "Hello!");

// With system prompt
let request = AiRequest::with_system(
    "claude-sonnet-4-20250514",
    "You are a helpful assistant.",
    "What is Rust?",
);

// Multi-turn conversation
let request = AiRequest::new("claude-sonnet-4-20250514", "What is GEMM?")
    .add_message(AiMessage::assistant("GEMM is..."))
    .add_message(AiMessage::user("How do I implement it?"));

// With configuration
let request = AiRequest::new("model", "prompt")
    .with_max_tokens(4096)
    .with_temperature(0.7)
    .with_json_format();
```

## Response Handling

```rust
let response = provider.complete(&request)?;

// Get text content
println!("{}", response.text().unwrap_or("No content"));

// Get token usage
println!("Tokens used: {}", response.total_tokens());

// Check finish reason
if let Some(reason) = response.finish_reason() {
    println!("Stopped because: {:?}", reason);
}

// Access all choices
for choice in &response.choices {
    println!("Choice {}: {}", choice.index, choice.message.content);
}
```

## Error Handling

```rust
use tpt_ai::{AiError, AiResult};

match provider.complete(&request) {
    Ok(response) => println!("{}", response.text().unwrap_or("")),
    Err(AiError::Authentication { message }) => {
        eprintln!("Auth error: {}", message);
    }
    Err(AiError::RateLimited { message, retry_after_secs }) => {
        eprintln!("Rate limited: {} (retry after {:?}s)", message, retry_after_secs);
    }
    Err(AiError::ProviderUnavailable { provider, message }) => {
        eprintln!("Provider {} unavailable: {}", provider, message);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Advanced Features

### Checking Provider Availability

```rust
let provider = ClaudeProvider::from_env()?;

if provider.is_available() {
    println!("Claude is ready!");
}

// List available models
let models = provider.list_models();
println!("Available models: {:?}", models);
```

### Custom Model Configuration

```rust
use tpt_ai::ModelConfig;

let config = ModelConfig::new("claude-sonnet-4-20250514")
    .with_max_tokens(8192)
    .with_temperature(0.5)
    .with_top_p(0.9)
    .with_json_format();

let response = provider.complete_with_config(&request, &config)?;
```

### Simple Single-Shot Completion

```rust
// Quick one-liner for simple use cases
let text = provider.ask(
    "You are a GPU kernel expert.",
    "Explain tiling in GEMM kernels.",
)?;
println!("{}", text);
```

## Use Cases for TPT GPU

This library is designed to support GPU kernel generation workflows:

1. **Kernel Generation** � Generate optimized CUDA/ROCm/Metal kernels
2. **Performance Hints** � Get optimization suggestions for specific hardware
3. **Natural Language Queries** � Ask questions about GPU programming concepts
4. **Multi-Provider Fallback** � Switch between providers based on availability

## Architecture

```
+---------------------------------------------+
�              Your Application               �
+---------------------------------------------+
                     �
                     ?
+---------------------------------------------+
�          AiProvider Trait (unified)         �
+---------------------------------------------+
                     �
        +------------+------------+
        ?            ?            ?
   +---------+ +----------+ +---------+
   � Claude  � �OpenRouter� � Ollama  �
   �Provider � � Provider � �Provider �
   +---------+ +----------+ +---------+
```

## Testing

```bash
# Run all tests
cargo test -p tpt-gpu-shared

# Run with output
cargo test -p tpt-gpu-shared -- --nocapture
```

## License

Apache 2.0 (with Express Patent Grant)