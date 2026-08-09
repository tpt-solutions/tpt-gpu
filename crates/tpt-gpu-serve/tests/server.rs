//! End-to-end test: spawn the real `tpt-serve` binary, load a minimal GGUF
//! model, and exercise the OpenAI-compatible `/v1/completions` endpoint.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command};
use std::time::Duration;

/// Write a minimal valid GGUF v2 file (mirrors the runtime test helper).
fn write_minimal_gguf(path: &std::path::Path) {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    // kv_count
    let kv: &[(&str, u32, &str)] = &[
        ("general.architecture", 0, "llama3"),
        ("llm.context_length", 4, ""),
        ("llm.embedding_length", 4, ""),
        ("llm.attention.head_count", 4, ""),
        ("llm.attention.head_count_kv", 4, ""),
        ("llm.feed_forward_length", 4, ""),
        ("llm.block_count", 4, ""),
        ("llm.vocab_size", 4, ""),
    ];
    buf.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (key, ty, val) in kv {
        let kb = key.as_bytes();
        buf.extend_from_slice(&(kb.len() as u64).to_le_bytes());
        buf.extend_from_slice(kb);
        match *ty {
            4 => {
                buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
                let v = match *key {
                    "llm.context_length" => 64u32,
                    "llm.embedding_length" => 64,
                    "llm.attention.head_count" => 4,
                    "llm.attention.head_count_kv" => 2,
                    "llm.feed_forward_length" => 128,
                    "llm.block_count" => 2,
                    "llm.vocab_size" => 32000,
                    _ => 0,
                };
                buf.extend_from_slice(&v.to_le_bytes());
            }
            _ => {
                buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
                let sb = val.as_bytes();
                buf.extend_from_slice(&(sb.len() as u64).to_le_bytes());
                buf.extend_from_slice(sb);
            }
        }
    }
    std::fs::write(path, &buf).unwrap();
}

struct ServerProcess {
    child: Child,
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

fn wait_for_server(addr: &str) -> TcpStream {
    for _ in 0..100 {
        if let Ok(s) = TcpStream::connect(addr) {
            return s;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("tpt-serve did not start listening on {addr}");
}

fn http_request(method: &str, addr: &str, path: &str, body: &str) -> String {
    let mut stream = wait_for_server(addr);
    let req = if body.is_empty() {
        format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
    } else {
        format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
    };
    stream.write_all(req.as_bytes()).unwrap();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).unwrap();
    match resp.find("\r\n\r\n") {
        Some(p) => resp[p + 4..].to_string(),
        None => resp,
    }
}

fn http_post(addr: &str, path: &str, body: &str) -> String {
    http_request("POST", addr, path, body)
}

fn http_get(addr: &str, path: &str) -> String {
    http_request("GET", addr, path, "")
}

#[test]
fn serve_completions_returns_openai_shape() {
    let dir = std::env::temp_dir();
    let model = dir.join(format!("tpt_serve_test_{}.gguf", std::process::id()));
    write_minimal_gguf(&model);

    let port = free_port();
    let addr = format!("127.0.0.1:{port}");

    let child = Command::new(env!("CARGO_BIN_EXE_tpt-serve"))
        .args(["--model", &model.to_string_lossy(), "--host", "127.0.0.1", "--port", &port.to_string()])
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn tpt-serve");
    let _server = ServerProcess { child };

    // /v1/models
    let models = http_get(&addr, "/v1/models");
    let models_json: serde_json::Value = serde_json::from_str(&models).unwrap();
    assert_eq!(models_json["object"], "list");
    assert_eq!(models_json["data"][0]["id"], model.to_string_lossy().to_string());

    // /v1/completions with explicit token ids
    let body = r#"{"model":"x","prompt_tokens":[1,2,3],"max_tokens":4}"#;
    let resp = http_post(&addr, "/v1/completions", body);
    let json: serde_json::Value = serde_json::from_str(&resp).unwrap();

    assert_eq!(json["object"], "text_completion");
    assert!(json["choices"].is_array());
    let choice = &json["choices"][0];
    assert!(choice["text"].is_string(), "expected a text field: {resp}");
    assert_eq!(choice["finish_reason"], "length");
    assert!(json["usage"]["completion_tokens"].as_u64().unwrap_or(0) <= 4);

    // /v1/chat/completions
    let chat_body = r#"{"model":"x","messages":[{"role":"user","content":"hello"}],"max_tokens":3}"#;
    let chat = http_post(&addr, "/v1/chat/completions", chat_body);
    let chat_json: serde_json::Value = serde_json::from_str(&chat).unwrap();
    assert_eq!(chat_json["object"], "chat.completion");
    assert_eq!(chat_json["choices"][0]["message"]["role"], "assistant");

    let _ = std::fs::remove_file(&model);
}
