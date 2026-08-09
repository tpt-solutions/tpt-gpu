//! OpenAI-compatible HTTP server for TPT GPU LLM inference.
//!
//! Implements a small, dependency-free HTTP/1.1 server (built on `std::net`)
//! exposing the subset of the OpenAI REST API needed to drive the runtime from
//! standard OpenAI client libraries:
//!
//! - `GET  /v1/models`
//! - `POST /v1/completions`
//! - `POST /v1/chat/completions`
//!
//! Both streaming (`stream: true`, Server-Sent Events) and non-streaming
//! responses are supported. Tokenization uses the placeholder [`WordTokenizer`]
//! until a real model vocabulary is wired in (see `tokenizer.rs`).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use tpt_gpu_runtime::{GpuInferenceEngine, LlmInference};

use crate::tokenizer::WordTokenizer;

/// Bind the server and handle connections until the process exits.
pub fn run(
    state: Arc<Mutex<GpuInferenceEngine>>,
    model_name: &str,
    host: &str,
    port: u16,
) -> std::io::Result<()> {
    let listener = TcpListener::bind((host, port))?;
    eprintln!("tpt-serve: listening on http://{host}:{port}");
    for s in listener.incoming().flatten() {
        let st = Arc::clone(&state);
        let m = model_name.to_string();
        std::thread::spawn(move || {
            if let Err(e) = handle_connection(s, &st, &m) {
                eprintln!("tpt-serve: connection error: {e}");
            }
        });
    }
    Ok(())
}

fn handle_connection(
    mut stream: TcpStream,
    state: &Arc<Mutex<GpuInferenceEngine>>,
    model_name: &str,
) -> std::io::Result<()> {
    let mut buf = [0u8; 8192];
    let mut header = Vec::new();
    loop {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        header.extend_from_slice(&buf[..n]);
        if header.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if header.len() > 4_000_000 {
            break;
        }
    }

    let header_text = String::from_utf8_lossy(&header).into_owned();
    let sep = header_text.find("\r\n\r\n").unwrap_or(header_text.len());
    let head = &header_text[..sep];
    let mut body = header.split_off(sep + 4);

    let content_length = head
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|s| s.trim().parse::<usize>().unwrap_or(0))
        })
        .unwrap_or(0);
    while body.len() < content_length {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&buf[..n]);
    }
    body.truncate(content_length);

    let request_line = head.lines().next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");

    match (method, path) {
        ("GET", "/v1/models") => respond_models(&mut stream, model_name),
        ("POST", "/v1/completions") => handle_completion(&mut stream, state, model_name, &body, false),
        ("POST", "/v1/chat/completions") => {
            handle_completion(&mut stream, state, model_name, &body, true)
        }
        _ => respond_404(&mut stream),
    }
}

fn handle_completion(
    stream: &mut TcpStream,
    state: &Arc<Mutex<GpuInferenceEngine>>,
    model_name: &str,
    body: &[u8],
    chat: bool,
) -> std::io::Result<()> {
    let req: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    let max_tokens = req
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(32)
        .min(2048) as u32;
    let stream_resp = req
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let model = req
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(model_name)
        .to_string();

    let vocab = state.lock().unwrap().model_info().vocab_size;
    let mut tok = WordTokenizer::new(vocab);

    // Resolve the prompt to token ids.
    let prompt_tokens: Vec<u32> = if let Some(arr) = req.get("prompt_tokens").and_then(|v| v.as_array())
    {
        arr.iter()
            .filter_map(|x| x.as_u64().map(|n| n as u32))
            .collect()
    } else {
        let text = if chat {
            req.get("messages")
                .and_then(|m| m.as_array())
                .map(|msgs| {
                    msgs.iter()
                        .map(|m| {
                            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
                            let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
                            format!("{role}: {content}\n")
                        })
                        .collect::<String>()
                })
                .unwrap_or_default()
        } else {
            match req.get("prompt") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(a)) => a
                    .iter()
                    .filter_map(|x| x.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => String::new(),
            }
        };
        if text.trim().is_empty() {
            vec![0]
        } else {
            tok.encode(&text)
        }
    };

    let created = now_secs();
    let id_prefix = if chat { "chatcmpl" } else { "cmpl" };
    let id = format!("{id_prefix}-{created}");

    let mut generated: Vec<u32> = Vec::new();

    if stream_resp {
        write_sse_headers(stream)?;
    }

    let mut engine = state.lock().unwrap();
    let infer_result = engine.infer(&prompt_tokens, max_tokens, |tok_id| {
        generated.push(tok_id);
        if stream_resp {
            let piece = tok.decode(&[tok_id]);
            let delta = if chat {
                json!({ "role": "assistant", "content": piece })
            } else {
                json!({ "content": piece })
            };
            let chunk = json!({
                "id": id,
                "object": if chat { "chat.completion.chunk" } else { "text_completion.chunk" },
                "created": created,
                "model": model,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": null }]
            });
            let _ = write_sse_chunk(stream, &chunk);
        }
    });

    if let Err(e) = infer_result {
        if stream_resp {
            let _ = write_sse_chunk(stream, &json!({ "error": e.to_string() }));
            let _ = write_sse_done(stream);
        } else {
            let err = json!({ "error": { "message": e.to_string(), "type": "runtime_error" } });
            return write_json(stream, &err, 500, false);
        }
        return Ok(());
    }

    let text_out = tok.decode(&generated);

    if stream_resp {
        let final_chunk = json!({
            "id": id,
            "object": if chat { "chat.completion.chunk" } else { "text_completion.chunk" },
            "created": created,
            "model": model,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "length" }]
        });
        let _ = write_sse_chunk(stream, &final_chunk);
        let _ = write_sse_done(stream);
        return Ok(());
    }

    let usage = json!({
        "prompt_tokens": prompt_tokens.len(),
        "completion_tokens": generated.len(),
        "total_tokens": prompt_tokens.len() + generated.len()
    });

    let resp = if chat {
        json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": text_out },
                "finish_reason": "length"
            }],
            "usage": usage
        })
    } else {
        json!({
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{
                "text": text_out,
                "index": 0,
                "finish_reason": "length"
            }],
            "usage": usage
        })
    };

    write_json(stream, &resp, 200, false)
}

fn respond_models(stream: &mut TcpStream, model_name: &str) -> std::io::Result<()> {
    let body = json!({
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "owned_by": "tpt-gpu",
            "permission": []
        }]
    });
    write_json(stream, &body, 200, false)
}

fn respond_404(stream: &mut TcpStream) -> std::io::Result<()> {
    let body = json!({ "error": { "message": "not found", "type": "invalid_request_error" } });
    write_json(stream, &body, 404, false)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    }
}

fn write_json(stream: &mut TcpStream, body: &Value, status: u16, _sse: bool) -> std::io::Result<()> {
    let payload = serde_json::to_vec(body)?;
    let header = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status_text(status),
        payload.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(&payload)?;
    stream.flush()
}

fn write_sse_headers(stream: &mut TcpStream) -> std::io::Result<()> {
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    stream.write_all(header.as_bytes())?;
    stream.flush()
}

fn write_sse_chunk(stream: &mut TcpStream, chunk: &Value) -> std::io::Result<()> {
    let payload = serde_json::to_vec(chunk)?;
    stream.write_all(b"data: ")?;
    stream.write_all(&payload)?;
    stream.write_all(b"\n\n")?;
    stream.flush()
}

fn write_sse_done(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b"data: [DONE]\n\n")?;
    stream.flush()
}
