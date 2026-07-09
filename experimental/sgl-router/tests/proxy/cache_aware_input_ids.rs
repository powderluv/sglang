// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end at the HTTP layer: the router tokenizes the prompt once at
//! ingress and forwards the ids to the engine as `input_ids` (so the engine
//! skips re-tokenizing the same prompt). Asserts the gating contract through
//! the real chat handler + a MockWorker backend:
//!
//! * A plain text chat request on the engine-equivalent chat-encoder path →
//!   the forwarded body carries `input_ids` AND retains `messages`.
//! * A request carrying `tools` → `input_ids` omitted (the router's encoder
//!   doesn't render tool schemas, so its ids would diverge from the engine).
//! * A request with multimodal (array) content → `input_ids` omitted (a text
//!   tokenizer can't represent image content).
//!
//! The model id contains `deepseek-v4` so the tokenizer registry auto-attaches
//! the built-in V4 chat encoder — the engine-equivalent path — without a
//! template fixture.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::config::{
    ActiveLoadConfig, CacheAwareConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::engine_load::EngineLoadTable;
use sgl_router::policies::factory::build_registry;
use sgl_router::policies::kv_events::{BlockSizeOracle, HashTree};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

const MODEL: &str = "deepseek-v4-tiny";

fn config() -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: MODEL.into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            tokenizer_backend: Default::default(),
            tokenizer_l1_cache_mb: 0,
            policy: PolicyKind::CacheAwareZmq,
            circuit_breaker: None,
            cache_aware: Some(CacheAwareConfig::default()),
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        admission: sgl_router::config::AdmissionConfig::default(),
        retry: sgl_router::config::RetryConfig::default(),
    }
}

fn build_ctx(url: String) -> Arc<AppContext> {
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    assert!(
        tokenizers.has_chat_encoder(MODEL),
        "deepseek-v4 model id must auto-attach the built-in chat encoder"
    );
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId(url.clone()),
        url,
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        bootstrap_port: None,
    });
    // Use the real loaded tokenizers (not the empty-registry test default) so
    // the cache-aware policy can tokenize at ingress.
    let policies = Arc::new(
        build_registry(
            &cfg,
            Arc::new(HashTree::new()),
            Arc::clone(&tokenizers),
            BlockSizeOracle::new(),
            EngineLoadTable::new(),
        )
        .unwrap(),
    );
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

async fn send(ctx: Arc<AppContext>, body: Value) -> StatusCode {
    let app = build_router(ctx);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    app.oneshot(req).await.unwrap().status()
}

fn captured(mock: &MockWorker) -> Value {
    let b = mock
        .captured
        .lock()
        .unwrap()
        .last_body
        .clone()
        .expect("worker captured a request body");
    serde_json::from_slice(&b).expect("captured body is valid JSON")
}

#[tokio::test]
async fn plain_chat_forwards_input_ids_and_keeps_messages() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello there friend"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock);
    let ids = body.get("input_ids").and_then(|v| v.as_array());
    assert!(
        ids.is_some_and(|a| !a.is_empty()),
        "engine must receive non-empty input_ids; got {body}"
    );
    assert!(
        body.get("messages").is_some(),
        "messages must be retained alongside input_ids; got {body}"
    );
}

#[tokio::test]
async fn tool_request_omits_input_ids() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock);
    assert!(
        body.get("input_ids").is_none(),
        "tool requests must not forward input_ids; got {body}"
    );
}

#[tokio::test]
async fn thinking_request_omits_input_ids() {
    // `chat_template_kwargs` steers engine-side thinking mode, which the
    // router's encoder renders in the default mode only — forwarding ids would
    // silently run the wrong mode, so the handler must omit them.
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": true},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock);
    assert!(
        body.get("input_ids").is_none(),
        "thinking-mode requests must not forward input_ids; got {body}"
    );
}

#[tokio::test]
async fn multimodal_request_omits_input_ids() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock);
    assert!(
        body.get("input_ids").is_none(),
        "multimodal requests must not forward input_ids; got {body}"
    );
}

/// The fail-fast context-length gate, end to end. With no worker having
/// disclosed a `max_req_input_len` (introspection pending / older SGLang)
/// the gate is disabled and requests flow to the engine — the pre-gate
/// behavior. Once the fleet's bound is known and an ingress-tokenized
/// input exceeds it, the router rejects with OpenAI's exact
/// `context_length_exceeded` contract WITHOUT dispatching: the worker
/// sees no request, and `sgl_router_context_length_rejected_total` counts
/// it.
#[tokio::test]
async fn over_context_length_rejected_at_ingress_without_dispatch() {
    use http_body_util::BodyExt;

    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello there friend"}],
    });

    // No disclosed bound ⇒ gate disabled ⇒ the request reaches the worker.
    let status = send(ctx.clone(), body.clone()).await;
    assert_eq!(status, StatusCode::OK, "gate must be inert without a bound");
    assert!(
        mock.captured.lock().unwrap().last_body.is_some(),
        "boundless fleet must dispatch to the worker"
    );
    mock.captured.lock().unwrap().last_body = None;

    // Disclose a 1-token bound (as introspection would) — the same request
    // now tokenizes far past it and must be rejected at ingress.
    ctx.registry
        .get(&WorkerId(mock.url.clone()))
        .expect("worker registered")
        .set_max_req_input_len(1);

    let app = build_router(Arc::clone(&ctx));
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = BodyExt::collect(resp.into_body()).await.unwrap().to_bytes();
    let env: Value = serde_json::from_slice(&bytes).expect("error envelope is JSON");
    assert_eq!(env["error"]["code"], "context_length_exceeded");
    assert_eq!(env["error"]["type"], "invalid_request_error");
    assert_eq!(env["error"]["param"], "messages");
    assert!(
        env["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("maximum context length is 1 tokens"),
        "message must carry the real limit; got {env}"
    );

    assert!(
        mock.captured.lock().unwrap().last_body.is_none(),
        "an ingress-rejected request must never reach the worker"
    );
    let metrics = ctx.metrics.render();
    assert!(
        metrics.contains(&format!(
            "sgl_router_context_length_rejected_total{{model_id=\"{MODEL}\"}} 1"
        )),
        "rejection must bump the counter; got:\n{metrics}"
    );
}

/// The slack band, end to end. A model WITHOUT a chat encoder under the
/// cache-aware policy takes the raw-prompt fallback (`engine_equivalent =
/// false`), whose count may drift from the engine's render — so the gate
/// must not fire inside the max(1024, limit/64) band above the bound
/// (false-reject guard), and must still fire beyond it.
#[tokio::test]
async fn slack_band_admits_near_limit_raw_fallback_requests() {
    const RAW_MODEL: &str = "tiny-raw"; // no "deepseek..v4" -> no chat encoder
    let mock = MockWorker::start(vec![]).await;
    let mut cfg = config();
    cfg.model.id = RAW_MODEL.into();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    assert!(
        !tokenizers.has_chat_encoder(RAW_MODEL),
        "test premise: raw model must have no chat encoder"
    );
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId(mock.url.clone()),
        url: mock.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(RAW_MODEL.into())],
        bootstrap_port: None,
    });
    let policies = Arc::new(
        build_registry(
            &cfg,
            Arc::new(HashTree::new()),
            Arc::clone(&tokenizers),
            BlockSizeOracle::new(),
            EngineLoadTable::new(),
        )
        .unwrap(),
    );
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let ctx = Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies));
    ctx.registry
        .get(&WorkerId(mock.url.clone()))
        .expect("worker registered")
        .set_max_req_input_len(1);

    // A short prompt is over the 1-token bound but inside the 1024-token
    // slack band: it must DISPATCH (the engine is the authority here).
    let status = send(
        ctx.clone(),
        json!({
            "model": RAW_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "raw-fallback request inside the slack band must not be rejected"
    );
    assert!(mock.captured.lock().unwrap().last_body.is_some());
    mock.captured.lock().unwrap().last_body = None;

    // Beyond bound + slack (1 + 1024 tokens; the byte-level fixture makes
    // one token per byte) the gate fires.
    let status = send(
        ctx,
        json!({
            "model": RAW_MODEL,
            "messages": [{"role": "user", "content": "x".repeat(1200)}],
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "raw-fallback request beyond the slack band must be rejected"
    );
    assert!(
        mock.captured.lock().unwrap().last_body.is_none(),
        "rejected request must not reach the worker"
    );
}

/// Boundary parity with the engine: `validate_input_length` rejects at
/// `input >= max_req_input_len`, so the gate must reject at EXACT
/// equality and admit at limit+1 (engine-equivalent path, zero slack).
#[tokio::test]
async fn gate_rejects_at_exact_limit_and_admits_one_above() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let messages = json!([{"role": "user", "content": "hello there friend"}]);
    let n = ctx
        .tokenizers
        .encode_chat(MODEL, &messages)
        .expect("chat encoder tokenizes the request")
        .len() as u64;
    let worker = ctx
        .registry
        .get(&WorkerId(mock.url.clone()))
        .expect("worker registered");
    let body = json!({"model": MODEL, "messages": messages});

    // limit = N + 1 -> input (N) is under the bound -> dispatches.
    worker.set_max_req_input_len(n + 1);
    let status = send(ctx.clone(), body.clone()).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "input below the bound must dispatch"
    );

    // limit = N -> input == limit -> engine would reject (>=), so must we.
    worker.set_max_req_input_len(n);
    let status = send(ctx, body).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "input exactly at the bound must be rejected (engine rejects at >=)"
    );
}
