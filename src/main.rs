use std::sync::Arc;

use axum::{
    extract::rejection::LengthLimitError,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use llm_utils::LLMRequest;
use once_cell::sync::{Lazy, OnceCell};
use serde::Deserialize;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, Level};
use vecdb::VecDB;
mod embedder;
mod llm_utils;
mod vecdb;
mod vectors;

use axum::http::StatusCode;
static VECDB: OnceCell<Arc<RwLock<VecDB>>> = OnceCell::new();
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    info!("Starting server...");
    let vecdb = init_vecdb(
        "/home/alonagmon/rust/schema-pilot/runtime_assets/vecdb/",
        "retail_dm",
    )
    .await?;
    if let Err(_) = VECDB.set(vecdb) {
        info!("vec db value was set");
    }

    // Add CORS middleware
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/generate", post(generate_response))
        .route("/test", get(test_service))
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();

    Ok(())
}

fn get_vecdb() -> &'static Arc<RwLock<VecDB>> {
    VECDB.get().expect("failed to get vecdb state")
}

async fn init_vecdb(path: &str, table_name: &str) -> anyhow::Result<Arc<RwLock<VecDB>>> {
    info!("initializing vecdb on path: {path} with name: {table_name}");
    let vec_db = VecDB::create_or_open(path, table_name, Some(384)).await?;
    let vec_db = Arc::new(RwLock::new(vec_db));
    Ok(vec_db)
}

async fn test_service() -> impl IntoResponse {
    let vecdb = get_vecdb();
    let vecdb = vecdb.read().await;
    let embedding_model = embedder::get_model_reference().unwrap();
    let prompt_embedding = embedding_model
        .embed_multiple(vec!["hello".to_string()])
        .unwrap();
    let prompt_embedding = prompt_embedding[0].clone();
    let prompt_embedding_norm = vectors::VectorsOps::normalize(&prompt_embedding);
    let tables = vecdb.find_similar_x(prompt_embedding_norm, 10).await;
    match tables {
        Ok(_) => (StatusCode::OK, String::from("all good!")),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[axum::debug_handler]
async fn generate_response(body: String) -> impl IntoResponse {
    info!("recieved query: {body}");
    let response = process_query(body).await;
    match response {
        Ok(response) => (StatusCode::OK, response),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn process_query(query: String) -> anyhow::Result<String> {
    let embedding = embed_and_normalize_query(query.clone())?;
    tracing::info!("embedding generated");
    let context_tables = get_relevant_context_tables(embedding, 20).await?;
    tracing::info!("context tables generated");
    let prompt_msgs = generate_prompt_msgs(query, context_tables).await?;
    tracing::info!("prompt messages generated");
    let response = generate_response_llm(prompt_msgs).await;
    tracing::info!("response generated");
    Ok(response)
}

#[derive(Debug, serde::Deserialize)]
pub struct TableContent {
    pub content: String,
}

fn embed_and_normalize_query(query: String) -> anyhow::Result<Vec<f32>> {
    let embedding_model = embedder::get_model_reference()?;
    let prompt_embedding = embedding_model.embed_multiple(vec![query.clone()])?;
    let prompt_embedding = prompt_embedding[0].clone();
    let prompt_embedding_norm = vectors::VectorsOps::normalize(&prompt_embedding);
    Ok(prompt_embedding_norm)
}

async fn get_relevant_context_tables(
    query_vec: Vec<f32>,
    num_tables: usize,
) -> anyhow::Result<Vec<TableContent>> {
    let vecdb = get_vecdb();
    let vecdb = vecdb.read().await;
    let tables = vecdb.find_similar_x(query_vec, num_tables).await?;
    let table_contents: Vec<TableContent> = serde_arrow::from_record_batch(&tables)?;
    Ok(table_contents)
}

async fn generate_prompt_msgs(
    query: String,
    context_tables: Vec<TableContent>,
) -> anyhow::Result<Vec<(String, String)>> {
    let context_str = context_tables
        .iter()
        .map(|t| t.content.clone())
        .collect::<Vec<String>>()
        .join("\n");
    let prompt = create_prompt(context_str, query);
    Ok(prompt)
}

async fn generate_response_llm(messages: Vec<(String, String)>) -> String {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let api_url = std::env::var("OPENAI_API_URL").expect("OPENAI_API_URL not set");
    let request = LLMRequest::builder()
        .messages(messages)
        .api_key(api_key)
        .api_url(api_url)
        .max_tokens(800)
        .temperature(0.5)
        .build();
    let response = request.send().await.expect("Failed to send request");
    response
}

fn create_prompt(context: String, query: String) -> Vec<(String, String)> {
    let sys = r#" 
    You are an AI assistant that answers questions about database schemas and tables. 
    Your answer always includes information about the relevant tables and their purpose. 
    When you add a query to your answer, always mark it with ```sql.
    Always use numbers when enumerating items.
    "#
    .to_string();

    let user = format!(
        r#"
        Here are the tables in our database:
        {context}

        Based on the tables in our database given above, please answer the following question:
        {query}
        "#
    );
    vec![(String::from("system"), sys), (String::from("user"), user)]
}
