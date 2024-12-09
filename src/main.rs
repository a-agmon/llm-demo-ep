use axum::{response::IntoResponse, routing::post, Json, Router};
use llm_utils::LLMRequest;
use serde::Deserialize;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use vecdb::VecDB;
mod embedder;
mod llm_utils;
mod vecdb;
mod vectors;

use axum::http::StatusCode;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // initialize tracing
    tracing_subscriber::fmt::init();
    info!("Starting server...");

    // Add CORS middleware
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/generate", post(generate_response))
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();

    Ok(())
}

#[axum::debug_handler]
async fn generate_response(body: String) -> impl IntoResponse {
    //let messages = vec![("user".to_string(), body)];
    let messages = prepare_query(body).await.unwrap();
    let response = generate_response_llm(messages).await;
    (StatusCode::OK, response)
}

async fn query_llm(body_str: String) -> (StatusCode, String) {
    // let body = request.into_body();
    // let bytes = axum::body::to_bytes(body, 2048usize).await.unwrap();
    // let body_str = String::from_utf8(bytes.to_vec()).unwrap();
    let messages = prepare_query(body_str).await.unwrap();
    let response = generate_response_llm(messages).await;
    (axum::http::StatusCode::OK, response)
}

#[derive(Debug, serde::Deserialize)]
pub struct TableContent {
    pub content: String,
}
async fn prepare_query(query: String) -> anyhow::Result<Vec<(String, String)>> {
    let embedding_model = embedder::get_model_reference().unwrap();
    let prompt_embedding = embedding_model.embed_multiple(vec![query.clone()]).unwrap();
    let prompt_embedding = prompt_embedding[0].clone();
    let prompt_embedding_norm = vectors::VectorsOps::normalize(&prompt_embedding);
    // search relevant tables
    let vecdb = VecDB::create_or_open(
        "/home/alonagmon/rust/schema-pilot/runtime_assets/vecdb",
        "hc100",
        Some(384),
    )
    .await?;
    let tables = vecdb.find_similar_x(prompt_embedding_norm, 10).await?;
    let table_contents: Vec<TableContent> = serde_arrow::from_record_batch(&tables)?;
    let context_str = table_contents
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
    Your answer always includes information about the relevant tables, columns and their purpose. 
    You can also add a query to the answer if the user asked for a query.
    "#
    .to_string();

    let user = format!(
        r#"
        Here are the tables in our database:
        {context}

        Based on the tables in our database given above, please answer the following question concisely and directly:
        {query}
        "#
    );
    vec![(String::from("system"), sys), (String::from("user"), user)]
}
