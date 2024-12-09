use reqwest::Client;
use serde_json::json;
use typed_builder::TypedBuilder;

#[derive(TypedBuilder)]
pub struct LLMRequest {
    messages: Vec<(String, String)>,
    #[builder(default = 200)]
    max_tokens: u32,
    #[builder(default = 1.0)]
    temperature: f32,
    api_url: String,
    api_key: String,
}

impl LLMRequest {
    pub async fn send(&self) -> anyhow::Result<String> {
        let json_request = self.get_json_request();
        let client = Client::new();
        let response = client
            .post(self.api_url.clone())
            .header("Content-Type", "application/json")
            .header("api-key", self.api_key.clone())
            .body(json_request.to_string())
            .send()
            .await?;

        let resp_text = response.text().await?;
        Ok(self.extract_response(resp_text))
    }

    fn extract_response(&self, response: String) -> String {
        let resp_json: serde_json::Value = serde_json::from_str(&response).unwrap();
        resp_json["choices"][0]["message"]["content"].to_string()
    }

    fn get_json_request(&self) -> serde_json::Value {
        let json_messages: Vec<serde_json::Value> = self
            .messages
            .iter()
            .map(|(role, content)| {
                json!({
                    "role": role,
                "content": content
                })
            })
            .collect();

        json!({
            "messages": json_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_json_request() {
        let messages = vec![("user".to_string(), "Hello, how are you?".to_string())];
        let request = LLMRequest::builder()
            .messages(messages)
            .api_key("api_key".to_string())
            .api_url("api_url".to_string())
            .build();
        let json_request = request.get_json_request();
        assert_eq!(json_request["messages"][0]["role"], "user");
        assert_eq!(
            json_request["messages"][0]["content"],
            "Hello, how are you?"
        );
        assert_eq!(json_request["max_tokens"], 200);
        assert_eq!(json_request["temperature"], 1.0);
        println!("{:?}", json_request.to_string());
    }

    #[tokio::test]
    async fn test_send() {
        // get api key from env
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
        let api_url = std::env::var("OPENAI_API_URL").expect("OPENAI_API_URL not set");
        let messages = vec![(
            "user".to_string(),
            "whats the capital of france?".to_string(),
        )];
        let request = LLMRequest::builder()
            .messages(messages)
            .api_key(api_key)
            .api_url(api_url)
            .build();
        let response = request.send().await.expect("Failed to send request");
        println!(" \n -- LLM response -- \n{} \n ---- \n", response);
        assert!(response.to_lowercase().contains("paris"));
    }
}
