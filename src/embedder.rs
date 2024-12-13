use std::rc::Rc;
use std::sync::Arc;

use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

use anyhow::anyhow;
use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

const EMB_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L12-v2";
const EMB_MODEL_REV: &str = "main";

thread_local! {
    static BERT_MODEL: Arc<EmbeddingModel> =   {
        tracing::info!("Loading a embedding model on thread: {:?}", std::thread::current().id());
        let model = EmbeddingModel::new(EMB_MODEL_ID.to_string(), EMB_MODEL_REV.to_string());
        match model {
            Ok(model) => Arc::new(model),
            Err(e) => {
                panic!("Failed to load the model: {}", e);
            }
        }
    }
}

pub fn get_model_reference() -> anyhow::Result<Arc<EmbeddingModel>> {
    BERT_MODEL.with(|model| Ok(model.clone()))
}
pub struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    pub fn new(model_id: String, revision: String) -> Result<Self> {
        let device = Device::Cpu;
        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("model.safetensors")?;

        let config = std::fs::read_to_string(config)?;
        let config: Config = serde_json::from_str(&config).map_err(E::msg)?;
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn embed_multiple(&self, sentences: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut tokenizer = self.tokenizer.clone();
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }
        let tokens = tokenizer.encode_batch(sentences, true).map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        //println!("running inference on batch {:?}", token_ids.shape());
        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
        //println!("generated embeddings {:?}", embeddings.shape());
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = embeddings.to_vec2::<f32>()?;
        Ok(embeddings)
    }
}
pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

pub fn apply_max_pooling(embeddings: &Tensor) -> anyhow::Result<Tensor> {
    Ok(embeddings.max(1)?)
}

// cosine similarity between two vectors
pub fn cosine_similarity(v1: &Vec<f32>, v2: &Vec<f32>) -> f32 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum::<f32>()
}

/// Apply mean pooling to the embeddings
/// The input tensor should either have the shape (n_sentences, n_tokens, hidden_size) or (n_tokens, hidden_size)
/// depending on whether the input is a batch of sentences or a single sentence
pub fn apply_mean_pooling(embeddings: &Tensor) -> anyhow::Result<Tensor> {
    match embeddings.rank() {
        3 => {
            let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
            (embeddings.sum(1)? / (n_tokens as f64)).map_err(anyhow::Error::msg)
        }
        2 => {
            let (n_tokens, _hidden_size) = embeddings.dims2()?;
            (embeddings.sum(0)? / (n_tokens as f64)).map_err(anyhow::Error::msg)
        }
        _ => anyhow::bail!("Unsupported tensor rank for mean pooling"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_multiple() {
        let model =
            EmbeddingModel::new(EMB_MODEL_ID.to_string(), EMB_MODEL_REV.to_string()).unwrap();
        let embeddings = model
            .embed_multiple(vec![
                "Hello, world!".to_string(),
                "This is a test".to_string(),
            ])
            .unwrap();
        println!("embeddings len: {}", embeddings.len());
        println!("embeddings shape: {:?}", embeddings[0].len());
    }

    #[test]
    fn test_embed_multiple2() {
        let model =
            EmbeddingModel::new(EMB_MODEL_ID.to_string(), EMB_MODEL_REV.to_string()).unwrap();
        let embeddings = model
            .embed_multiple(vec![
                "Hello, world!".to_string(),
                "This is a test".to_string(),
            ])
            .unwrap();
        println!("embeddings len: {}", embeddings.len());
        println!("embeddings shape: {:?}", embeddings[0].len());
        let embeddings = model
            .embed_multiple(vec![
                "I love dogs, lions, and cats and going to the zoo".to_string(),
                "The integral of x squared dx equals x cubed over three plus C".to_string(),
            ])
            .unwrap();
        println!("embeddings len: {}", embeddings.len());
        println!("embeddings shape: {:?}", embeddings[0].len());
    }

    // test cosine similarity
    #[test]
    fn test_cosine_similarity() {
        let model =
            EmbeddingModel::new(EMB_MODEL_ID.to_string(), EMB_MODEL_REV.to_string()).unwrap();
        let embeddings = model
            .embed_multiple(vec![
                "I love dogs, lions, and cats and going to the zoo".to_string(),
                "The integral of x squared dx equals x cubed over three plus C".to_string(),
            ])
            .unwrap();
        let animals_embedding = model
            .embed_multiple(vec!["a little dog ate my lunch with his birds".to_string()])
            .unwrap();
        // check that the cosine similarity between the first sentence and the animals embedding is higher than the cosine similarity between the second sentence and the animals embedding
        let similarity_first = cosine_similarity(&embeddings[0], &animals_embedding[0]);
        let similarity_second = cosine_similarity(&embeddings[1], &animals_embedding[0]);
        println!(
            "similarity_first: {}, similarity_second: {}",
            similarity_first, similarity_second
        );
        assert!(similarity_first > similarity_second);
    }

    // test embed 3 sentences, 2 of the same subject and make su
}
