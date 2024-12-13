use std::sync::Arc;

use anyhow::Ok;
use arrow_array::{
    cast::AsArray, types::Float32Type, FixedSizeListArray, RecordBatch, RecordBatchIterator,
    StringArray,
};
use arrow_schema::ArrowError;

use lancedb::query::{ExecutableQuery, QueryBase, VectorQuery};

use futures::TryStreamExt;
use lancedb::{
    arrow::arrow_schema::{DataType, Field, Schema},
    Table,
};
use serde::{Deserialize, Serialize};

pub struct VecDB {
    default_table: Table,
}
#[derive(Deserialize, Serialize, Debug)]
pub struct ContextRecord {
    pub filename: String,
    pub context: String,
    pub content: String,
    pub vector: Vec<f32>,
}
const DEFAULT_VEC_DIM: i32 = 384;

impl VecDB {
    pub async fn create_or_open(
        db_path: &str,
        default_table: &str,
        vec_size: Option<i32>,
    ) -> anyhow::Result<Self> {
        let connection = lancedb::connect(db_path).execute().await?;
        let table_exists = connection
            .table_names()
            .execute()
            .await?
            .contains(&default_table.to_string());
        if !table_exists {
            println!("Table {} does not exist, creating it", default_table);
            let vec_size = vec_size.unwrap_or(DEFAULT_VEC_DIM);
            let schema = Self::get_default_schema(vec_size);
            connection
                .create_empty_table(default_table, schema)
                .execute()
                .await?;
        }
        let table = connection.open_table(default_table).execute().await?;
        Ok(Self {
            default_table: table,
        })
    }
    pub async fn find_similar_x(&self, vector: Vec<f32>, n: usize) -> anyhow::Result<RecordBatch> {
        let query_results = self
            .default_table
            .query()
            .nearest_to(vector)?
            .column("vector")
            .select(lancedb::query::Select::Columns(vec!["content".to_string()]))
            .limit(n)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let first = query_results
            .first()
            .ok_or(anyhow::anyhow!("error getting first result"))?;
        Ok(first.clone())
    }

    pub async fn find_similar(&self, vector: Vec<f32>, n: usize) -> anyhow::Result<RecordBatch> {
        let results = self
            .default_table
            .query()
            .nearest_to(vector)?
            .select(lancedb::query::Select::Columns(vec!["content".to_string()]))
            .limit(n)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let first = results.first().unwrap();
        Ok(first.clone())
    }

    /// Get the default schema for the VecDB
    pub fn get_default_schema(vec_size: i32) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("filename", DataType::Utf8, false),
            Field::new("context", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    vec_size,
                ),
                true,
            ),
        ]))
    }

    pub async fn insert_vector_as_struct(&self, records: Vec<ContextRecord>) -> anyhow::Result<()> {
        let schema = self.default_table.schema().await?;
        let fields = schema.fields();
        let batches = serde_arrow::to_record_batch(&fields, &records)
            .map_err(|e| ArrowError::from_external_error(e.into()));
        let batches = vec![batches];
        let batch_iterator = RecordBatchIterator::new(batches, schema);
        // Create a RecordBatch stream.
        let boxed_batches = Box::new(batch_iterator);
        // add them to the table
        self.default_table.add(boxed_batches).execute().await?;
        Ok(())
    }

    pub async fn add_vector(
        &self,
        filenames: &[&str],
        contexts: &[&str],
        contents: &[&str],
        vectors: Vec<Vec<f32>>,
    ) -> anyhow::Result<()> {
        let schema = self.default_table.schema().await?;
        // get vector dimension from the vectors array
        let vec_dim = vectors[0].len() as i32;
        let key_array = StringArray::from_iter_values(filenames);
        let context_array = StringArray::from_iter_values(contexts);
        let content_array = StringArray::from_iter_values(contents);
        let vectors_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vectors
                .into_iter()
                .map(|v| Some(v.into_iter().map(|i| Some(i)))),
            vec_dim,
        );
        let batches = vec![Ok(RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(key_array),
                Arc::new(context_array),
                Arc::new(content_array),
                Arc::new(vectors_array),
            ],
        )?)
        .map_err(|e| ArrowError::from_external_error(e.into()))];
        let batch_iterator = RecordBatchIterator::new(batches, schema);
        // Create a RecordBatch stream.
        let boxed_batches = Box::new(batch_iterator);
        // add them to the table
        self.default_table.add(boxed_batches).execute().await?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use tempdir;

    #[tokio::test]
    async fn test_vector_insertion_and_similarity() -> anyhow::Result<()> {
        // Create a temporary directory for the test database
        let temp_dir = tempdir::TempDir::new("test_db")?;
        let db_path = temp_dir.path().to_str().unwrap();
        // Initialize the database
        let vec_db = VecDB::create_or_open(db_path, "test_table", Some(3)).await?;

        // Create three test vectors
        let vectors = vec![
            vec![1.0, 0.0, 0.0], // Vector pointing in x direction
            vec![0.0, 1.0, 0.0], // Vector pointing in y direction
            vec![0.0, 0.0, 1.0], // Vector pointing in z direction
        ];

        let filenames = vec!["vec1", "vec2", "vec3"];
        let contexts = vec!["context1", "context2", "context3"];
        let contents = vec!["content1", "content2", "content3"];
        // Insert the vectors
        vec_db
            .add_vector(&filenames, &contexts, &contents, vectors)
            .await?;

        // Test similarity search with a vector similar to the first one
        let query_vector = vec![0.9, 0.1, 0.0];
        let results = vec_db.find_similar(query_vector, 3).await?;

        // Get the filename column
        let filename_array = results.column_by_name("filename").unwrap();
        let filename_array = filename_array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        // The most similar vector should be vec1 (the x-direction vector)
        assert_eq!(filename_array.value(0), "vec1");

        Ok(())
    }

    #[tokio::test]
    async fn test_vector_record_retrieval() -> anyhow::Result<()> {
        // Create a temporary directory for the test database
        let temp_dir = tempdir::TempDir::new("test_db")?;
        let db_path = temp_dir.path().to_str().unwrap();

        // Initialize the database
        let vec_db = VecDB::create_or_open(db_path, "test_table", Some(3)).await?;

        // Create test vectors
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let filenames = vec!["vec1", "vec2", "vec3"];
        let contexts = vec!["context1", "context2", "context3"];
        let contents = vec!["content1", "content2", "content3"];

        // Insert the vectors
        vec_db
            .add_vector(&filenames, &contexts, &contents, vectors)
            .await?;

        // Search for similar vectors
        let query_vector = vec![0.9, 0.1, 0.0];
        let results = vec_db.find_similar(query_vector, 2).await?;

        let records: Vec<ContextRecord> = serde_arrow::from_record_batch(&results)?;

        println!("{:?}", records);

        Ok(())
    }

    #[tokio::test]
    async fn test_struct_insertion_and_retrieval() -> anyhow::Result<()> {
        // Create a temporary directory for the test database
        let temp_dir = tempdir::TempDir::new("test_db")?;
        let db_path = temp_dir.path().to_str().unwrap();

        // Initialize the database
        let vec_db = VecDB::create_or_open(db_path, "test_table", Some(3)).await?;

        // Create test data
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let entries = vec![
            ContextRecord {
                filename: "test1".to_string(),
                context: "test context 1".to_string(),
                content: "test content 1".to_string(),
                vector: vectors[0].clone(),
            },
            ContextRecord {
                filename: "test2".to_string(),
                context: "test context 2".to_string(),
                content: "test content 2".to_string(),
                vector: vectors[1].clone(),
            },
            ContextRecord {
                filename: "test3".to_string(),
                context: "test context 3".to_string(),
                content: "test content 3".to_string(),
                vector: vectors[2].clone(),
            },
        ];

        vec_db.insert_vector_as_struct(entries).await?;

        // Query and verify results
        let query_vector = vec![0.9, 0.1, 0.0];
        let results = vec_db.find_similar(query_vector, 1).await?;

        let retrieved_records: Vec<ContextRecord> = serde_arrow::from_record_batch(&results)?;

        // Verify first result matches expected
        assert_eq!(retrieved_records[0].filename, "test1");
        assert_eq!(retrieved_records[0].context, "test context 1");
        assert_eq!(retrieved_records[0].content, "test content 1");
        assert_eq!(retrieved_records[0].vector, vectors[0]);

        Ok(())
    }
}
