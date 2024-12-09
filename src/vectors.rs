pub struct VectorsOps;

impl VectorsOps {
    pub fn weighted_average(
        e1: &Vec<f32>, // Embedding of the paragraph
        w1: f32,       // Weight for the paragraph
        e2: &Vec<f32>, // Embedding of the heavier sentence
        w2: f32,
        normalize_result: bool, // Weight for the heavier sentence
    ) -> Vec<f32> {
        let total_weight = w1 + w2;
        // Compute the weighted average
        let weighted_avg: Vec<f32> = e1
            .iter()
            .zip(e2.iter())
            .map(|(&x1, &x2)| (w1 * x1 + w2 * x2) / total_weight)
            .collect();
        // Normalize the weighted average embedding
        if normalize_result {
            Self::normalize(&weighted_avg)
        } else {
            weighted_avg
        }
    }

    pub fn normalize(v: &Vec<f32>) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Avoid division by zero
        if norm == 0.0 {
            return v.clone();
        }
        v.iter().map(|x| x / norm).collect()
    }

    fn cosine_similarity(v1: &Vec<f32>, v2: &Vec<f32>) -> f32 {
        v1.iter().zip(v2.iter()).map(|(&x1, &x2)| x1 * x2).sum()
    }

    fn find_closest_embedding<'a>(
        query: &Vec<f32>,
        embeddings: &'a [Vec<f32>],
    ) -> Option<&'a Vec<f32>> {
        embeddings.iter().max_by(|a, b| {
            Self::cosine_similarity(query, a)
                .partial_cmp(&Self::cosine_similarity(query, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_average_dominance() {
        let v1 = vec![1.0, 0.0, 0.0]; // Vector pointing in x direction
        let v2 = vec![0.0, 1.0, 0.0]; // Vector pointing in y direction

        // When v2 has much higher weight, result should be more similar to v2
        let weighted = VectorsOps::weighted_average(&v1, 1.0, &v2, 3.0, true);

        let sim_to_v1 = VectorsOps::cosine_similarity(&weighted, &v1);
        let sim_to_v2 = VectorsOps::cosine_similarity(&weighted, &v2);

        println!("sim_to_v1: {}", sim_to_v1);
        println!("sim_to_v2: {}", sim_to_v2);

        // The weighted result should be more similar to v2 (the heavier vector)
        assert!(sim_to_v2 > sim_to_v1);

        // With normalized vectors and results, we can be more precise about the expected similarities
        assert!((sim_to_v2 - 0.95).abs() < 0.05); // Should be very similar to v2
        assert!((sim_to_v1 - 0.31).abs() < 0.05); // Should be less similar to v1
    }
}
