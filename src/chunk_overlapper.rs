use crate::chunk_merger::SemanticChunk;
use crate::tiktoken_core::CoreBPE;
use crate::error::ProcessingError;
use crate::chunking::ChunkMetadata;
use log::debug;

/// Adds overlap between chunks for better context preservation
/// 
/// Implements token-level overlap:
/// 1. For each chunk after the first, include last N tokens from previous chunk
/// 2. Updates token counts to reflect overlap
/// 3. Maintains semantic boundaries when possible
pub struct ChunkOverlapper {
    overlap_tokens: usize,
    tokenizer: CoreBPE,
}

impl ChunkOverlapper {
    pub fn new(overlap_tokens: usize, tokenizer: CoreBPE) -> Self {
        Self {
            overlap_tokens,
            tokenizer,
        }
    }
    
    /// Add overlap to chunks and convert to final ChunkMetadata format
    /// 
    /// Strategy:
    /// 1. Keep first chunk as-is
    /// 2. For subsequent chunks, prepend overlap from previous chunk
    /// 3. Update token counts to reflect actual content
    /// 4. Convert to ChunkMetadata format for compatibility
    pub fn add_overlap_and_finalize(
        &self,
        semantic_chunks: Vec<SemanticChunk>,
        page_num: usize,
        source: &str,
    ) -> Result<Vec<ChunkMetadata>, ProcessingError> {
        debug!("Adding {} token overlap to {} chunks", self.overlap_tokens, semantic_chunks.len());
        
        if semantic_chunks.is_empty() {
            return Ok(vec![]);
        }
        
        let mut final_chunks = Vec::new();
        let mut previous_chunk_tokens: Option<Vec<u32>> = None;
        
        for (chunk_id, semantic_chunk) in semantic_chunks.iter().enumerate() {
            let chunk_text = if chunk_id == 0 || previous_chunk_tokens.is_none() {
                // First chunk or no previous chunk - no overlap needed
                semantic_chunk.text.clone()
            } else {
                // Add overlap from previous chunk
                self.add_overlap_to_chunk(semantic_chunk, &previous_chunk_tokens.as_ref().unwrap())?
            };
            
            // Tokenize the final chunk text to get accurate count
            let chunk_tokens = self.tokenizer.encode_ordinary(&chunk_text);
            let token_count = chunk_tokens.len();
            
            // Store tokens for next iteration's overlap
            previous_chunk_tokens = Some(chunk_tokens);
            
            // Convert to ChunkMetadata format
            final_chunks.push(ChunkMetadata {
                page: page_num,
                chunk_id,
                text: chunk_text,
                source: source.to_string(),
                token_count,
            });
            
            debug!("Chunk {}: {} tokens (with overlap)", chunk_id, token_count);
        }
        
        debug!("Finalized {} chunks with overlap", final_chunks.len());
        Ok(final_chunks)
    }
    
    /// Add overlap tokens from previous chunk to current chunk
    fn add_overlap_to_chunk(
        &self,
        current_chunk: &SemanticChunk,
        previous_tokens: &[u32],
    ) -> Result<String, ProcessingError> {
        if self.overlap_tokens == 0 || previous_tokens.is_empty() {
            return Ok(current_chunk.text.clone());
        }
        
        // Get the last N tokens from previous chunk
        let overlap_start = if previous_tokens.len() > self.overlap_tokens {
            previous_tokens.len() - self.overlap_tokens
        } else {
            0
        };
        
        let overlap_tokens = &previous_tokens[overlap_start..];
        
        // Decode overlap tokens back to text
        let overlap_text = self.tokenizer.decode(overlap_tokens)
            .map_err(|e| ProcessingError::ChunkingError(
                format!("Failed to decode overlap tokens: {}", e)
            ))?;
        
        // Combine overlap with current chunk
        // Add a space separator if both parts have content
        let combined_text = if overlap_text.trim().is_empty() {
            current_chunk.text.clone()
        } else if current_chunk.text.trim().is_empty() {
            overlap_text
        } else {
            format!("{} {}", overlap_text.trim(), current_chunk.text.trim())
        };
        
        Ok(combined_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk_merger::SemanticChunk;

    #[test]
    fn test_overlap_addition() {
        let tokenizer = CoreBPE::new_o200k_base().unwrap();
        let overlapper = ChunkOverlapper::new(5, tokenizer);
        
        let chunks = vec![
            SemanticChunk {
                text: "First chunk with some content.".to_string(),
                token_count: 6,
                start_offset: 0,
                end_offset: 30,
                segments: vec![0],
            },
            SemanticChunk {
                text: "Second chunk with different content.".to_string(),
                token_count: 6,
                start_offset: 31,
                end_offset: 67,
                segments: vec![1],
            },
        ];
        
        let result = overlapper.add_overlap_and_finalize(chunks, 1, "test.pdf").unwrap();
        
        assert_eq!(result.len(), 2);
        assert!(result[1].text.len() > result[0].text.len()); // Second chunk should have overlap
        assert!(result[1].token_count > 6); // Should include overlap tokens
    }
}
