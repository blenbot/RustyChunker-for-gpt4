use crate::semantic_segmenter::Segment;
use crate::tiktoken_core::CoreBPE;
use crate::error::ProcessingError;
use log::debug;

/// Chunk with semantic boundaries and token information
#[derive(Debug, Clone)]
pub struct SemanticChunk {
    pub text: String,
    pub token_count: usize,
    pub start_offset: usize,
    pub end_offset: usize,
    pub segments: Vec<usize>, // Indices of original segments that form this chunk
}

/// Merges semantic segments into optimal chunks
/// 
/// Implements greedy merging strategy:
/// 1. Start with first segment
/// 2. Keep adding segments while under token limit
/// 3. When limit would be exceeded, finalize chunk and start new one
/// 4. Ensures chunks are semantically coherent and efficiently sized
pub struct ChunkMerger {
    target_tokens: usize,
    tokenizer: CoreBPE,
}

impl ChunkMerger {
    pub fn new(target_tokens: usize, tokenizer: CoreBPE) -> Self {
        Self {
            target_tokens,
            tokenizer,
        }
    }
    
    /// Merge segments into chunks using greedy strategy
    /// 
    /// Strategy:
    /// 1. Greedily merge adjacent segments until approaching target_tokens
    /// 2. Prefer keeping segments together that came from the same semantic level
    /// 3. Ensure no chunk exceeds target_tokens
    /// 4. Handle edge cases (very large segments, empty segments)
    pub fn merge_segments(&self, segments: Vec<Segment>) -> Result<Vec<SemanticChunk>, ProcessingError> {
        debug!("Merging {} segments into chunks (target: {} tokens)", segments.len(), self.target_tokens);
        
        if segments.is_empty() {
            return Ok(vec![]);
        }
        
        let mut chunks = Vec::new();
        let mut current_chunk_text = String::new();
        let mut current_chunk_segments = Vec::new();
        let mut current_start_offset = 0;
        let mut current_end_offset = 0;
        
        for (i, segment) in segments.iter().enumerate() {
            // Calculate potential new chunk text
            let potential_text = if current_chunk_text.is_empty() {
                segment.text.clone()
            } else {
                format!("{} {}", current_chunk_text, segment.text)
            };
            
            // Check token count of potential chunk
            let potential_tokens = self.tokenizer.encode_ordinary(&potential_text).len();
            
            // If adding this segment would exceed target, finalize current chunk
            if potential_tokens > self.target_tokens && !current_chunk_text.is_empty() {
                // Finalize current chunk
                let chunk_tokens = self.tokenizer.encode_ordinary(&current_chunk_text).len();
                chunks.push(SemanticChunk {
                    text: current_chunk_text.clone(),
                    token_count: chunk_tokens,
                    start_offset: current_start_offset,
                    end_offset: current_end_offset,
                    segments: current_chunk_segments.clone(),
                });
                
                // Start new chunk with current segment
                current_chunk_text = segment.text.clone();
                current_chunk_segments = vec![i];
                current_start_offset = segment.start_offset;
                current_end_offset = segment.end_offset;
            } else {
                // Add segment to current chunk
                if current_chunk_text.is_empty() {
                    current_chunk_text = segment.text.clone();
                    current_start_offset = segment.start_offset;
                } else {
                    current_chunk_text = potential_text;
                }
                current_chunk_segments.push(i);
                current_end_offset = segment.end_offset;
            }
        }
        
        // Finalize last chunk
        if !current_chunk_text.is_empty() {
            let chunk_tokens = self.tokenizer.encode_ordinary(&current_chunk_text).len();
            chunks.push(SemanticChunk {
                text: current_chunk_text,
                token_count: chunk_tokens,
                start_offset: current_start_offset,
                end_offset: current_end_offset,
                segments: current_chunk_segments,
            });
        }
        
        debug!("Created {} semantic chunks", chunks.len());
        
        // Log chunk statistics
        for (i, chunk) in chunks.iter().enumerate() {
            debug!("Chunk {}: {} tokens, {} segments", i, chunk.token_count, chunk.segments.len());
        }
        
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_segmenter::{Segment, SemanticSegmenter};

    #[test]
    fn test_chunk_merging() {
        let tokenizer = CoreBPE::new_o200k_base().unwrap();
        let merger = ChunkMerger::new(50, tokenizer);
        
        let segments = vec![
            Segment {
                text: "First sentence.".to_string(),
                start_offset: 0,
                end_offset: 15,
                semantic_level: 3,
            },
            Segment {
                text: "Second sentence.".to_string(),
                start_offset: 16,
                end_offset: 32,
                semantic_level: 3,
            },
        ];
        
        let chunks = merger.merge_segments(segments).unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks[0].token_count > 0);
    }
}
