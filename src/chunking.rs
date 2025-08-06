use crate::error::ProcessingError;
use serde::{Serialize, Deserialize};
use log::debug;

/// Metadata structure for each text chunk
/// 
/// This represents the output format that will be converted to Python dictionaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub page: usize,
    pub chunk_id: usize,
    pub text: String,
    pub source: String,
}

/// Text chunking component implementing sliding window logic
/// 
/// Architecture:
/// - Target chunk size: 300 words
/// - Overlap size: 60 words  
/// - Sliding window: moves 240 words (300 - 60) at each step
/// - Handles edge cases: pages with <300 words return single chunk
pub struct TextChunker {
    chunk_size: usize,    // Target words per chunk (300)
    overlap_size: usize,  // Overlap between chunks (60)
    step_size: usize,     // How far to move the window (240 = 300 - 60)
}

impl TextChunker {
    /// Create new text chunker with specified parameters
    pub fn new(chunk_size: usize, overlap_size: usize) -> Self {
        let step_size = chunk_size.saturating_sub(overlap_size);
        
        debug!("Initialized chunker: chunk_size={}, overlap={}, step_size={}", 
               chunk_size, overlap_size, step_size);
        
        TextChunker {
            chunk_size,
            overlap_size,
            step_size,
        }
    }
    
    /// Apply chunking logic to page text
    /// 
    /// Logic:
    /// 1. If page has ≤300 words: return single chunk (chunk_id=0)
    /// 2. If page has >300 words: apply sliding window chunking
    /// 3. Each chunk maintains 60-word overlap with previous chunk
    /// 4. Final chunk includes all remaining words
    pub fn chunk_page_text(
        &self,
        page_num: usize,
        text: &str,
        words: Vec<String>,
        source: &str,
    ) -> Result<Vec<ChunkMetadata>, ProcessingError> {
        let word_count = words.len();
        
        debug!("Chunking page {}: {} words", page_num, word_count);
        
        // Case 1: Page has ≤300 words - return single chunk
        if word_count <= self.chunk_size {
            debug!("Page {} has ≤{} words, returning single chunk", page_num, self.chunk_size);
            
            return Ok(vec![ChunkMetadata {
                page: page_num,
                chunk_id: 0,
                text: text.to_string(),
                source: source.to_string(),
            }]);
        }
        
        // Case 2: Page has >300 words - apply sliding window chunking
        debug!("Page {} has >{} words, applying sliding window", page_num, self.chunk_size);
        
        let mut chunks = Vec::new();
        let mut chunk_id = 0;
        let mut start_idx = 0;
        
        // Sliding window loop
        while start_idx < word_count {
            // Calculate end index for current chunk
            let end_idx = std::cmp::min(start_idx + self.chunk_size, word_count);
            
            // Extract words for current chunk
            let chunk_words = &words[start_idx..end_idx];
            let chunk_text = chunk_words.join(" ");
            
            debug!("Page {} chunk {}: words {}-{} ({} words)", 
                   page_num, chunk_id, start_idx, end_idx-1, chunk_words.len());
            
            // Create chunk metadata
            chunks.push(ChunkMetadata {
                page: page_num,
                chunk_id,
                text: chunk_text,
                source: source.to_string(),
            });
            
            // Break if we've reached the end
            if end_idx >= word_count {
                break;
            }
            
            // Move window forward by step_size (240 words)
            // This creates the 60-word overlap
            start_idx += self.step_size;
            chunk_id += 1;
            
            // Safety check to prevent infinite loops
            if chunk_id > 1000 {
                return Err(ProcessingError::ChunkingError(
                    format!("Too many chunks generated for page {}", page_num)
                ));
            }
        }
        
        debug!("Page {} chunking complete: {} chunks generated", page_num, chunks.len());
        Ok(chunks)
    }
}