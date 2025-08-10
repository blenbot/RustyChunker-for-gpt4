use crate::error::ProcessingError;
use crate::tiktoken_core::CoreBPE;
use crate::semantic_chunker::SemanticChunker;
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
    pub token_count: usize,  // Real tiktoken count
}

/// Chunking strategy options
#[derive(Debug, Clone)]
pub enum ChunkingStrategy {
    /// Simple token-based sliding window (original approach)
    SimpleToken,
    /// Semantic-aware chunking with recursive text splitting (new approach)
    SemanticAware,
}

/// Enhanced text chunker with multiple strategies
/// 
/// Supports both:
/// 1. Simple token-based chunking (original approach)
/// 2. Semantic-aware chunking (new RecursiveTextSplitter approach)
/// 
/// Uses real tiktoken tokenization for accurate GPT-4 compatibility
pub struct TextChunker {
    chunk_size: usize,    // Target tokens per chunk (256)
    step_size: usize,     // How far to move the window (240 = 256 - 16)  
    overlap_size: usize,  // Overlap tokens (16)
    tokenizer: CoreBPE,   // tiktoken tokenizer for accurate token counting
    semantic_chunker: SemanticChunker, // New semantic-aware chunker
    strategy: ChunkingStrategy, // Which strategy to use
}

impl TextChunker {
    /// Create new text chunker with specified parameters and strategy
    pub fn new(chunk_size: usize, overlap_size: usize) -> Result<Self, ProcessingError> {
        let step_size = chunk_size.saturating_sub(overlap_size);
        
        debug!("Initializing text chunker: chunk_size={}, overlap={}, step_size={}", 
               chunk_size, overlap_size, step_size);
        
        // Initialize tiktoken o200k_base tokenizer
        let tokenizer = CoreBPE::new_o200k_base()?;
        
        // Initialize semantic chunker with same parameters
        let semantic_chunker = SemanticChunker::new(chunk_size, overlap_size)?;
        
        Ok(TextChunker {
            chunk_size,
            step_size,
            overlap_size,
            tokenizer,
            semantic_chunker,
            strategy: ChunkingStrategy::SemanticAware, // Default to semantic-aware
        })
    }
    
    /// Set chunking strategy
    pub fn with_strategy(mut self, strategy: ChunkingStrategy) -> Self {
        self.strategy = strategy;
        self
    }
    
    /// Apply chunking logic to page text using selected strategy
    pub fn chunk_page_text(
        &self,
        page_num: usize,
        text: &str,
        source: &str,
    ) -> Result<Vec<ChunkMetadata>, ProcessingError> {
        match self.strategy {
            ChunkingStrategy::SimpleToken => {
                self.chunk_page_text_simple(page_num, text, source)
            }
            ChunkingStrategy::SemanticAware => {
                self.semantic_chunker.chunk_page_text(page_num, text, source)
            }
        }
    }
    
    /// Original simple token-based chunking (preserved for comparison)
    fn chunk_page_text_simple(
        &self,
        page_num: usize,
        text: &str,
        source: &str,
    ) -> Result<Vec<ChunkMetadata>, ProcessingError> {
        debug!("Simple tokenizing page {}", page_num);
        
        // Tokenize the entire page text using tiktoken
        let tokens = self.tokenizer.encode_ordinary(text);
        let token_count = tokens.len();
        
        debug!("Page {} contains {} tokens", page_num, token_count);
        
        // Case 1: Page has ≤256 tokens - return single chunk
        if token_count <= self.chunk_size {
            debug!("Page {} has ≤{} tokens, returning single chunk", page_num, self.chunk_size);
            
            return Ok(vec![ChunkMetadata {
                page: page_num,
                chunk_id: 0,
                text: text.to_string(),
                source: source.to_string(),
                token_count,
            }]);
        }
        
        // Case 2: Page has >256 tokens - apply sliding window chunking
        debug!("Page {} has >{} tokens, applying sliding window", page_num, self.chunk_size);
        
        let mut chunks = Vec::new();
        let mut chunk_id = 0;
        let mut start_token_idx = 0;
        
        // Sliding window loop over tokens
        while start_token_idx < token_count {
            // Calculate end token index for current chunk
            let end_token_idx = std::cmp::min(start_token_idx + self.chunk_size, token_count);
            
            // Extract tokens for current chunk
            let chunk_tokens = &tokens[start_token_idx..end_token_idx];
            let chunk_token_count = chunk_tokens.len();
            
            // Decode tokens back to text
            let chunk_text = self.tokenizer.decode(chunk_tokens)
                .map_err(|e| ProcessingError::ChunkingError(
                    format!("Failed to decode tokens for page {} chunk {}: {}", page_num, chunk_id, e)
                ))?;
            
            debug!("Page {} chunk {}: tokens {}-{} ({} tokens)", 
                   page_num, chunk_id, start_token_idx, end_token_idx-1, chunk_token_count);
            
            // Create chunk metadata
            chunks.push(ChunkMetadata {
                page: page_num,
                chunk_id,
                text: chunk_text,
                source: source.to_string(),
                token_count: chunk_token_count,
            });
            
            // Break if we've reached the end
            if end_token_idx >= token_count {
                break;
            }
            
            // Move window forward by step_size (240 tokens for 256-16)
            // This creates the 16-token overlap
            start_token_idx += self.step_size;
            chunk_id += 1;
            
            // Safety check to prevent infinite loops
            if chunk_id > 1000 {
                return Err(ProcessingError::ChunkingError(
                    format!("Too many chunks generated for page {}", page_num)
                ));
            }
        }
        
        debug!("Page {} simple token-based chunking complete: {} chunks generated", page_num, chunks.len());
        Ok(chunks)
    }
}