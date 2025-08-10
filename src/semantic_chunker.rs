use crate::error::ProcessingError;
use crate::tiktoken_core::CoreBPE;
use crate::chunking::ChunkMetadata;
use crate::text_preprocessor::TextPreprocessor;
use crate::semantic_segmenter::SemanticSegmenter;
use crate::chunk_merger::ChunkMerger;
use crate::chunk_overlapper::ChunkOverlapper;
use log::debug;

/// Advanced semantic-aware text chunker
/// 
/// Combines the best of both worlds:
/// 1. Exact tiktoken tokenization for accurate GPT-4 compatibility
/// 2. Semantic awareness to preserve meaning and structure
/// 3. RecursiveTextSplitter-style hierarchical splitting
/// 4. Token-level overlap for context preservation
/// 
/// Architecture:
/// - Preprocess: Clean excessive whitespace (your Python regex)
/// - Segment: Recursive semantic splitting (paragraphs -> sentences -> words)
/// - Merge: Greedy merging to target token size
/// - Overlap: Add token-level overlap between chunks
pub struct SemanticChunker {
    target_tokens: usize,
    overlap_tokens: usize,
    preprocessor: TextPreprocessor,
    segmenter: SemanticSegmenter,
    tokenizer: CoreBPE,
}

impl SemanticChunker {
    /// Create new semantic chunker with tiktoken integration
    pub fn new(target_tokens: usize, overlap_tokens: usize) -> Result<Self, ProcessingError> {
        debug!("Initializing semantic chunker: target={} tokens, overlap={} tokens", 
               target_tokens, overlap_tokens);
        
        let tokenizer = CoreBPE::new_o200k_base()?;
        
        Ok(Self {
            target_tokens,
            overlap_tokens,
            preprocessor: TextPreprocessor::new(),
            segmenter: SemanticSegmenter::new(),
            tokenizer,
        })
    }
    
    /// Apply semantic-aware chunking to page text
    /// 
    /// Process:
    /// 1. Preprocess: Apply your Python cleaning (excessive newlines, etc.)
    /// 2. Segment: Recursive splitting by semantic boundaries
    /// 3. Merge: Greedily combine segments up to target tokens
    /// 4. Overlap: Add token-level overlap between chunks
    /// 5. Return: Standard ChunkMetadata format
    pub fn chunk_page_text(
        &self,
        page_num: usize,
        text: &str,
        source: &str,
    ) -> Result<Vec<ChunkMetadata>, ProcessingError> {
        debug!("Semantic chunking page {}: {} characters", page_num, text.len());
        
        // Step 1: Preprocess text (your Python regex + cleanup)
        let cleaned_text = self.preprocessor.preprocess(text);
        
        if cleaned_text.trim().is_empty() {
            debug!("Page {} is empty after preprocessing", page_num);
            return Ok(vec![]);
        }
        
        // Check if entire page fits in one chunk
        let total_tokens = self.tokenizer.encode_ordinary(&cleaned_text).len();
        debug!("Page {} total tokens: {}", page_num, total_tokens);
        
        if total_tokens <= self.target_tokens {
            // Single chunk case
            return Ok(vec![ChunkMetadata {
                page: page_num,
                chunk_id: 0,
                text: cleaned_text,
                source: source.to_string(),
                token_count: total_tokens,
            }]);
        }
        
        // Step 2: Semantic segmentation using recursive strategy
        let segments = self.segmenter.segment(&cleaned_text, self.target_tokens, &self.tokenizer);
        debug!("Page {} segmented into {} semantic segments", page_num, segments.len());
        
        if segments.is_empty() {
            return Ok(vec![]);
        }
        
        // Step 3: Merge segments into chunks
        let merger = ChunkMerger::new(self.target_tokens, self.tokenizer.clone());
        let semantic_chunks = merger.merge_segments(segments)?;
        debug!("Page {} merged into {} semantic chunks", page_num, semantic_chunks.len());
        
        // Step 4: Add overlap and convert to final format
        let overlapper = ChunkOverlapper::new(self.overlap_tokens, self.tokenizer.clone());
        let final_chunks = overlapper.add_overlap_and_finalize(semantic_chunks, page_num, source)?;
        
        debug!("Page {} semantic chunking complete: {} final chunks", page_num, final_chunks.len());
        
        // Log final chunk statistics
        for (i, chunk) in final_chunks.iter().enumerate() {
            debug!("  Chunk {}: {} tokens, {} chars", i, chunk.token_count, chunk.text.len());
        }
        
        Ok(final_chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_chunking() {
        let chunker = SemanticChunker::new(100, 10).unwrap();
        
        let text = "First paragraph with some content.\n\nSecond paragraph with more content.\n\nThird paragraph with even more content to test the chunking logic.";
        
        let chunks = chunker.chunk_page_text(1, text, "test.pdf").unwrap();
        
        assert!(!chunks.is_empty());
        assert!(chunks[0].token_count > 0);
        
        // Should preserve paragraph boundaries when possible
        if chunks.len() > 1 {
            assert!(chunks[1].token_count > chunks[0].token_count); // Overlap included
        }
    }

    #[test]
    fn test_excessive_newlines_cleaning() {
        let chunker = SemanticChunker::new(100, 10).unwrap();
        
        let text = "Line 1\n\n\n\nLine 2\n \n \n\nLine 3";
        let chunks = chunker.chunk_page_text(1, text, "test.pdf").unwrap();
        
        assert!(!chunks.is_empty());
        // Should not contain excessive newlines
        assert!(!chunks[0].text.contains("\n\n\n"));
    }
}
