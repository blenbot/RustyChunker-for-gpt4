use crate::error::ProcessingError;
use crate::text_extractor::TextExtractor;
use crate::chunking::{ChunkMetadata, TextChunker};
use pdfium_render::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;
use log::{info, debug, error};
use std::sync::Once;

static INIT: Once = Once::new();

/// Parallel processing coordinator for PDF pages
/// 
/// Architecture:
/// - Uses Rayon for CPU-bound parallel processing
/// - Dynamic batch sizing based on available cores
/// - Load balancing across logical processors
/// - Optimized for Windows systems with 16 logical cores
pub struct ParallelProcessor {
    batch_size: usize,      // Pages per batch
    max_parallelism: usize, // Maximum parallel threads
}

impl ParallelProcessor {
    /// Initialize parallel processor with system-aware configuration
    /// 
    /// Calculates optimal batch sizes based on:
    /// - Available logical cores (16 on your system)
    /// - Expected memory usage per page
    /// - PDF processing overhead
    pub async fn new(logical_cores: usize) -> Result<Self, ProcessingError> {
        // Calculate batch size: aim for 2-4x logical cores for I/O bound work
        // This ensures good CPU utilization without excessive memory usage
        let batch_size = std::cmp::max(logical_cores * 2, 8);
        
        // Set maximum parallelism to logical cores
        let max_parallelism = logical_cores;
        
        info!("Parallel processor initialized: batch_size={}, max_parallelism={}", 
              batch_size, max_parallelism);
        
        // Configure Rayon thread pool for optimal performance
        // Use a flag to track if initialization was successful
        let mut thread_pool_error: Option<String> = None;
        
        INIT.call_once(|| {
            if let Err(e) = rayon::ThreadPoolBuilder::new()
                .num_threads(max_parallelism)
                .build_global()
            {
                error!("Thread pool setup failed: {}", e);
                // Store the error in our local variable
                thread_pool_error = Some(format!("Thread pool setup failed: {}", e));
            }
        });
        
        // Check if thread pool initialization failed and return error to Python
        if let Some(error_msg) = thread_pool_error {
            return Err(ProcessingError::ParallelError(error_msg));
        }
        
        Ok(ParallelProcessor {
            batch_size,
            max_parallelism,
        })
    }
    
    /// Process all PDF pages in parallel batches
    /// 
    /// Processing Strategy:
    /// 1. Pre-extract text from all pages sequentially (pdfium limitation)
    /// 2. Process text chunks in parallel using Rayon
    /// 3. Collect and merge results maintaining page order
    /// 4. Handle errors gracefully with detailed reporting
    pub async fn process_pages_parallel<'a>(
        &self,
        document: &PdfDocument<'a>,
        source_filename: &str,
        text_extractor: &TextExtractor,
        text_chunker: &TextChunker,
    ) -> Result<Vec<ChunkMetadata>, ProcessingError> {
        let page_count = document.pages().len() as usize;
        let source = source_filename.to_string();
        
        info!("Starting parallel processing: {} pages in batches of {}", 
              page_count, self.batch_size);
        
        // Step 1: Extract text from all pages sequentially (pdfium is not thread-safe)
        let mut page_texts: Vec<(usize, String)> = Vec::with_capacity(page_count);
        
        for page_idx in 0..page_count {
            let page_idx_u16 = page_idx as u16;
            
            match document.pages().get(page_idx_u16) {
                Ok(page) => {
                    match text_extractor.extract_page_text(&page, page_idx) {
                        Ok(text) => {
                            if !text.trim().is_empty() {
                                page_texts.push((page_idx, text));
                            }
                        }
                        Err(e) => {
                            error!("Failed to extract text from page {}: {}", page_idx, e);
                            continue;
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to get page {}: {}", page_idx, e);
                    continue;
                }
            }
        }
        
        info!("Text extraction complete: {} pages with content", page_texts.len());
        
        // Step 2: Process extracted text in parallel (thread-safe)
        let source_arc = Arc::new(source);
        let text_extractor_arc = Arc::new(text_extractor);
        let text_chunker_arc = Arc::new(text_chunker);
        
        let batch_results: Result<Vec<Vec<ChunkMetadata>>, ProcessingError> = 
            page_texts
                .chunks(self.batch_size)
                .enumerate()
                .collect::<Vec<_>>()  // Collect to enable parallel processing
                .into_par_iter()      // Convert to parallel iterator
                .map(|(batch_idx, batch_pages)| {
                    debug!("Processing batch {} with {} pages", batch_idx, batch_pages.len());
                    self.process_text_batch(
                        batch_pages, 
                        &source_arc, 
                        &text_extractor_arc, 
                        &text_chunker_arc
                    )
                })
                .collect();
        
        // Step 3: Flatten batch results and maintain page order
        let batch_results = batch_results?;
        let mut all_chunks: Vec<ChunkMetadata> = batch_results
            .into_iter()
            .flatten()
            .collect();
        
        // Sort by page number to ensure consistent output order
        // This is important since parallel processing can complete out of order
        all_chunks.sort_by(|a, b| {
            a.page.cmp(&b.page)
                .then(a.chunk_id.cmp(&b.chunk_id))
        });
        
        info!("Parallel processing complete: {} total chunks", all_chunks.len());
        Ok(all_chunks)
    }
    
    /// Process a single batch of pre-extracted text (thread-safe)
    /// 
    /// This is called in parallel for each batch and handles:
    /// - Word counting and chunking logic
    /// - Error handling per page
    /// - Memory management for large documents
    fn process_text_batch(
        &self,
        page_texts: &[(usize, String)], // (page_index, text)
        source: &Arc<String>,
        text_extractor: &Arc<&TextExtractor>,
        text_chunker: &Arc<&TextChunker>,
    ) -> Result<Vec<ChunkMetadata>, ProcessingError> {
        debug!("Processing text batch: {} pages", page_texts.len());
        
        let mut batch_chunks = Vec::new();
        
        // Process each page's text in the batch
        for (page_idx, text) in page_texts {
            match self.process_single_page_text(*page_idx, text, source, text_extractor, text_chunker) {
                Ok(mut page_chunks) => {
                    batch_chunks.append(&mut page_chunks);
                }
                Err(e) => {
                    error!("Failed to process page {} text: {}", page_idx, e);
                    // Continue processing other pages rather than failing the entire batch
                    continue;
                }
            }
        }
        
        debug!("Text batch complete: {} chunks generated", batch_chunks.len());
        Ok(batch_chunks)
    }
    
    /// Process a single page's pre-extracted text (thread-safe)
    /// 
    /// Text processing pipeline:
    /// 1. Count words for chunking decision
    /// 2. Apply chunking logic (300/60 word rules)
    /// 3. Generate chunk metadata
    /// 4. Handle edge cases (empty text, processing errors)
    fn process_single_page_text(
        &self,
        page_idx: usize,
        text: &str,
        source: &Arc<String>,
        text_extractor: &Arc<&TextExtractor>,
        text_chunker: &Arc<&TextChunker>,
    ) -> Result<Vec<ChunkMetadata>, ProcessingError> {
        debug!("Processing page {} text", page_idx);
        
        // Extract words for chunking analysis
        let words = text_extractor.extract_words(text);
        let word_count = words.len();
        
        debug!("Page {} contains {} words", page_idx, word_count);
        
        // Apply chunking logic based on word count
        let chunks = text_chunker.chunk_page_text(
            page_idx + 1, // Convert to 1-based page numbers for user output
            text,
            words,
            source,
        )?;
        
        debug!("Page {} generated {} chunks", page_idx, chunks.len());
        Ok(chunks)
    }
}