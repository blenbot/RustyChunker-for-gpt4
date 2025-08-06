use crate::error::ProcessingError;
use crate::parallel_processor::ParallelProcessor;
use crate::text_extractor::TextExtractor;
use crate::chunking::{ChunkMetadata, TextChunker};
use pdfium_render::prelude::*;
use std::path::Path;
use log::{info, warn};

/// Main PDF processor that orchestrates the entire pipeline
pub struct PdfProcessor {
    parallel_processor: ParallelProcessor,
    text_extractor: TextExtractor,
    text_chunker: TextChunker,
    pdfium: Pdfium,
}

impl PdfProcessor {
    /// Initialize the PDF processor with dynamic system configuration
    pub async fn new() -> Result<Self, ProcessingError> {
        info!("Initializing PDF processor...");
        
        // Try multiple paths for pdfium library
        // Try to load pdfium from multiple paths
        let pdfium = Pdfium::new(
            Pdfium::bind_to_library(
                Pdfium::pdfium_platform_library_name_at_path("../")
            )
            .or_else(|_| {
                info!("Failed to load pdfium from '../', trying current directory");
                Pdfium::bind_to_library(
                    Pdfium::pdfium_platform_library_name_at_path("./")
                )
            })
            .or_else(|_| {
                info!("Failed to load pdfium from './', trying absolute path");
                Pdfium::bind_to_library(
                    Pdfium::pdfium_platform_library_name_at_path("C:/Users/harsh/Desktop/HackCentral/rustmethods/")
                )
            })
            .or_else(|_| {
                info!("Failed to load pdfium from absolute path, trying system library");
                Pdfium::bind_to_system_library()
            })
            .map_err(|e| ProcessingError::SystemError(format!("Failed to initialize pdfium: {}", e)))?
        );
        
        // Detect system capabilities for optimal parallel processing
        let logical_cores = num_cpus::get();
        info!("Detected {} logical cores", logical_cores);
        
        // Initialize components with system-aware configuration
        let parallel_processor = ParallelProcessor::new(logical_cores).await?;
        let text_extractor = TextExtractor::new();
        let text_chunker = TextChunker::new(300, 60); // 300 words per chunk, 60 word overlap
        
        Ok(PdfProcessor {
            parallel_processor,
            text_extractor,
            text_chunker,
            pdfium,
        })
    }
    
    /// Process a PDF file and return chunk metadata
    pub async fn process_pdf(&self, pdf_path: &str) -> Result<Vec<ChunkMetadata>, ProcessingError> {
        let path = Path::new(pdf_path);
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.pdf")
            .to_string();
        
        info!("Processing PDF: {}", pdf_path);
        
        // Load PDF document using pdfium
        let document = self.pdfium
            .load_pdf_from_file(path, None)
            .map_err(|e| ProcessingError::PdfLoadError(format!("Failed to load {}: {}", pdf_path, e)))?;
        
        let page_count = document.pages().len();
        info!("PDF loaded successfully. Pages: {}", page_count);
        
        if page_count == 0 {
            warn!("PDF contains no pages");
            return Ok(vec![]);
        }
        
        // Process pages in parallel batches
        let all_chunks = self.parallel_processor
            .process_pages_parallel(&document, &filename, &self.text_extractor, &self.text_chunker)
            .await?;
        
        info!("Processing complete. Generated {} total chunks", all_chunks.len());
        Ok(all_chunks)
    }
}