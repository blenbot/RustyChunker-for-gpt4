use thiserror::Error;

/// Custom error types for the PDF processing pipeline
/// 
/// This provides structured error handling throughout the application
/// and allows for better error reporting to Python
#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error("PDF loading failed: {0}")]
    PdfLoadError(String),
    
    #[error("Text extraction failed on page {page}: {error}")]
    TextExtractionError { page: usize, error: String },
    
    #[error("Chunking failed: {0}")]
    ChunkingError(String),
    
    #[error("Parallel processing error: {0}")]
    ParallelError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("System error: {0}")]
    SystemError(String),
}