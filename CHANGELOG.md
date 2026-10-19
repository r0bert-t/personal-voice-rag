# Changelog

## v0.0.7 - 19-10-2026

### Added or Changed
- fixed file MacOS permission issues in Claude Desktop integration by migrating from chromadb.Client to chromadb.PersistentClient
- Added support ChromaDB collections, enabling isolation different sets of documents and improving query performance

## v0.0.6 - 16-10-2026

### Added or Changed
- updated Text-to-Speech (TTS) conversion to support the latest Eleven v4 turbo model
- added functionality to check and list indexed pdf files in the vector database
- some minor fixes

## v0.0.5 - 29-08-2026

### Added or Changed
- enhanced support in MacOS for Claude Desktop integration with local RAG (MCP with stdio transport)
- some minor fixes

## v0.0.4 - 27-08-2026

### Added or Changed
- added MultiQueryRetriever support to improve retrieval accuracy for complex user queries
- refactored MCP server startup to use asynchronous architecture (asyncio) instead of operating system threads
- added validation to prevent the ingestion of empty chunks

## v0.0.3 - 25-08-2026

### Added or Changed
- added MCP server support
- added guide for connecting local RAG pipelines to the Claude desktop app, enabling private data querying directly from the Claude chat interface.

## v0.0.2 - 21-08-2026

### Added or Changed
- added Gradio UI support

## v0.0.1 - 16-08-2026

### Added or Changed
- releasing the project fully into the public domain
