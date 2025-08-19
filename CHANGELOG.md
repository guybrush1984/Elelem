# Changelog

All notable changes to the Elelem project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-19

### Added
- Initial release of Elelem as a standalone Python package
- Unified API wrapper for OpenAI, GROQ, and DeepInfra
- Comprehensive cost tracking with tag-based categorization
- JSON validation and retry logic with temperature adjustment
- Rate limit handling with exponential backoff
- Automatic think tag removal for reasoning models
- Provider-specific parameter handling (auto-remove unsupported parameters)
- Complete test suite with async support
- Modern Python packaging with pyproject.toml
- Detailed documentation and API reference

### Core Features
- Support for 15+ models across 3 providers
- OpenAI-compatible response format
- Precise token and cost calculation
- Statistics tracking (overall and by tag)
- Configuration management via JSON config file
- Environment variable support for API keys

### Provider Support
- **OpenAI**: gpt-4.1, gpt-4.1-mini, gpt-5, gpt-5-mini, o3, o3-mini
- **GROQ**: GPT OSS models, Kimi K2, Llama 4 variants
- **DeepInfra**: GPT OSS models, Llama 4 variants, Kimi K2, DeepSeek-R1

### Technical Details
- Async/await support throughout
- Type hints and mypy compatibility
- Black/isort code formatting
- Pytest test framework with fixtures
- MIT license
- Python 3.8+ compatibility

## [Unreleased]

### Planned
- Additional model support as providers release new models
- Enhanced error reporting and debugging features
- Performance optimizations for batch requests
- Extended statistics and analytics capabilities