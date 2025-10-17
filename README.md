# ACE Framework

Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models

## Overview

ACE is a framework for building and evolving contexts that enable self-improving language models. It provides a structured approach to context engineering through three core components: Generator, Reflector, and Curator.

## Installation

```bash
pip install -e .
```

## Core Components

**Generator**: Creates initial responses and outputs based on the current context.

**Reflector**: Analyzes performance and generates insights for context improvement.

**Curator**: Manages and evolves the context based on reflections and performance data.

## Key Features

### Grow-and-Refine

ACE implements the grow-and-refine principle from the paper to maintain high-quality, comprehensive contexts:

- **Incremental Delta Updates**: New insights are added as bullets rather than rewriting the entire context
- **Semantic De-duplication**: Uses OpenAI embeddings to detect and merge semantically similar bullets
  - **Optimized Performance**: Vectorized numpy operations for 10-100x faster similarity computation
  - Computes all pairwise similarities in a single matrix multiplication
  - Leverages hardware acceleration (SIMD, multi-threading via BLAS)
- **Intelligent Pruning**: When context exceeds max_bullets, removes lowest-value bullets based on helpful/harmful feedback
- **Two Refinement Modes**:
  - **Proactive**: Refines after each delta update
  - **Lazy**: Refines only when max_bullets threshold is exceeded

#### Example Usage

```python
from ace import Context, OpenAIEmbedder

# Initialize embedder for semantic similarity
embedder = OpenAIEmbedder(model="text-embedding-3-small")

# Create context with grow-and-refine
context = Context(
    sections=["strategies", "common_mistakes", "insights"],
    max_bullets=50,                    # Maximum total bullets
    embedder=embedder,                 # Enable semantic de-duplication
    similarity_threshold=0.85,         # Threshold for duplicate detection
    refinement_mode="lazy"             # Refine only when max exceeded
)

# Add bullets (embeddings computed automatically)
context.add_bullet("strategies", "Always validate input before processing")
context.add_bullet("common_mistakes", "Forgetting to check edge cases")

# Manually trigger refinement to see stats
stats = context.refine()
print(f"Deduplicated: {stats['deduplicated']} bullets")
print(f"Pruned: {stats['pruned']} bullets")
```

### Embeddings Support

ACE provides a flexible embedder abstraction supporting multiple backends:

**OpenAI Embeddings**: Production-ready embeddings via OpenAI API
```python
from ace import OpenAIEmbedder

embedder = OpenAIEmbedder(
    model="text-embedding-3-small",  # or "text-embedding-3-large"
    dimensions=512                    # Optional: reduce dimensions
)
```

**Future Support**: The embedder abstraction is designed to support local models via Transformers in future releases.

## Basic Usage

```python
from ace import Context, Generator, Reflector, Curator
from ace.llm import OpenAIClient
from ace.utils import PromptManager

# Initialize LLM client
llm = OpenAIClient(model="gpt-4")

# Load prompts
prompts = PromptManager.from_yaml("prompts.yml")

# Initialize components
context = Context()
generator = Generator(llm=llm, prompts=prompts)
reflector = Reflector(llm=llm, prompts=prompts)
curator = Curator(llm=llm, prompts=prompts)

# Generate and improve
output = generator.generate(context, task_input)
reflection = reflector.reflect(context, task_input, output)
context = curator.curate(context, reflection)
```

## Examples

See the `examples/` directory for complete examples:

- **FINER Example** (`examples/finer/`): Financial Named Entity Recognition demonstrating:
  - Offline context adaptation from training data
  - Grow-and-refine with semantic de-duplication
  - OpenAI embeddings for duplicate detection
  - Lazy refinement mode with max_bullets limit
  - Context persistence and loading with embeddings

Run the example:
```bash
cd examples/finer
python finer_example.py
```

## Architecture

ACE follows a modular abstraction pattern:

- **LLM Abstraction** (`ace.llm.BaseLLM`): Supports multiple LLM providers (OpenAI, Anthropic, local models)
- **Embedder Abstraction** (`ace.embeddings.BaseEmbedder`): Supports multiple embedding backends (OpenAI, Transformers)
- **Context** (`ace.Context`): Structured, evolving knowledge base with grow-and-refine
- **ACE Components**: Generator, Reflector, and Curator work with any LLM/embedder combination

This design allows easy switching between providers and models without changing application code.

## Requirements

Python 3.8 or higher

OpenAI API key (set in environment or .env file)

Dependencies:
- `openai` - For LLM and embeddings
- `numpy` - For embedding operations
- `pyyaml` - For prompt management
- `python-dotenv` - For environment variables

## Development Notes

This project was developed using Cursor with Claude 4.5 Sonnet. The generated code required careful human review and multiple corrections. 

Examples of issues that were identified and fixed:

* Incorrect assumptions about LLM models and the OpenAI API (using old models like 3.5 and old API definitions).
* Fabricated context bloat that served no purpose like additional metadata and variables that made no sense and are not present in the paper.
* Repetitive and ineffective testing patterns.
* Poorly designed prompts for the Reflector component, the first prompt created was ridiculous although the prompt examples from the papers were provided.
* Over-specialized components tied to FINER that needed refactoring into a general approach using kwargs. The code was only looking for specific variables that wouldn't be useful for other use cases. It also failed to see the obvious common logic between the 2 usecases in the paper like the output format of the component, which I manually standardized.
* Unnecessary limits and truncation logic added without justification which caused problems in experimentation.
* Incorrect license specification in the documentation, even though the license doc is in the root path of the project.
* Inefficient O(n²) implementation of semantic de-duplication using nested loops with individual cosine similarity calls, instead of vectorized matrix operations (np.dot) that compute all pairwise similarities at once, resulting in 10-100x performance improvement.

These issues underscore the importance of human oversight when working with AI-generated code, even with state-of-the-art models. It is also questionable how much more productive I was with the help of cursor, as compared to me writing the codebase with small to little AI involvement, considering how much code I had to re-write. In my opinion, it was probably 25-50% faster vs coding without any llm help, which is still a decent improvement.
## Citation

This implementation is based on the paper:

```
Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models
Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, 
Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, 
Urmish Thakker, James Zou, Kunle Olukotun
arXiv:2510.04618, 2024
```

Paper: https://arxiv.org/abs/2510.04618

## License

Apache License 2.0

