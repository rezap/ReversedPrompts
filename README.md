# ReversedPrompts

An agentic flow to guess prompts that generate an output from an input set.

Given pairs of `(input files, output artifacts)`, recover the prompt that maps inputs to
outputs — including the style, structure, and content-selection rules that a naive
instruction like *"summarize this document"* would miss — then execute that recovered
prompt against new inputs using agentic RAG.

📄 **[Design & implementation plan](docs/DESIGN.md)**

## Status

Design phase. No implementation yet.
